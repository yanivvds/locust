import argparse
import json
import os
import re

# ColBERT JIT-compiles CUDA C++ extensions during indexing, which requires CUDA_HOME (nvcc +
# headers). This check must happen before importing colbert or torch — once those are imported
# CUDA is already initialised and setting CUDA_VISIBLE_DEVICES has no effect.
# If CUDA_HOME is absent (e.g. Snellius with only the runtime loaded, not the full toolkit),
# hide all GPUs now so ColBERT/Faiss/PyTorch fall back to CPU throughout.
if not os.environ.get('CUDA_HOME') and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

from collections import OrderedDict
from itertools import islice
from typing import Literal, Dict

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

BASE_PATH = f"{config.PATH_DIR_DATA}/colbert_retriever"


class ColBERTRetriever(BaseRetriever):
    pid_node_map: Dict[int, str]

    def __init__(self,
            checkpoint: str,
            mode: Literal['all', 'table', 'node'] = 'table',
            nbits: int = 4,
            search_depth: int = 2_500,
            override_index: bool = False,
            collection_variant: str = None,
            kg_rerank: bool = False,
            kg_rerank_alpha: float = 0.5):
        """
            :param checkpoint: Directory containing model checkpoint. Can be:
                               - A directory name relative to {BASE_PATH} (e.g. 'my_model')
                               - An absolute local path (e.g. '/data/models/my_model')
                               - A HuggingFace repo ID (e.g. 'org/model-name'), which will be
                                 downloaded and cached automatically
            :param mode: (Sub)set of nodes to train model for
            :param nbits: Number of bits used in quantization per dimension for storing the index.
                          Should be an integer of [1, 2, 4, 8].
            :param search_depth: depth of search space when calling the ColBERT model. Only relevant if mode is in
                                 'all' or 'nodes', otherwise set to k during retrieval.
            :param override_index: always creates a new index if true, regardless if an index already exists.
            :param kg_rerank: if True, apply KG schema-overlap reranker after ColBERT ranking. Loads
                              node label tokens from the cached node_labels JSON at init time.
            :param kg_rerank_alpha: weight for the KG overlap bonus added to each ColBERT score.
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.mode = mode
        self.collection_variant = collection_variant

        # Explicit casts required when passed through model args in evaluation script
        self.nbits = int(nbits)
        self.search_depth = int(search_depth)
        self.kg_rerank = bool(kg_rerank)
        self.kg_rerank_alpha = float(kg_rerank_alpha)
        self.table_label_tokens: Dict[str, set] = {}

        if not config.IS_UNIT_TESTING:
            self.index_colbert(override_index)
            if self.kg_rerank:
                self._load_label_tokens()

    @staticmethod
    def _build_table_label_tokens(node_labels: dict) -> Dict[str, set]:
        """
        Build a per-table token set from measure/dimension node prefLabels (body as fallback).
        Table-type entries are excluded — we want schema label vocabulary, not table descriptions.

        :param node_labels: dict loaded from cbs-en_node_labels.json; keys are 'table_id' or
                            'table_id#node_id'; values have 'type', 'prefLabel', 'body', 'table'.
        :return: {table_id: set(lowercase_word_tokens)}
        """
        table_tokens: Dict[str, set] = {}
        for key, data in node_labels.items():
            if data.get('type') == 'table':
                continue
            table_id = data.get('table') or key.split('#')[0]
            text = data.get('prefLabel') or data.get('body', '')
            tokens = set(re.findall(r'\w+', text.lower()))
            if table_id not in table_tokens:
                table_tokens[table_id] = set()
            table_tokens[table_id].update(tokens)
        return table_tokens

    @staticmethod
    def _compute_kg_overlap(
            query: str,
            ranked_items: OrderedDict,
            table_label_tokens: Dict[str, set],
            alpha: float) -> OrderedDict:
        """
        Re-score and re-sort ranked_items by adding alpha * token_overlap to each table's score.
        Uses 'combined_score' as the base if present (set by aggregate_table_node_scores in 'all'
        mode), falling back to 'score' (the raw ColBERT score in table mode).

        Each entry is copied before modification — the input dict is not mutated.

        :param query: the natural language question
        :param ranked_items: OrderedDict from retrieve_tables before final islice
        :param table_label_tokens: per-table token sets from _build_table_label_tokens
        :param alpha: weight for overlap bonus (0.0 = no reranking)
        :return: re-sorted OrderedDict with 'combined_score' set on each entry
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        reranked = {}
        for table_id, data in ranked_items.items():
            label_tokens = table_label_tokens.get(table_id, set())
            overlap = len(query_tokens & label_tokens) / max(len(query_tokens), 1)
            base = data.get('combined_score', data.get('score', 0))
            reranked[table_id] = dict(data) | {'combined_score': base + alpha * overlap}
        return OrderedDict(sorted(reranked.items(), key=lambda x: x[1]['combined_score'], reverse=True))

    @staticmethod
    def aggregate_table_node_scores(
            ranked_items: Dict[str, Dict[str, float]],
            alpha: float = 0.8,
            node_score_threshold: float = .01) -> OrderedDict:
        """
            Construct an ordered dictionary from the pool of returned tables and nodes when running in 'all' mode.
            This function aggregates the table and corresponding node scores above a threshold and consolidates this
            into a ranked dictionary. Inner node dictionaries are NOT sorted.

            :param ranked_items: ranked items from the ColBERT model. Dictionary looks something like
                                 `{'85055ENG': {'score': 41.6875}, '84708ENG#2031210': {'score': 41.46875}, '85251ENG': {'score': 41.46875}}`
            :param alpha: weighted mean factor for combined table scores with corresponding node scores
            :param node_score_threshold: threshold for node score aggregation. Nodes below this threshold are ignored.
            :return: OrderedDict with ranked tables and corresponding node scores.
        """
        table_scores = {}
        for item_id, score in ranked_items.items():
            if '#' not in item_id:  # table node
                if item_id in table_scores:
                    table_scores[item_id]['score'] = score['score']
                    continue
                table_scores[item_id] = score | {'nodes': {}}
            else:  # measure/dim node
                if score['score'] < node_score_threshold:
                    continue
                ids_ = item_id.split('#', 1)
                table_id = ids_[0]
                node_id = ids_[1]

                if table_id not in table_scores:
                    table_scores[table_id] = {'score': 0, 'nodes': {}}
                table_scores[table_id]['nodes'][node_id] = score

        # Calculate combined table scores and format for sorting
        for table_id, data in table_scores.items():
            nodes_scores = sum(node['score'] for node in data['nodes'].values())
            if data['nodes']:
                mean_node_score = nodes_scores / len(data['nodes'])
                combined_score = data['score'] + alpha * mean_node_score
            else:
                combined_score = data['score']
            data['combined_score'] = combined_score

        # Sort tables by the new combined score in descending order
        sorted_tables = OrderedDict(sorted(table_scores.items(), key=lambda item: item[1]['combined_score'], reverse=True))
        return sorted_tables

    def retrieve_tables(self, query: str, k: int) -> OrderedDict:
        """
            Return an ordered and ranked dictionary of tables. If running the 'table' mode, only
            tables are returned in the dictionary with their respective score. When running in
            'all' mode, also relevant nodes above a thresholded score for each table is returned,
            and the table score is a weighted aggregation of its own and node scores.

            If kg_rerank is enabled, an additional KG schema-overlap bonus is applied after ColBERT
            ranking and before the final top-k slice.

            :param query: the natural language question
            :param k: number of tables to return
            :return: ordered dictionary of tables and nodes
        """
        ranking = self.searcher.search(query, k=self.search_depth if self.mode in ['all', 'nodes'] else k)
        ranked_items = {self.pid_node_map[pid]: {
            'score': ranking[2][idx]
        } for idx, pid in enumerate(ranking[0])}

        if self.mode in ['all', 'nodes']:
            ranked_items = self.aggregate_table_node_scores(ranked_items)

        if self.kg_rerank and self.table_label_tokens:
            ranked_items = self._compute_kg_overlap(query, ranked_items, self.table_label_tokens, self.kg_rerank_alpha)

        return OrderedDict(islice(ranked_items.items(), k))

    def _resolve_checkpoint(self) -> str:
        """
            Resolve the checkpoint to a local directory path.

            Supports three formats:
                - Relative path: resolved against BASE_PATH (e.g. 'my_model' -> '{BASE_PATH}/my_model')
                - Absolute path: used as-is (e.g. '/data/models/my_model')
                - HuggingFace repo ID: downloaded and cached (e.g. 'org/model-name')

            :return: absolute local path to the checkpoint directory
        """
        # Absolute local path
        if os.path.isabs(self.checkpoint) and os.path.isdir(self.checkpoint):
            return self.checkpoint

        # Relative local path under BASE_PATH
        local_path = f"{BASE_PATH}/{self.checkpoint}"
        if os.path.isdir(local_path):
            return local_path

        # Treat as HuggingFace repo ID
        logger.info(f"Checkpoint '{self.checkpoint}' not found locally, attempting HuggingFace download...")
        from huggingface_hub import snapshot_download
        cached_path = snapshot_download(repo_id=self.checkpoint)
        logger.info(f"Downloaded checkpoint to {cached_path}")
        return cached_path

    @staticmethod
    def _data_dir(checkpoint: str) -> str:
        """Return the shared data directory for collection and mapping files."""
        return f"{BASE_PATH}/{checkpoint.replace('/', '_')}"

    def _load_label_tokens(self):
        """
        Load the node_labels JSON cache and build per-table token sets for the KG overlap reranker.
        If the cache is missing, disables kg_rerank with a warning rather than crashing — the
        retriever still works, just without the reranker.
        """
        data_dir = self._data_dir(self.checkpoint)
        labels_path = f"{data_dir}/{config.GRAPH_DB_REPO}_node_labels.json"
        if not os.path.exists(labels_path):
            logger.warning(
                f"Node labels cache not found at {labels_path}; "
                f"KG overlap reranker disabled. Run colbert_trainer.py --rebuild_collection first."
            )
            self.kg_rerank = False
            return
        with open(labels_path) as f:
            node_labels = json.load(f)
        self.table_label_tokens = self._build_table_label_tokens(node_labels)
        logger.info(f"KG reranker loaded: {len(self.table_label_tokens)} tables, alpha={self.kg_rerank_alpha}")

    def index_colbert(self, override_index: bool = False):
        """
            Create a ColBERT index. Requires a collection_<mode>.tsv and node_pid_map.json file for
            indexing the documents. This can be created using the create_train_set function in the
            colbert_training.py script.

            Collection and node_pid_map files are read from the shared data directory
            ({BASE_PATH}/{checkpoint_name}/), decoupled from the model checkpoint itself.
        """
        # Colbert-specific imports are deferred to here so the module can be imported (and its
        # static methods used in tests) without requiring the colbert package to be installed.
        import models.retrievers.colbert.patch_colbert  # noqa: F401 — applies monkey-patches on import
        from colbert import Searcher, Indexer
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from torch.utils.cpp_extension import verify_ninja_availability
        verify_ninja_availability()

        model_path = self._resolve_checkpoint()
        experiment_name = self.checkpoint.replace('/', '_')
        data_dir = self._data_dir(self.checkpoint)
        variant_suffix = f"_{self.collection_variant}" if self.collection_variant else ""
        mode_key = f"{self.mode}{variant_suffix}"
        collection_path = f"{data_dir}/collection_{mode_key}.tsv"
        node_pid_map_path = f"{data_dir}/collection_{mode_key}_node_pid_map.json"

        if not os.path.exists(collection_path):
            raise FileNotFoundError(f"A collection file must exist for indexing ColBERT! Missing {collection_path}")
        if not os.path.exists(node_pid_map_path):
            raise FileNotFoundError(f"A node-PID-mapping file must exist for running ColBERT! Missing {node_pid_map_path}")

        index_dir = f"./experiments/{experiment_name}/indexes/{mode_key}.nbits={self.nbits}/centroids.pt"
        overwrite_index = 'reuse' if os.path.exists(index_dir) and not override_index else True
        # ColBERT JIT-compiles CUDA C++ extensions during indexing, which requires CUDA_HOME
        # (nvcc + headers). If only the CUDA runtime is present (no toolkit), hide the GPU so
        # ColBERT falls back to CPU and skips the JIT compilation step entirely.
        # Also use avoid_fork_if_possible=True so ColBERT runs in-process (single process)
        # instead of spawning a subprocess — spawned subprocesses crash when CUDA_VISIBLE_DEVICES=''
        # because they fail to initialise the distributed backend.
        with Run().context(RunConfig(nranks=1, avoid_fork_if_possible=True, experiment=experiment_name)):
            colbert_config = ColBERTConfig(nbits=self.nbits, root=".", index_bsize=32)

            if overwrite_index != 'reuse':
                logger.info("Creating ColBERTv2 index...")
                indexer = Indexer(checkpoint=model_path, config=colbert_config)
                indexer.index(name=f"{mode_key}.nbits={self.nbits}",
                              overwrite=overwrite_index,
                              collection=collection_path)
            else:
                logger.info(f"Loading ColBERTv2 index from path {index_dir}...")

            self.searcher = Searcher(index=f"{mode_key}.nbits={self.nbits}",
                                     checkpoint=model_path,
                                     collection=collection_path,
                                     config=colbert_config)

        with open(node_pid_map_path, 'r') as f:
            node_pid_map = json.load(f)
            self.pid_node_map = {v: k for k, v in node_pid_map.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ColBERTv2 evaluation')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help=f"Model checkpoint: local directory (relative to {BASE_PATH} or absolute path), "
                             f"or a HuggingFace repo ID (e.g. 'org/model-name').")
    parser.add_argument('--mode', type=str, choices=['table', 'node', 'all'], default='all',
                        help='(Sub)set of nodes to that model was trained on.')
    parser.add_argument('--nbits', type=int, choices=[1, 2, 4, 8], default=4,
                        help='Number of bits used in quantization per dimension for storing the index.')
    parser.add_argument('--kg_rerank', action='store_true',
                        help='Enable KG schema-overlap reranker post-retrieval.')
    parser.add_argument('--kg_rerank_alpha', type=float, default=0.5,
                        help='Weight for KG overlap bonus (default: 0.5).')

    args = parser.parse_args()

    retriever = ColBERTRetriever(
        checkpoint=args.checkpoint,
        mode=args.mode,
        nbits=args.nbits,
        kg_rerank=args.kg_rerank,
        kg_rerank_alpha=args.kg_rerank_alpha,
    )

    q = "How many people went on vacation to Germany in 2020?"
    top_tables = retriever.retrieve_tables(query=q, k=10)
    print(top_tables)
