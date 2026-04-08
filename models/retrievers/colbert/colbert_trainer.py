import argparse
import difflib
import itertools
import json
import locale
import numpy as np
import os
import pandas as pd
import pickle
import regex as re
import shutil
from colbert import Trainer, Indexer, Searcher
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from math import ceil
from operator import itemgetter
from rdflib.namespace import QB
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.cpp_extension import verify_ninja_availability
from tqdm import tqdm
from typing import Literal, Dict, List, Optional

# Patch for ColBERT training issues on Windows and Mac
import colbert.infra.launcher
from odata_graph import engine
from models.retrievers.colbert.patch_colbert import patched_setup_new_process

colbert.infra.launcher.setup_new_process = patched_setup_new_process

import config
from logs import logging
from utils.global_functions import parse_for_table_id

verify_ninja_availability()

logger = logging.getLogger(__name__)


class ColBERTTrainer(object):
    RANDOM_STATE = 42
    SPLIT = 0.1

    def __init__(self,
                 output_name: str,
                 mode: Literal['all', 'table', 'node'],
                 model_name: str = 'BAAI/bge-m3',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 negative_n: int = 10,
                 learning_rate: float = 5e-6,
                 batch_size: int = 8,
                 azure: bool = False,
                 skip_create_dataset: bool = False,
                 num_rounds: int = 1,
                 hn_checkpoint: Optional[str] = None,
                 search_k: int = 100,
                 mix_ratio: float = 0.0,
                 nbits: int = 4,
                 rebuild_only: bool = False):
        """
            Manages the full ColBERT training pipeline: dataset creation, negative sampling, and
            (optionally) iterative hard negative mining across multiple training rounds.

            On construction, loads the QA data from config.TRAIN_QA_FILE and creates a table-stratified
            train/test split to prevent data leakage. Also initializes the embedding model used for
            cosine-similarity-based negative sampling in round 1.

            Training modes:
                - Single-round (num_rounds=1, default): creates a training set with cosine-sim negatives
                  from the embedding model, then trains a ColBERT model. Equivalent to the legacy flow.
                - Iterative (num_rounds>1): round 1 uses cosine-sim negatives; subsequent rounds index
                  the previous round's checkpoint with ColBERT, retrieve each training query, and select
                  the highest-scoring false positives as hard negatives for retraining.
                - With hn_checkpoint: skips round 1's cosine-sim step entirely and mines hard negatives
                  from an existing checkpoint from the first round onwards.

            :param output_name: name for the output directory under data/colbert_retriever/
            :param mode: which document types to include in the collection
                         ('table' = tables only, 'node' = measures/dimensions only, 'all' = both)
            :param model_name: HuggingFace model name or local path for the base checkpoint
            :param embedding_model_name: SentenceTransformer model used for cosine-sim negative sampling
            :param negative_n: number of negatives to sample per (query, positive) pair
            :param learning_rate: learning rate for ColBERT fine-tuning
            :param batch_size: training batch size
            :param azure: if True, use Azure OpenAI API for embeddings instead of SentenceTransformers
            :param skip_create_dataset: skip dataset creation when collection/triples already exist on disk
            :param num_rounds: number of training rounds (1 = single-round, 2+ = iterative hard negative mining)
            :param hn_checkpoint: path to an existing ColBERT checkpoint to mine hard negatives from in round 1,
                                  skipping the initial cosine-sim training. Must contain collection, queries,
                                  and node_pid_map files.
            :param search_k: number of results to retrieve per query during hard negative mining
            :param mix_ratio: fraction of negatives to retain from the previous round's triples
                              (0.0 = all model-mined, 0.5 = half old / half new)
            :param nbits: quantization bits for the ColBERT index during hard negative mining (1, 2, 4, or 8)
            :param rebuild_only: if True, only initialize paths — skip loading QA data and embedding model.
                                 Used when the trainer is only needed for rebuild_collection().
        """
        self.mode = mode
        self.model_name = model_name
        self.negative_n = negative_n
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.azure = azure
        self.skip_create_dataset = skip_create_dataset
        self.num_rounds = num_rounds
        self.hn_checkpoint = hn_checkpoint
        self.search_k = search_k
        self.mix_ratio = mix_ratio
        self.nbits = nbits

        # Paths
        output_name = output_name.replace('/', '_')
        self.base_path = f"{config.PATH_DIR_DATA}/colbert_retriever/{output_name}"
        self.triples_path = f"{self.base_path}/triples_{mode}.jsonl"
        self.queries_path = f"{self.base_path}/queries_{mode}.tsv"
        self.collection_path = f"{self.base_path}/collection_{mode}.tsv"

        if rebuild_only:
            # Minimal init — only paths are needed for rebuild_collection()
            self.embedding_model = None
            self.azure_client = None
            self.embedding_model_name = embedding_model_name
            self.queries = None
            self.train_df = None
            self.test_df = None
            logger.info(
                f"=== ColBERT Trainer initialized (rebuild-only mode) ==="
                f"\n\tOutput directory:    {self.base_path}"
                f"\n\tMode:                {mode}"
            )
            return

        # Embedding model — only needed for cosine-sim negative sampling (round 1 without hn_checkpoint)
        self.embedding_model = None
        self.azure_client = None
        self.embedding_model_name = embedding_model_name
        needs_embedding_model = hn_checkpoint is None and not skip_create_dataset
        if needs_embedding_model:
            if azure:
                from openai import AzureOpenAI
                self.azure_client = AzureOpenAI(
                    api_version=config.AZURE_API_VERSION,
                    azure_endpoint=config.AZURE_ENDPOINT,
                    api_key=config.AZURE_KEY,
                )
            else:
                self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load QA data and create train/test split
        assert os.path.exists(config.TRAIN_QA_FILE)
        with open(config.TRAIN_QA_FILE, 'r') as outfile:
            self.queries = [json.loads(l) for l in outfile.read().splitlines()]

        # Stratified split by primary table to prevent data leakage
        query_groups = [parse_for_table_id(q['sexp'], 'sexp')[0] for q in self.queries]
        gss = GroupShuffleSplit(n_splits=1, test_size=self.SPLIT, random_state=self.RANDOM_STATE)
        train_idx, test_idx = next(gss.split(self.queries, groups=query_groups))
        self.train_df = [self.queries[i] for i in train_idx]
        self.test_df = [self.queries[i] for i in test_idx]

        train_tables = {query_groups[i] for i in train_idx}
        test_tables = {query_groups[i] for i in test_idx}
        overlap = train_tables & test_tables

        # Log configuration summary
        if num_rounds > 1 or hn_checkpoint is not None:
            training_strategy = f"iterative hard negative mining ({num_rounds} round(s))"
            if hn_checkpoint:
                training_strategy += f", starting from checkpoint: {hn_checkpoint}"
        else:
            training_strategy = "single-round with cosine-sim negatives"

        embedding_source = "Azure OpenAI" if azure else embedding_model_name
        logger.info(
            f"=== ColBERT Trainer initialized ==="
            f"\n\tOutput directory:    {self.base_path}"
            f"\n\tBase model:          {model_name}"
            f"\n\tMode:                {mode}"
            f"\n\tTraining strategy:   {training_strategy}"
            f"\n\tNegative samples:    {negative_n} per (query, positive) pair"
            f"\n\tLearning rate:       {learning_rate}"
            f"\n\tBatch size:          {batch_size}"
            f"\n\tEmbedding model:     {embedding_source}{' (skipped)' if not needs_embedding_model else ''}"
            f"\n\tSearch depth (k):    {search_k}"
            f"\n\tMix ratio:           {mix_ratio}"
            f"\n\tQuantization (nbits): {nbits}"
            f"\n\tData split:          {len(self.train_df)} train / {len(self.test_df)} test, "
            f"{len(train_tables)} train tables / {len(test_tables)} test tables, "
            f"{len(overlap)} overlapping (should be 0)"
        )

    def get_graph_node_labels(self, include_time_geo_dims: bool = False) -> Dict[str, Dict[str, str]]:
        """
            Return a dictionary with all the textual elements for every table and measure/dimension in the graph.
        """
        node_labels_path = f"{self.base_path}/{config.GRAPH_DB_REPO}_node_labels{{}}.json"
        node_labels_path = node_labels_path.format('_including_time_geo' if include_time_geo_dims else '')
        if os.path.isfile(node_labels_path):
            logger.info(f"Loading node labels from {node_labels_path}")
            with open(node_labels_path) as f:
                node_labels = json.load(f)
                return node_labels

        tables = {}
        nodes = {}

        # Fetch all tables in the graph, including entries containing title and descriptions from the tables
        table_query = ("""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            PREFIX dct: <http://purl.org/dc/terms/>

            SELECT DISTINCT ?id ?label WHERE {{
                ?s a dcat:Dataset .
                ?s dct:identifier ?id .
                OPTIONAL {{ ?s dct:title|dct:abstract|dct:description ?label }}
            }}
        """)

        try:
            fetch_tables = engine.select(table_query)
            props = [(r['id']['value'], (r.get('label', False) or {'value': ''})['value']) for r in fetch_tables]
            table_it = itertools.groupby(props, itemgetter(0))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch table IDs: {e}")

        for table, table_props in tqdm(table_it, total=len(set(map(itemgetter(0), props))),
                                       desc="Fetching graph nodes for all tables", bar_format=config.TQDM_BAR_FMT):
            val = ' '.join(f"{prop[1].strip()}{'' if prop[1][-1] == '.' else '.'}" for prop in table_props)
            tables[table] = {'body': val, 'type': 'table'}

            # Get all measures and dimensions for a table
            query = (f"""
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                PREFIX dct: <http://purl.org/dc/terms/>
                PREFIX qb: <http://purl.org/linked-data/cube#>

                SELECT ?node_id ?type ?label WHERE {{
                    ?table ?type ?node .
                    FILTER (?type = qb:measure || ?type = qb:dimension) .
                    {'''
                    FILTER NOT EXISTS {
                        VALUES ?timegeo {'TimeDimension' 'GeoDimension'}
                        ?node a ?timegeo.
                    }
                    ''' if not include_time_geo_dims else ''}
                    ?table dct:identifier ?table_id .
                    ?node qb:concept ?concept ;
                          dct:identifier ?node_id .
                    ?concept dct:isPartOf ?table .
                    OPTIONAL {{ ?concept skos:prefLabel|skos:altLabel|skos:definition|dct:description|dct:subject ?label }}
                    FILTER (?table_id = "{table}")
                }}
            """)

            try:
                result = engine.select(query)
                props = [(r['node_id']['value'], r['type']['value'], (r.get('label', False) or {'value': ''})['value'])
                         for r in result]
                node_it = itertools.groupby(props, itemgetter(0))
                for node_id, node_props in node_it:
                    node_props = list(node_props)
                    node_type = node_props[0][1]
                    if node_type == str(QB.measure):
                        type_ = 'measure'
                    elif node_type == str(QB.dimension):
                        type_ = 'dimension'
                    else:
                        raise ValueError(f"Unknown node type: {node_type}")

                    val = ' '.join(f"{prop[2].strip()}{'' if not prop[2] or prop[2][-1] == '.' else '.'}" for prop in node_props)
                    # Because nodes can have different concepts per table, store them using unique identifiers per table
                    nodes[f"{table}#{node_id}"] = {'body': val, 'type': type_, 'table': table}
            except Exception as e:
                logger.error(f"Failed to fetch table nodes: {e}")

        nodes.update(tables)
        os.makedirs(os.path.dirname(node_labels_path), exist_ok=True)
        with open(node_labels_path, 'w') as f:
            json.dump(nodes, f)

        logger.info(f"Successfully saved node properties to {node_labels_path}")
        return nodes

    @staticmethod
    def build_enriched_collection(
            nodes: Dict[str, Dict[str, str]],
            max_child_labels: int = 15) -> Dict[str, Dict[str, str]]:
        """
            Enrich document representations for the ColBERT collection by cross-pollinating
            information between tables and their child measures/dimensions.

            For tables: appends a structured summary of child measure and dimension labels to the body.
            For measures/dimensions: prepends the parent table's title to the body.

            :param nodes: raw output from get_graph_node_labels
            :param max_child_labels: max number of child labels to append per type to a table body
                                     (to stay within ColBERT's doc_maxlen)
            :return: A new nodes dict with enriched 'body' fields
        """
        tables = {k: v for k, v in nodes.items() if v['type'] == 'table'}
        children = {k: v for k, v in nodes.items() if v['type'] != 'table'}

        # Group children by parent table
        table_children: Dict[str, Dict[str, List[str]]] = {}
        for node_id, node_data in children.items():
            table_id = node_id.split('#')[0]
            if table_id not in table_children:
                table_children[table_id] = {'measure': [], 'dimension': []}
            table_children[table_id][node_data['type']].append(node_data['body'])

        # Extract table titles (first sentence of body, before the first period)
        table_titles = {}
        for table_id, table_data in tables.items():
            title = table_data['body'].split('.')[0].strip()
            table_titles[table_id] = title

        enriched = {}

        # Enrich table documents: append child measure and dimension labels
        for table_id, table_data in tables.items():
            body = table_data['body']
            tc = table_children.get(table_id, {'measure': [], 'dimension': []})

            msr_labels = [b.split('.')[0].strip() for b in tc['measure']][:max_child_labels]
            dim_labels = [b.split('.')[0].strip() for b in tc['dimension']][:max_child_labels]

            suffix_parts = []
            if msr_labels:
                suffix_parts.append(f"Measures: {', '.join(msr_labels)}")
            if dim_labels:
                suffix_parts.append(f"Dimensions: {', '.join(dim_labels)}")

            if suffix_parts:
                body = f"{body} {'. '.join(suffix_parts)}."

            enriched[table_id] = {**table_data, 'body': body}

        # Enrich child node documents: prepend parent table title
        for node_id, node_data in children.items():
            table_id = node_id.split('#')[0]
            title = table_titles.get(table_id, '')
            if title:
                body = f"[{title}] {node_data['body']}"
            else:
                body = node_data['body']

            enriched[node_id] = {**node_data, 'body': body}

        logger.info(f"Enriched {len(tables)} table(s) and {len(children)} node(s)")
        return enriched

    def _embed(self, text: str):
        """Embed a single text using the configured embedding model."""
        if self.azure:
            return np.array(self.azure_client.embeddings.create(
                input=[text], model=self.embedding_model_name
            ).data[0].embedding)
        else:
            return self.embedding_model.encode(text, show_progress_bar=False)

    def _build_positives(self, node_pid_map: dict, dataset: list = None) -> Dict[int, dict]:
        """
            Build a dict mapping query PIDs to their question text and positive document PIDs,
            using the given node-to-PID mapping.

            :param node_pid_map: mapping from node/table IDs to collection PIDs
            :param dataset: list of QA dicts to process (defaults to self.train_df)
        """
        if dataset is None:
            dataset = self.train_df

        positives = {}
        for i, r in enumerate(dataset):
            golden_tables = parse_for_table_id(r['sexp'], 'sexp')
            golden_msrs = re.findall(r"(?:\(MSR\s\(\s*(?=[^)])|\G(?!^)\s+)(\w+)", r['sexp'])
            golden_dims = re.findall(r"(?:\(DIM\s\S+\s\(\s*(?=[^)])|\G(?!^)\s+)(\w+)", r['sexp'])

            q_nodes = [node_pid_map[t] for t in golden_tables if t in node_pid_map]
            for n in golden_msrs + golden_dims:
                for t in golden_tables:
                    node_id = f"{t}#{n}"
                    if node_id in node_pid_map:
                        q_nodes.append(node_pid_map[node_id])

            positives[i] = {
                'query': r['question'],
                'positives': q_nodes,
            }
        return positives

    def _mine_cosine_sim_negatives(self, train_questions: Dict[int, dict], collection: pd.DataFrame):
        """
            Mine negative samples using cosine similarity from the embedding model.
            For each (query, positive) pair, selects negative_n negatives by alternating between the
            most and least similar candidates in the embedding space.

            :param train_questions: output of _build_positives() for the training split
            :param collection: DataFrame with 'body' and 'embedding' columns
        """
        if os.path.exists(self.triples_path):
            os.remove(self.triples_path)

        for qpid, r in tqdm(train_questions.items(), desc='Selecting negatives...'):
            query_embedding = self._embed(r['query'])

            for positive in r['positives']:
                # Get n negative nodes per query partially based on cos sim of embeddings
                too_similar = collection[collection.body.isin(
                    difflib.get_close_matches(collection.loc[positive]['body'], collection['body'], cutoff=0.95)
                )]

                negative_candidates = collection[~collection.index.isin(
                    set(r['positives'] + list(too_similar.index.values))
                )]
                similarities = cosine_similarity(query_embedding.reshape(1, -1), np.vstack(negative_candidates['embedding'].values))[0]
                sorted_indices = np.argsort(similarities)

                for i in range(self.negative_n):
                    negative = (sorted_indices[::-1] if i % 2 == 0 else sorted_indices)[ceil(i / 2)]
                    with open(self.triples_path, 'a+', newline='', encoding=locale.getpreferredencoding(), errors='ignore') as triple_tsv:
                        triple_tsv.write(f"[{qpid}, {positive}, {negative}]\n")

    def _mine_hard_negatives_from_model(self, checkpoint_path: str):
        """
            Mine hard negatives using a trained ColBERT model's own retrieval errors.
            Indexes the collection with the given checkpoint, searches for each training query,
            and selects the highest-scoring non-positive results as hard negatives.

            :param checkpoint_path: path to the ColBERT checkpoint directory (must also contain
                                    collection, queries, and node_pid_map files)
        """
        collection_path = f"{checkpoint_path}/collection_{self.mode}.tsv"
        queries_path = f"{checkpoint_path}/queries_{self.mode}_train.tsv"
        triples_path = f"{checkpoint_path}/triples_{self.mode}.jsonl"
        node_pid_map_path = f"{checkpoint_path}/collection_{self.mode}_node_pid_map.json"

        for required_file in [collection_path, queries_path, node_pid_map_path]:
            if not os.path.exists(required_file):
                raise FileNotFoundError(f"Required file for hard negative mining not found: {required_file}")

        with open(node_pid_map_path, 'r') as f:
            node_pid_map = json.load(f)

        positives_per_query = {
            i: {**data, 'positives': set(data['positives'])}
            for i, data in self._build_positives(node_pid_map).items()
        }

        # Read collection size for random negative fallback
        collection_df = pd.read_csv(collection_path, sep='\t', header=None, names=['body'],
                                    encoding=locale.getpreferredencoding())
        all_pids = set(range(len(collection_df)))

        # Index the collection using the checkpoint
        logger.info(f"Indexing collection for hard negative mining with checkpoint: {checkpoint_path}")
        index_name = f"{self.mode}.nbits={self.nbits}.hn_mining"
        with Run().context(RunConfig(nranks=1, experiment=os.path.basename(checkpoint_path))):
            colbert_config = ColBERTConfig(nbits=self.nbits, root=os.path.dirname(checkpoint_path), index_bsize=32)
            indexer = Indexer(checkpoint=checkpoint_path, config=colbert_config)
            index_path = f"./experiments/{os.path.basename(checkpoint_path)}/indexes/{index_name}/centroids.pt"
            overwrite = 'reuse' if os.path.exists(index_path) else True
            indexer.index(name=index_name, overwrite=overwrite, collection=collection_path)
            searcher = Searcher(index=index_name, config=colbert_config)

        # Load existing triples for mixing if needed
        old_triples = {}
        if self.mix_ratio > 0 and os.path.exists(triples_path):
            with open(triples_path, 'r', encoding=locale.getpreferredencoding()) as f:
                for line in f:
                    triple = json.loads(line)
                    key = (triple[0], triple[1])
                    if key not in old_triples:
                        old_triples[key] = []
                    old_triples[key].append(triple[2])

        # Mine hard negatives
        new_triples = []
        for qpid, q_data in tqdm(positives_per_query.items(), desc='Mining hard negatives...'):
            ranking = searcher.search(q_data['query'], k=self.search_k)
            ranked_pids = ranking[0]

            # Filter out positives to get hard negative candidates (ordered by model confidence)
            hard_neg_candidates = [pid for pid in ranked_pids if pid not in q_data['positives']]

            for positive in q_data['positives']:
                n_old = int(np.floor(self.negative_n * self.mix_ratio)) if self.mix_ratio > 0 else 0
                n_new = self.negative_n - n_old

                # Select hard negatives from model
                selected_negatives = hard_neg_candidates[:n_new]

                # Pad with random negatives if not enough hard negatives found
                if len(selected_negatives) < n_new:
                    logger.warning(f"Query {qpid}: only found {len(selected_negatives)}/{n_new} hard negatives, "
                                   f"padding with random negatives")
                    available_random = list(all_pids - q_data['positives'] - set(selected_negatives))
                    n_pad = n_new - len(selected_negatives)
                    selected_negatives.extend(np.random.choice(available_random, size=min(n_pad, len(available_random)),
                                                               replace=False).tolist())

                # Mix in old negatives if requested
                if n_old > 0:
                    key = (qpid, positive)
                    old_negs = old_triples.get(key, [])
                    selected_old = old_negs[:n_old]
                    # If not enough old negatives, fill with more model-mined ones
                    if len(selected_old) < n_old:
                        extra = n_old - len(selected_old)
                        selected_negatives = hard_neg_candidates[:n_new + extra]
                    selected_negatives = selected_old + selected_negatives

                for neg_pid in selected_negatives:
                    new_triples.append([qpid, positive, neg_pid])

        # Backup previous triples file
        if os.path.exists(triples_path):
            round_num = 1
            while os.path.exists(triples_path.replace('.jsonl', f'_round_{round_num}.jsonl')):
                round_num += 1
            backup_path = triples_path.replace('.jsonl', f'_round_{round_num}.jsonl')
            shutil.copy2(triples_path, backup_path)
            logger.info(f"Backed up previous triples to {backup_path}")

        # Write new triples
        with open(triples_path, 'w', newline='', encoding=locale.getpreferredencoding(), errors='ignore') as f:
            for triple in new_triples:
                f.write(json.dumps(triple) + '\n')

        logger.info(f"Mined {len(new_triples)} hard negative triples and saved to {triples_path}")

    def rebuild_collection(self, include_time_geo_dims: bool = False):
        """
            Rebuild the collection TSV and node_pid_map from the current state of the SPARQL graph.
            This can be used independently of training to update the collection when new tables or
            nodes are added to the graph. After rebuilding, any existing ColBERT index should be
            re-created by re-instantiating the ColBERTRetriever.

            Invalidates the pickle cache and removes stale index directories to ensure a clean state.

            :param include_time_geo_dims: whether to include time and geo dimensions in the collection
            :return: tuple of (collection DataFrame, node_pid_map dict)
        """
        pk_file = self.collection_path.replace('.tsv', '.pk')

        # Invalidate pickle cache to force a fresh fetch from the graph
        if os.path.exists(pk_file):
            logger.info(f"Removing stale collection cache: {pk_file}")
            os.remove(pk_file)

        # Also invalidate the cached node labels so they are re-fetched from the graph
        for suffix in ['', '_including_time_geo']:
            labels_path = f"{self.base_path}/{config.GRAPH_DB_REPO}_node_labels{suffix}.json"
            if os.path.exists(labels_path):
                logger.info(f"Removing stale node labels cache: {labels_path}")
                os.remove(labels_path)

        # Fetch all nodes from the graph and enrich
        nodes = self.get_graph_node_labels(include_time_geo_dims=include_time_geo_dims)
        nodes = self.build_enriched_collection(nodes)
        nodes = {k: v for k, v in nodes.items() if self.mode == 'all' or v['type'] == self.mode}

        collection = pd.DataFrame.from_dict({
            i: doc | {'embedding': None}
            for i, (id_, doc) in tqdm(enumerate(nodes.items()), total=len(nodes), desc='Building collection...')
        }, orient='index')
        collection['id'] = nodes.keys()
        collection['body'] = collection['body'].str.replace(r'(((\r|\\r)?(\n|\\n))|\s+)', ' ', regex=True)

        os.makedirs(os.path.dirname(self.collection_path), exist_ok=True)

        # Save pickle cache for subsequent use by create_train_set
        with open(pk_file, 'wb') as f:
            pickle.dump(collection, f)

        if self.mode != 'all':
            collection = collection[collection['type'] == self.mode]
            collection.reset_index(drop=True, inplace=True)

        # Write TSV to disk => doc_pid \t body
        collection.to_csv(self.collection_path,
                          columns=['body'], header=False, sep='\t',
                          encoding=locale.getpreferredencoding(), errors='ignore')
        node_pid_map = {v['id']: k for k, v in collection.iterrows()}
        with open(self.collection_path.replace('.tsv', '_node_pid_map.json'), 'w') as f:
            json.dump(node_pid_map, f)

        # Remove stale ColBERT indexes that were built against the old collection
        index_dir = f"./experiments/{os.path.basename(self.base_path)}/indexes"
        if os.path.isdir(index_dir):
            logger.info(f"Removing stale indexes: {index_dir}")
            shutil.rmtree(index_dir)

        logger.info(f"Rebuilt collection: {len(collection)} documents, "
                    f"saved to {self.collection_path}")
        return collection, node_pid_map

    def create_train_set(self, include_time_geo_dims: bool = False):
        """
            Create a training dataset for the ColBERTv2 training. The training set consists of three separate files:
                * collection | tsv: pid, passage
                * queries | tsv: pid, passage
                * triples | jsonl: query, positive passage, negative passage
            The triples file contains `negative_n` samples of a query pid, the positive node/table and a selected
            negative node/table, based on a sampling done using the cosine similarity of the embedded description.
        """
        pk_file = self.collection_path.replace('.tsv', '.pk')

        if not os.path.exists(pk_file):
            # No cached collection — build it from the graph
            self.rebuild_collection(include_time_geo_dims=include_time_geo_dims)

        collection = pd.read_pickle(pk_file)
        if collection['embedding'].isnull().any():
            # Embed the descriptions if not done yet
            embeddings = {}
            for i, doc in tqdm(collection.iterrows(), total=len(collection), desc='Encoding collection...'):
                if doc['body'] in embeddings:
                    continue
                embeddings[doc['body']] = self._embed(doc['body'])

            collection['embedding'] = collection['body'].map(embeddings)
            with open(pk_file, 'wb') as f:
                pickle.dump(collection, f)

        if self.mode != 'all':
            collection = collection[collection['type'] == self.mode]
            collection.reset_index(drop=True, inplace=True)

        # Load the node_pid_map written by rebuild_collection
        with open(self.collection_path.replace('.tsv', '_node_pid_map.json'), 'r') as f:
            node_pid_map = json.load(f)

        # Validate that all tables and nodes referenced by QA pairs are present in the collection.
        # Missing entries mean the model can never retrieve them, silently hurting evaluation scores.
        # This might mean that the graph is incomplete
        missing_tables = set()
        missing_nodes = set()
        for r in self.queries:  # queries contains ALL QA pairs (train + test)
            golden_tables = parse_for_table_id(r['sexp'], 'sexp')
            golden_msrs = re.findall(r"(?:\(MSR\s\(\s*(?=[^)])|\G(?!^)\s+)(\w+)", r['sexp'])
            golden_dims = re.findall(r"(?:\(DIM\s\S+\s\(\s*(?=[^)])|\G(?!^)\s+)(\w+)", r['sexp'])

            for t in golden_tables:
                if t not in node_pid_map:
                    missing_tables.add(t)
                for n in golden_msrs + golden_dims:
                    node_id = f"{t}#{n}"
                    if node_id not in node_pid_map:
                        missing_nodes.add(node_id)

        if missing_tables and self.mode != 'node':
            logger.warning(f"{len(missing_tables)} table(s) referenced in QA data are missing from the collection: "
                           f"{missing_tables}")
        if missing_nodes and self.mode != 'table':
            logger.warning(f"{len(missing_nodes)} node(s) referenced in QA data are missing from the collection. "
                           f"Examples: {list(missing_nodes)[:10]}")

        # Build positives for train and test splits, and write query TSVs
        for split, ds in [('train', self.train_df), ('test', self.test_df)]:
            questions = self._build_positives(node_pid_map, dataset=ds)
            query_tsv = [(k, v['query']) for k, v in questions.items()]
            query_df = pd.DataFrame(query_tsv, columns=['i', 'q']).set_index('i', drop=True)

            path = self.queries_path.replace('.tsv', f"_{split}.tsv")
            query_df.to_csv(path, columns=['q'], header=False, sep='\t',
                            encoding=locale.getpreferredencoding(), errors='ignore')

        # Mine negatives using cosine similarity of the embedding model
        train_questions = self._build_positives(node_pid_map)
        self._mine_cosine_sim_negatives(train_questions, collection)

    @staticmethod
    def recommend_colbert_config(model_checkpoint: str, collection_path: str):
        """
        Analyze the base model and collection to recommend doc_maxlen and dim values.

        Tokenizes all documents in the collection using the base model's tokenizer to find
        the token length distribution, and reads the model's hidden size to inform dim.

        If the checkpoint is a previously trained ColBERT model (contains artifact.metadata),
        the dim from that checkpoint is used to avoid shape mismatches in the projection layer.

        :param model_checkpoint: HuggingFace model name or local path for the base model
        :param collection_path: path to the collection TSV file
        :return: dict with recommended 'doc_maxlen' and 'dim' values
        """
        from transformers import AutoTokenizer, AutoConfig

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Read collection documents from TSV (format: pid \t body)
        docs = pd.read_csv(collection_path, sep='\t', header=None, names=['body'],
                           encoding=locale.getpreferredencoding())
        token_lengths = [len(tokenizer.encode(str(doc), add_special_tokens=True))
                         for doc in tqdm(docs['body'], desc='Tokenizing collection for length analysis')]
        token_lengths = np.array(token_lengths)

        p50 = int(np.percentile(token_lengths, 50))
        p95 = int(np.percentile(token_lengths, 95))
        p99 = int(np.percentile(token_lengths, 99))
        max_len = int(token_lengths.max())

        # Round p99 up to the nearest power of 2 for ColBERT efficiency
        recommended_doc_maxlen = int(2 ** np.ceil(np.log2(p99)))

        # Determine dim: respect existing checkpoint dim if present, otherwise use hidden_size
        model_config = AutoConfig.from_pretrained(model_checkpoint)
        hidden_size = model_config.hidden_size

        checkpoint_metadata = os.path.join(model_checkpoint, 'artifact.metadata')
        if os.path.exists(checkpoint_metadata):
            with open(checkpoint_metadata) as f:
                checkpoint_dim = json.load(f).get('dim', hidden_size)
            recommended_dim = checkpoint_dim
            dim_source = f"from checkpoint artifact.metadata (trained with dim={checkpoint_dim})"
        else:
            recommended_dim = hidden_size
            dim_source = "full hidden size (fresh base model)"

        logger.info(f"=== ColBERT config recommendations for '{model_checkpoint}' ==="
                    f"\n\tCollection size: {len(docs)} documents"
                    f"\n\tToken length distribution: median={p50}, p95={p95}, p99={p99}, max={max_len}"
                    f"\n\tRecommended doc_maxlen: {recommended_doc_maxlen} (covers 99% of documents, "
                    f"truncates {(token_lengths > recommended_doc_maxlen).sum()}/{len(docs)})"
                    f"\n\tModel hidden size: {hidden_size}"
                    f"\n\tRecommended dim: {recommended_dim} ({dim_source})")
        if recommended_dim != hidden_size:
            logger.warning(f"To train with dim={hidden_size}, use the original base model as checkpoint")
        if recommended_dim > 256:
            logger.warning(f"Recommended dim is larger than 256 ({recommended_dim}). Currently this doesn't work for ColBERTv2. Using dim=256 with slight loss in model quality for now.")

        return {
            'doc_maxlen': recommended_doc_maxlen,
            'dim': recommended_dim,
            'token_length_stats': {'p50': p50, 'p95': p95, 'p99': p99, 'max': max_len},
        }

    def train(self, model_checkpoint: str, round_num: int = 0):
        """
            Train a ColBERT model.

            :param model_checkpoint: HuggingFace model name or local checkpoint path
            :param round_num: training round number (0 = single-round/legacy, 1+ = iterative round)
        """
        recommendations = self.recommend_colbert_config(model_checkpoint, self.collection_path)

        with Run().context(RunConfig(nranks=1, experiment=self.base_path)):
            # Note: regardless of given root, the model seems always to be saved in `experiments/<model_name>/**`
            accumsteps = min(self.batch_size, 16)
            colbert_cfg = ColBERTConfig(bsize=self.batch_size,
                                        lr=self.learning_rate,
                                        accumsteps=accumsteps,
                                        query_maxlen=64,
                                        doc_maxlen=recommendations['doc_maxlen'],
                                        dim=min(256, recommendations['dim']),
                                        save_every=1_000)
            q_path = self.queries_path.replace('.tsv', '_train.tsv')
            trainer = Trainer(triples=self.triples_path, queries=q_path,
                              collection=self.collection_path, config=colbert_cfg)

            trainer.train(checkpoint=model_checkpoint)
            checkpoint_path = trainer.best_checkpoint_path()

            if round_num > 0:
                # Save to round-specific directory
                round_dir = f"{self.base_path}/round_{round_num}"
                logger.info(f"Saved checkpoint to {checkpoint_path}, copying to round dir {round_dir}")
                shutil.copytree(checkpoint_path, round_dir, dirs_exist_ok=True)
                # Copy collection and node_pid_map into the round directory for subsequent mining
                shutil.copyfile(self.collection_path, f"{round_dir}/collection_{self.mode}.tsv")
                shutil.copyfile(self.collection_path.replace('.tsv', '_node_pid_map.json'),
                                f"{round_dir}/collection_{self.mode}_node_pid_map.json")
                # Copy queries into the round directory
                q_train_path = self.queries_path.replace('.tsv', '_train.tsv')
                q_test_path = self.queries_path.replace('.tsv', '_test.tsv')
                if os.path.exists(q_train_path):
                    shutil.copyfile(q_train_path, f"{round_dir}/queries_{self.mode}_train.tsv")
                if os.path.exists(q_test_path):
                    shutil.copyfile(q_test_path, f"{round_dir}/queries_{self.mode}_test.tsv")

            # Always copy to base_path as "latest" (preserves backward compat)
            logger.info(f"Copying checkpoint files to {self.base_path} as latest.")
            shutil.copytree(checkpoint_path, self.base_path, dirs_exist_ok=True)
            shutil.copyfile(self.collection_path, f"{self.base_path}/collection.tsv")
            shutil.copyfile(self.collection_path.replace('.tsv', '_node_pid_map.json'),
                            f"{self.base_path}/node_pid_map.json")

        logger.info(f"=== DONE TRAINING (round {round_num}) ===")

    def train_iterative(self):
        """
            Orchestrate multi-round iterative hard negative mining and training.
            Uses self.num_rounds, self.hn_checkpoint, self.search_k, self.negative_n,
            self.nbits, and self.mix_ratio for configuration.
        """
        os.makedirs(self.base_path, exist_ok=True)

        # When starting from an external checkpoint, copy shared dataset files into base_path
        # so that train() and recommend_colbert_config() can find them at the expected paths
        if self.hn_checkpoint is not None:
            hn_files = [
                (f"{self.hn_checkpoint}/collection_{self.mode}.tsv", self.collection_path),
                (f"{self.hn_checkpoint}/collection_{self.mode}_node_pid_map.json",
                 self.collection_path.replace('.tsv', '_node_pid_map.json')),
                (f"{self.hn_checkpoint}/queries_{self.mode}_train.tsv",
                 self.queries_path.replace('.tsv', '_train.tsv')),
                (f"{self.hn_checkpoint}/queries_{self.mode}_test.tsv",
                 self.queries_path.replace('.tsv', '_test.tsv')),
            ]
            for src, dst in hn_files:
                if os.path.exists(src) and src != dst:
                    shutil.copyfile(src, dst)
                    logger.info(f"Copied {src} -> {dst}")

        for round_num in range(1, self.num_rounds + 1):
            logger.info(f"=== Starting round {round_num}/{self.num_rounds} ===")

            if round_num == 1 and self.hn_checkpoint is not None:
                # Skip initial training — mine hard negatives from provided checkpoint
                logger.info(f"Mining hard negatives from provided checkpoint: {self.hn_checkpoint}")
                self._mine_hard_negatives_from_model(checkpoint_path=self.hn_checkpoint)
                # Copy triples from hn_checkpoint dir to base_path
                hn_triples = f"{self.hn_checkpoint}/triples_{self.mode}.jsonl"
                if hn_triples != self.triples_path:
                    shutil.copyfile(hn_triples, self.triples_path)
                self.train(model_checkpoint=self.hn_checkpoint, round_num=round_num)

            elif round_num == 1 and self.hn_checkpoint is None:
                # Round 1 without existing checkpoint: use cosine-sim negatives
                if not self.skip_create_dataset:
                    self.create_train_set()
                self.train(model_checkpoint=self.model_name, round_num=round_num)

            else:
                # Rounds 2+: mine hard negatives from previous round's checkpoint
                prev_round_dir = f"{self.base_path}/round_{round_num - 1}"
                logger.info(f"Mining hard negatives from round {round_num - 1} checkpoint: {prev_round_dir}")
                self._mine_hard_negatives_from_model(checkpoint_path=prev_round_dir)
                # Copy mined triples to base_path
                mined_triples = f"{prev_round_dir}/triples_{self.mode}.jsonl"
                if mined_triples != self.triples_path:
                    shutil.copyfile(mined_triples, self.triples_path)
                self.train(model_checkpoint=prev_round_dir, round_num=round_num)

        logger.info(f"=== Iterative training complete ({self.num_rounds} rounds). "
                    f"Latest checkpoint at {self.base_path} ===")

    def run(self):
        """Main entry point: dispatches to single-round or iterative training."""
        if self.num_rounds > 1 or self.hn_checkpoint is not None:
            self.train_iterative()
        else:
            if not self.skip_create_dataset:
                self.create_train_set()
            self.train(model_checkpoint=self.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ColBERTv2 training script')
    parser.add_argument('--skip_create_dataset', action='store_true',
                        help='Skip creating a new dataset (collection & triples) when already present.')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-m3',
                        help='Name/Huggingface handle of the starting checkpoint')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='Name/Huggingface handle of embedding model.')
    parser.add_argument('--azure', action='store_true',
                        help='Envoke the Azure AI/ML API on the given model_name for generating embeddings. '
                             'If False, the model will be retrieved from SentenceTransformers.')
    parser.add_argument('--negative_n', type=int, default=10,
                        help='Number of hard mined negatives per training same.')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Output name and dir of the trained model')
    parser.add_argument('--mode', type=str, choices=['table', 'node', 'all'], required=True,
                        help='(Sub)set of nodes to train model for')
    parser.add_argument('--learning_rate', '-lr', type=float, dest='learning_rate', default=5e-6,
                        help='Learning rate for the model trainer')
    parser.add_argument('--batch_size', '-b', type=int, dest='batch_size', default=8,
                        help='Batch size for the model trainer')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Number of training rounds. Round 1 uses cosine-sim negatives, '
                             'rounds 2+ mine from the trained model.')
    parser.add_argument('--hn_checkpoint', type=str, default=None,
                        help='Path to a pre-trained checkpoint to use for hard negative mining from round 1. '
                             'Skips initial cosine-sim training. The checkpoint directory must contain '
                             'the collection, queries, and node_pid_map files.')
    parser.add_argument('--search_k', type=int, default=100,
                        help='Search depth for hard negative mining.')
    parser.add_argument('--mix_ratio', type=float, default=0.0,
                        help='Fraction of negatives to keep from previous round (0.0 = all model-mined).')
    parser.add_argument('--nbits', type=int, choices=[1, 2, 4, 8], default=4,
                        help='Number of bits for quantization during hard negative mining indexing.')
    parser.add_argument('--rebuild_collection', action='store_true',
                        help='Rebuild the collection TSV and node_pid_map from the current SPARQL graph, '
                             'then exit. Does not train or require QA data. Useful when new tables/nodes '
                             'have been added to the graph and the index needs updating.')

    args = parser.parse_args()

    if args.rebuild_collection:
        trainer = ColBERTTrainer(
            output_name=args.output_name,
            mode=args.mode,
            rebuild_only=True,
        )
        trainer.rebuild_collection()
    else:
        trainer = ColBERTTrainer(
            output_name=args.output_name,
            mode=args.mode,
            model_name=args.model_name,
            embedding_model_name=args.embedding_model,
            negative_n=args.negative_n,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            azure=args.azure,
            skip_create_dataset=args.skip_create_dataset,
            num_rounds=args.num_rounds,
            hn_checkpoint=args.hn_checkpoint,
            search_k=args.search_k,
            mix_ratio=args.mix_ratio,
            nbits=args.nbits,
        )
        trainer.run()
