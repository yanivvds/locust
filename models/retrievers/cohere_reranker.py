import httpx
import os
import time

from collections import OrderedDict
from dotenv import load_dotenv
from typing import List

load_dotenv()

from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever
from odata_graph import engine
from s_expression import Table

logger = logging.getLogger(__name__)


class AzureCohereReranker(object):
    def __init__(self):
        self.model_name = "Cohere-rerank-v4.0-pro"

    def rerank(self, question: str, documents: List, k: int) -> dict:
        """Call Cohere Rerank endpoint, retrying up to 3 times on transient errors.

        429 rate-limit responses use a longer backoff (30s / 60s / 90s) to let
        the Azure quota window reset. Other errors use a shorter backoff (5s / 10s / 15s).
        """
        payload = {
            "model": self.model_name,
            "query": question,
            "documents": documents,
            "top_n": k,
        }
        headers = {
            "api-key": os.environ["AZURE_FOUNDRY_KEY"],
            "Content-Type": "application/json",
        }
        for attempt in range(3):
            with httpx.Client(timeout=30, headers=headers) as client:
                response = client.post(
                    "https://tzv-meta.services.ai.azure.com/providers/cohere/v2/rerank",
                    json=payload,
                )
            if response.status_code == 200:
                return response.json()
            is_rate_limit = response.status_code == 429
            wait = 30 * (attempt + 1) if is_rate_limit else 5 * (attempt + 1)
            logger.warning(
                f"Cohere API returned {response.status_code}: {response.text[:200]} "
                f"(attempt {attempt + 1}/3, retrying in {wait}s)"
            )
            time.sleep(wait)
        raise RuntimeError(f"Cohere API failed after 3 attempts: {response.status_code} {response.text[:200]}")

    def rerank_tables(self, question: str, tables: dict, k: int = 5) -> dict:
        """
        Rerank retrieved tables using Cohere.

        :param question: natural language question
        :param tables: retriever output dict {table_id: {score, nodes}}
        :param k: number of tables to return after reranking
        :return: dict in the same {table_id: {score, nodes}} format, reranked and trimmed to top k
        """
        if not tables:
            return {}

        table_ids = list(tables.keys())
        table_titles = engine.get_table_titles([str(Table(tid).uri) for tid in table_ids])

        documents = []
        for tid in table_ids:
            title = table_titles.get(tid, {}).get('title', tid)
            description = table_titles.get(tid, {}).get('description', '')
            documents.append(f"{title}: {description}" if description else title)

        data = self.rerank(question=question, documents=documents, k=k)

        reranked = {}
        for result in data['results'][:k]:
            tid = table_ids[result['index']]
            reranked[tid] = {
                **tables[tid],
                'score': result['relevance_score'],
            }

        return reranked


class CohereTableRetriever(BaseRetriever):
    """Two-stage table retriever: ColBERT or RRF (recall) + Cohere Rerank v4 (precision).

    Uses the Azure AI Foundry Cohere Rerank v4.0-pro endpoint to rerank a
    first-stage candidate pool. Runs fully via API — no GPU or vLLM needed.

    First stage is controlled by use_rrf:
      - use_rrf=False  → ColBERTRetriever          (experiment C1-a)
      - use_rrf=True   → HybridRRFRetriever (B5-a)  (experiment C1-b)

    Requires AZURE_FOUNDRY_KEY environment variable (or set in config.py).

    Usage:
        env PYTHONPATH=. LANGUAGE=en AZURE_FOUNDRY_KEY=<key> \\
            python3 evaluation/evaluate_table_retrieval.py \\
            --model_path models/retrievers/cohere_reranker.py \\
            --query_type sql --k 10 \\
            --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
            --collection_variant kg_units \\
            --retrieval_k 20 \\
            --output_path evaluation/results/table_retrieval/c1a_cohere_colbert.json

        # With RRF first stage (C1-b):
        env ... --use_rrf True --rrf_k 60 \\
            --output_path evaluation/results/table_retrieval/c1b_cohere_rrf.json
    """

    def __init__(
        self,
        checkpoint: str,
        collection_variant: str = None,
        retrieval_k: int = 20,
        use_rrf: bool = False,
        rrf_k: int = 60,
        request_delay: float = 3.0,
    ):
        super().__init__()
        self.retrieval_k = int(retrieval_k)
        self.request_delay = float(request_delay)
        self.reranker = AzureCohereReranker()

        use_rrf = use_rrf is True or str(use_rrf).lower() in ('true', '1')
        if collection_variant is True:
            collection_variant = None

        if use_rrf:
            from models.retrievers.hybrid_rrf_retriever import HybridRRFRetriever
            self.first_stage = HybridRRFRetriever(
                checkpoint=checkpoint,
                collection_variant=collection_variant,
                retrieval_k=int(retrieval_k),
                rrf_k=int(rrf_k),
            )
        else:
            self.first_stage = ColBERTRetriever(
                checkpoint=checkpoint,
                mode='table',
                collection_variant=collection_variant,
            )

    def retrieve_tables(self, query: str, k: int) -> OrderedDict:
        pool_k = max(self.retrieval_k, k)
        candidates = self.first_stage.retrieve_tables(query, k=pool_k)

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        try:
            reranked = self.reranker.rerank_tables(query, candidates, k=k)
            # Pad with any candidates Cohere dropped (keeps top-k count guarantee)
            seen = set(reranked.keys())
            rank = len(reranked) + 1
            for tid in candidates:
                if len(reranked) >= k:
                    break
                if tid not in seen:
                    reranked[tid] = {'score': 1.0 / rank}
                    rank += 1
            return OrderedDict(list(reranked.items())[:k])
        except Exception as e:
            logger.warning(f"Cohere rerank failed: {e} — falling back to first-stage order")
            return OrderedDict(list(candidates.items())[:k])


if __name__ == "__main__":
    # Test the Cohere reranker pro hosted within Azure AI Foundry
    import pprint

    reranker = AzureCohereReranker()
    rank_response = reranker.rerank(
        question="What are the health benefits of green tea?",
        documents=[
                "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
                "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
                "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
                "Basketball is one of the most popular sports in the United States.",
                "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
                "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
            ],
        k=3
    )

    pprint.pp(rank_response.json(), indent=4)
