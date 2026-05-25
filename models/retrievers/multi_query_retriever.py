"""
Multi-Query ColBERT Retriever with LLM Question Expansion.

Literature basis:
    Gao et al. (2022/2023). "Precise Zero-Shot Dense Retrieval without Relevance
    Labels." NAACL 2023. (HyDE — Hypothetical Document Embeddings)

    Wang et al. (2023). "Query2Doc: Query Expansion with Large Language Models."
    EMNLP 2023.

    Shi et al. (2024). RAG-Fusion — multiple query variants fused with RRF.

Key distinction from the CRUSH4SQL hallucination approach (which was tried and
failed -7.3pp on LOCuST): CRUSH4SQL generates a fake *schema* as the retrieval
query. This component generates *question paraphrases* instead — the question
vocabulary is varied so that different SKOS label clusters in the dense index are
activated.  ColBERT already encodes the schema; the gap it misses is vocabulary
mismatch between the question and the measure/dimension labels in the table
document.

Prompt design (three variants per question):
    1. Original question (always included as variant 0)
    2. CBS-formalized version: rewrite using official statistical terminology
    3. Synonym/keyword-expanded version: surface domain synonyms and entity expansions

Usage (Snellius, requires vLLM server running with Gemma 4 31B):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/multi_query_retriever.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --model google/gemma-4-31B-it \\
        --base_url http://localhost:8000/v1 \\
        --n_variants 3 \\
        --per_query_k 20 \\
        --output_path evaluation/results/table_retrieval/b6_multiquery_n3_k20.json

Experiment matrix (B6-a through B6-c):
    B6-a: n_variants=3, per_query_k=20
    B6-b: n_variants=5, per_query_k=15
    B6-c: n_variants=3, per_query_k=20, enable_thinking=False (already default)
"""
import json
import re
from collections import OrderedDict, defaultdict

from openai import OpenAI

from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """/no_think
You are a query expansion assistant for a Statistics Netherlands (CBS) text-to-SQL \
retrieval system. The retrieval index contains ~2,244 CBS statistical table documents \
enriched with SKOS dimension/measure labels and QUDT unit strings.

Given a Dutch-statistics question, return a JSON array of {n} alternative phrasings \
designed to maximise recall over the table index. Rules:
- Element 0: the original question, copied verbatim.
- Element 1: rewrite using formal CBS/statistical terminology \
  (e.g. "how many" → "what is the count of", trade sector names in official CBS form).
- Element 2 onward: expand domain synonyms and named entities \
  (e.g. "companies" → "enterprises, establishments, firms", region names in full).
- Keep each variant to one sentence.
- Return ONLY the JSON array, no commentary or markdown fences."""


class MultiQueryRetriever(BaseRetriever):
    """ColBERT retriever that expands each question into N variants with an LLM,
    then fuses the per-variant ranked lists via Reciprocal Rank Fusion.

    The LLM call adds ~0.3-0.5 s per question (no-think mode, Gemma 4 31B).
    On parse failure the retriever falls back to a single ColBERT call on the
    original question.
    """

    def __init__(
        self,
        checkpoint: str,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        n_variants: int = 3,
        per_query_k: int = 20,
        collection_variant: str = 'kg_units',
        rrf_k: int = 60,
        max_tokens: int = 256,
    ):
        """
        :param checkpoint: ColBERT checkpoint (HuggingFace repo ID or local path).
        :param model: vLLM-served model name for question expansion.
        :param base_url: vLLM OpenAI-compatible API base URL.
        :param n_variants: number of question variants to generate (including original).
        :param per_query_k: ColBERT candidate pool size per variant.
        :param collection_variant: ColBERT collection variant (e.g. 'kg_units').
        :param rrf_k: RRF smoothing constant (paper default: 60).
        :param max_tokens: maximum tokens for the expansion LLM call.
        """
        super().__init__()
        self.n_variants = int(n_variants)
        self.per_query_k = int(per_query_k)
        self.rrf_k = int(rrf_k)
        self.max_tokens = int(max_tokens)
        self.model = model

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )
        self.client = OpenAI(api_key='vllm', base_url=base_url, timeout=60)

    def _generate_variants(self, question: str) -> list[str]:
        """Call the LLM to expand question into self.n_variants paraphrases.

        Returns a list starting with the original question. On any failure
        (parse error, empty response, network error) returns [question] so
        the caller can fall back to single-query ColBERT.
        """
        system = _SYSTEM_PROMPT.replace('{n}', str(self.n_variants))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                max_tokens=self.max_tokens,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"LLM expansion failed ({exc}); falling back to single query")
            return [question]

        # Strip ```json or ```sql fences that Gemma 4 sometimes emits
        raw = re.sub(r'^```(?:json|sql)?\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```\s*$', '', raw)
        # Strip stray </think> leakage
        raw = re.sub(r'</think>\s*', '', raw)

        try:
            variants = json.loads(raw)
            if not isinstance(variants, list) or not variants:
                raise ValueError("not a non-empty list")
            # Ensure original is always first, clamp to n_variants
            variants = [str(v) for v in variants]
            if variants[0].strip() != question.strip():
                variants.insert(0, question)
            return variants[:self.n_variants]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"LLM expansion parse failed ({exc}); raw={raw[:120]!r}; falling back")
            return [question]

    def retrieve_tables(self, question: str, k: int = 10) -> OrderedDict:
        """Retrieve top-k tables by running ColBERT on N question variants and
        fusing results with Reciprocal Rank Fusion.

        :param question: natural-language question
        :param k: number of tables to return
        :return: OrderedDict {table_id: {'score': rrf_score}}
        """
        variants = self._generate_variants(question)
        logger.debug(f"Generated {len(variants)} variants: {variants}")

        scores: dict[str, float] = defaultdict(float)
        for variant in variants:
            results = self.colbert.retrieve_tables(variant, k=self.per_query_k)
            for rank, table_id in enumerate(results):
                scores[table_id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return OrderedDict(
            (tid, {'score': scores[tid]}) for tid in sorted_ids[:k]
        )
