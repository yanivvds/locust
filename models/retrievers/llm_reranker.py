import os
import re
import time
from collections import OrderedDict

from openai import OpenAI

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever
from odata_graph import engine
from s_expression import Table

logger = logging.getLogger(__name__)


class LLMTableReranker(BaseRetriever):
    """Two-stage table retriever: ColBERT (recall) + LLM reranker (precision).

    Retrieves a larger candidate pool with ColBERT, then uses an LLM served via
    either a vLLM OpenAI-compatible API (Snellius) or the OpenAI API directly.

    Usage (Snellius, vLLM server must be running):
        env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
            --model_path models/retrievers/llm_reranker.py \\
            --query_type sql --k 10 \\
            --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
            --collection_variant kg_units \\
            --model google/gemma-4-31B-it \\
            --base_url http://localhost:8000/v1 \\
            --retrieval_k 20 \\
            --output_path evaluation/results/table_retrieval_llm_reranker_kg_units_en_results.json

    Usage (OpenAI API, local):
        env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
            python3 evaluation/evaluate_table_retrieval.py \\
            --model_path models/retrievers/llm_reranker.py \\
            --query_type sql --k 10 \\
            --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
            --collection_variant kg_units \\
            --backend openai \\
            --model gpt-5.4-mini \\
            --retrieval_k 20 \\
            --output_path evaluation/results/table_retrieval/c2a_openai_colbert.json
    """

    def __init__(
        self,
        checkpoint: str,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        retrieval_k: int = 20,
        max_tokens: int = 256,
        collection_variant: str = None,
        enable_thinking: bool = False,
        backend: str = 'vllm',
    ):
        super().__init__()
        self.retrieval_k = int(retrieval_k)
        self.max_tokens = int(max_tokens)
        # Guard: argparse flag-only args may arrive as True rather than a string
        if collection_variant is True:
            collection_variant = None
        self.collection_variant = collection_variant
        self.enable_thinking = enable_thinking is True or str(enable_thinking).lower() == 'true'
        self.backend = str(backend).lower()

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )

        if self.backend == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when backend='openai'")
            self.ml_client = OpenAI(api_key=api_key, timeout=600)
        else:
            self.ml_client = OpenAI(api_key='vllm', base_url=base_url, timeout=600)

        self.model_name = model

        prompt_path = os.path.join(
            os.path.dirname(__file__),
            f"{config.LANGUAGE}_reranker_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()

        # Split on [SYSTEM] / [USER] markers; parts[0] is empty, [1] = system, [2] = user
        parts = re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve_tables(self, query: str, k: int) -> OrderedDict:
        pool_k = max(self.retrieval_k, k)
        candidates = self.colbert.retrieve_tables(query, k=pool_k)
        table_ids = list(candidates.keys())

        try:
            table_titles = engine.get_table_titles(
                [str(Table(tid).uri) for tid in table_ids]
            )
        except Exception:
            table_titles = {}

        table_info = self._build_table_info(table_ids, table_titles)
        user_prompt = self.user_prompt_template.format(
            question=query,
            table_info=table_info,
            k=k,
        )

        raw_response = self._call_llm(user_prompt)

        valid_ids_upper = {tid.upper(): tid for tid in table_ids}
        ordered_ids = self._parse_response(raw_response, valid_ids_upper, table_ids)

        return self._build_result(ordered_ids, k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        if self.backend == 'openai':
            system_content = self.system_prompt
            # gpt-5.x models require max_completion_tokens; older models use max_tokens
            create_kwargs = {"max_completion_tokens": self.max_tokens}
        else:
            system_content = self.system_prompt if self.enable_thinking else "/no_think\n" + self.system_prompt
            create_kwargs = {
                "max_tokens": self.max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            }

        for attempt in range(3):
            try:
                response = self.ml_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.model_name,
                    **create_kwargs,
                )
                response_data = response.choices[0].message.content
                if '</think>' in response_data:
                    response_data = response_data.split('</think>')[-1].strip()
                return response_data
            except Exception as e:
                wait = 10 * (attempt + 1)
                logger.warning(
                    f"LLM call failed ({self.backend}): {e} (attempt {attempt + 1}/3, retrying in {wait}s)"
                )
                time.sleep(wait)
        return ""

    # ------------------------------------------------------------------
    # Static methods (tested independently, no ColBERT/vLLM needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, valid_ids_upper: dict, fallback_ids: list) -> list:
        """Parse LLM output into an ordered list of valid table IDs.

        :param raw: raw LLM response text
        :param valid_ids_upper: {UPPER_ID: original_id} for all candidates in the pool
        :param fallback_ids: original ColBERT-ordered IDs — used for padding and as
                             fallback when the LLM output is entirely unparseable
        :return: ordered list of original table IDs, padded to full pool size
        """
        seen = set()
        result = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading numbering: "1. ", "2) ", "3: ", etc.
            line = re.sub(r'^\d+[\.\):]\s*', '', line)
            # Take first token (split on whitespace or colon)
            token = re.split(r'[\s:]+', line)[0].strip()
            if not token:
                continue
            upper = token.upper()
            if upper not in valid_ids_upper:
                continue
            orig = valid_ids_upper[upper]
            if orig in seen:
                continue
            seen.add(orig)
            result.append(orig)

        if not result:
            return list(fallback_ids)

        # Pad with any candidates the LLM didn't mention, in original ColBERT order
        for tid in fallback_ids:
            if tid not in seen:
                result.append(tid)

        return result

    @staticmethod
    def _build_table_info(table_ids: list, titles: dict) -> str:
        """Build the numbered table listing injected into the prompt.

        :param table_ids: ordered list of table IDs (ColBERT order)
        :param titles: {table_id: {'title': ..., 'description': ...}}
        :return: newline-joined string, one table per line
        """
        lines = []
        for i, tid in enumerate(table_ids):
            meta = titles.get(tid, {})
            title = meta.get('title', tid)
            desc = meta.get('description', '')
            if desc and desc != title:
                line = f"{i + 1}. {tid}: {title} — {desc[:120]}"
            else:
                line = f"{i + 1}. {tid}: {title}"
            lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def _build_result(ordered_ids: list, k: int) -> OrderedDict:
        """Build the retriever result dict with reciprocal-rank scores.

        :param ordered_ids: reranked list of table IDs (most relevant first)
        :param k: number of results to return
        :return: OrderedDict {table_id: {'score': 1/(rank+1)}} sliced to k
        """
        return OrderedDict(
            (ordered_ids[i], {'score': 1.0 / (i + 1)})
            for i in range(min(k, len(ordered_ids)))
        )
