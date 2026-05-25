"""Side-quest retriever: CRUSH4SQL-style query hallucination as a pre-retrieval step.

NOT part of the main thesis pipeline. See sidequests/README.md.

Wraps a ColBERTRetriever. For each question, asks an LLM (Gemma 4 31B via vLLM by
default) to hallucinate a `CREATE TABLE` block; that block is then sent to ColBERT
as the retrieval query in place of the raw question. Hallucinations are cached to
JSON so repeat runs are free.

Two prompt variants are supported:
- 'vanilla': generic DDL hallucination, no domain priming
- 'kg': adds an RDF Data Cube + CBS-style dimension/measure label primer

Usage (Snellius, vLLM server must be running):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/sidequests/query_hallucination_retriever.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --prompt_variant kg \\
        --base_url http://localhost:8000/v1 \\
        --model google/gemma-4-31B-it \\
        --output_path evaluation/results/sidequests/table_retrieval_hallucination_kg_en.json
"""
import json
import os
import re
import time
from collections import OrderedDict

from openai import OpenAI

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever

logger = logging.getLogger(__name__)

PROMPT_VARIANTS = ('vanilla', 'kg')
DEFAULT_CACHE_DIR = 'evaluation/results/sidequests'


class QueryHallucinationRetriever(BaseRetriever):

    def __init__(
        self,
        checkpoint: str,
        prompt_variant: str = 'kg',
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        max_tokens: int = 384,
        collection_variant: str = None,
        cache_path: str = None,
    ):
        super().__init__()

        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_variant must be one of {PROMPT_VARIANTS}, got {prompt_variant!r}"
            )
        self.prompt_variant = prompt_variant
        self.max_tokens = int(max_tokens)

        # argparse flag-only args may arrive as True rather than a string
        if collection_variant is True:
            collection_variant = None
        self.collection_variant = collection_variant

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )
        self.ml_client = OpenAI(api_key='vllm', base_url=base_url, timeout=120)
        self.model_name = model

        prompt_path = os.path.join(
            os.path.dirname(__file__),
            f"{config.LANGUAGE}_hallucination_{prompt_variant}_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()
        parts = re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

        self.cache_path = cache_path or self._default_cache_path()
        self.cache = self._load_cache(self.cache_path)
        logger.info(
            f"QueryHallucinationRetriever ready: variant={prompt_variant}, "
            f"model={model}, cache={self.cache_path} ({len(self.cache)} entries)"
        )

    def retrieve_tables(self, query: str, k: int) -> OrderedDict:
        hallucinated = self._get_or_hallucinate(query)
        # If hallucination is empty (LLM failed all retries), fall back to raw question
        # so retrieval still produces a result rather than crashing the eval loop.
        retrieval_query = hallucinated if hallucinated else query
        return self.colbert.retrieve_tables(retrieval_query, k=k)

    def _get_or_hallucinate(self, question: str) -> str:
        if question in self.cache:
            return self.cache[question]
        hallucinated = self._call_llm(question)
        self.cache[question] = hallucinated
        self._save_cache()
        return hallucinated

    def _call_llm(self, question: str) -> str:
        user_prompt = self.user_prompt_template.format(question=question)
        for attempt in range(3):
            try:
                response = self.ml_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "/no_think\n" + self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = response.choices[0].message.content or ""
                if '</think>' in raw:
                    raw = raw.split('</think>')[-1]
                return self._clean_ddl(raw)
            except Exception as e:
                wait = 10 * (attempt + 1)
                logger.warning(
                    f"vLLM hallucination call failed: {e} "
                    f"(attempt {attempt + 1}/3, retrying in {wait}s)"
                )
                time.sleep(wait)
        logger.error(
            f"All hallucination attempts failed for question: {question[:80]!r}"
        )
        return ""

    @staticmethod
    def _clean_ddl(raw: str) -> str:
        """Strip markdown fences and surrounding whitespace from the LLM output.

        Some models wrap DDL in ```sql ... ``` even when told not to. We don't try
        to validate the SQL — ColBERT just embeds it as text — but we do strip the
        fences so the embedding isn't dominated by the backtick noise.
        """
        text = raw.strip()
        # Drop markdown code fences if present
        text = re.sub(r'^```(?:sql|SQL)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()

    def _default_cache_path(self) -> str:
        model_slug = self.model_name.replace('/', '_').replace(':', '_')
        return os.path.join(
            DEFAULT_CACHE_DIR,
            f"hallucination_cache_{config.LANGUAGE}_{self.prompt_variant}_{model_slug}.json",
        )

    @staticmethod
    def _load_cache(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        # Atomic write: tmp + rename so a crash mid-write can't corrupt the cache.
        tmp_path = self.cache_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.cache_path)
