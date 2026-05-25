import os
import re
import time
from openai import OpenAI
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.kg_enriched_bm25_retriever import KGEnrichedBM25Retriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class VLLMBaselineSQLModel(BaseLLMGenerator):
    """Baseline SQL generator using a self-hosted vLLM server (OpenAI-compatible API)
    or the real OpenAI API.

    Switching between backends is automatic based on base_url:
      - localhost / 127.0.0.1        → vLLM mode (fake api_key, extra_body for thinking)
      - integrate.api.nvidia.com     → NVIDIA NIM mode (reads NVIDIA_API_KEY from env)
      - *.azure.com                  → Azure AI mode (reads AZURE_API_KEY from env)
      - anything else                → OpenAI mode (reads OPENAI_API_KEY from env)

    Pass reasoning_effort='low'|'medium'|'high'|'xhigh' to enable OpenAI reasoning
    tokens (uses the Responses API; ignored in vLLM and NIM modes).

    Usage (vLLM / BM25 retriever):
        env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_query_generation.py \\
            --model-path models/generators/llm_baseline/vllm_sql_model.py \\
            --query_type sql --task query-only \\
            --output_path evaluation/results/out.json \\
            --results_path evaluation/results/res.json \\
            --model Qwen/Qwen3.5-27B \\
            --base_url http://localhost:8000/v1

    Usage (OpenAI API with reasoning):
        ... same as above but:
            --model gpt-5.4-mini --base_url https://api.openai.com/v1
            --reasoning_effort low
    """

    def __init__(
        self,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        checkpoint: str = None,
        max_tokens: int = 2048,
        max_nodes_per_table: int = 500,
        collection_variant: str = None,
        reasoning_effort: Optional[str] = None,
        use_llm_reranker: bool = False,
        use_enriched_reranker: bool = False,
        retrieval_k: int = 20,
        reranker_model: str = 'gpt-5.4-mini',
        reranker_backend: str = 'openai',
    ):
        super().__init__()
        _use_llm_reranker = use_llm_reranker if isinstance(use_llm_reranker, bool) else str(use_llm_reranker).lower() in ('true', '1')
        _use_enriched_reranker = use_enriched_reranker if isinstance(use_enriched_reranker, bool) else str(use_enriched_reranker).lower() in ('true', '1')
        if checkpoint and _use_enriched_reranker:
            from models.retrievers.llm_enriched_reranker import LLMEnrichedReranker
            self.retriever = LLMEnrichedReranker(
                checkpoint=checkpoint,
                model=reranker_model,
                base_url=base_url if reranker_backend != 'openai' else 'https://api.openai.com/v1',
                retrieval_k=int(retrieval_k),
                collection_variant=collection_variant,
                backend=reranker_backend,
            )
        elif checkpoint and _use_llm_reranker:
            from models.retrievers.llm_reranker import LLMTableReranker
            self.retriever = LLMTableReranker(
                checkpoint=checkpoint,
                model=model,
                base_url=base_url,
                retrieval_k=int(retrieval_k),
                collection_variant=collection_variant,
            )
        elif checkpoint:
            self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table', collection_variant=collection_variant)
        else:
            self.retriever = KGEnrichedBM25Retriever()
        self.model_name = model
        self._is_vllm = 'localhost' in base_url or '127.0.0.1' in base_url
        self._is_nim = 'integrate.api.nvidia.com' in base_url
        self._is_azure = '.azure.com' in base_url and not self._is_vllm
        self._reasoning_effort = reasoning_effort if not self._is_vllm and not self._is_nim else None
        client_kwargs: dict = dict(timeout=600)
        if self._is_vllm:
            client_kwargs['api_key'] = 'vllm'
            client_kwargs['base_url'] = base_url
        elif self._is_nim:
            client_kwargs['api_key'] = os.environ.get('NVIDIA_API_KEY', '')
            client_kwargs['base_url'] = base_url
        elif self._is_azure:
            # Azure AI Services (OpenAI-compatible /openai/v1/ endpoint)
            client_kwargs['api_key'] = os.environ.get('AZURE_API_KEY', '')
            client_kwargs['base_url'] = base_url
        # For real OpenAI, let the client read OPENAI_API_KEY from env automatically
        self.ml_client = OpenAI(**client_kwargs)
        # For OpenAI reasoning models, max_completion_tokens covers both reasoning
        # and output tokens. Auto-scale upward when reasoning_effort is set so the
        # model doesn't exhaust its budget on thinking before writing the JSON.
        if self._reasoning_effort and not self._is_vllm:
            min_tokens = {'low': 4096, 'medium': 8192, 'high': 16384}.get(self._reasoning_effort, 8192)
            max_tokens = max(max_tokens, min_tokens)
        # NIM thinking models need headroom for reasoning tokens
        if self._is_nim:
            max_tokens = max(max_tokens, 16384)
        self.max_tokens = max_tokens
        self.max_nodes_per_table = int(max_nodes_per_table)

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  temperature: Optional[float] = None) -> Tuple[str, Tuple[int, int]]:
        for attempt in range(5):
            try:
                tokens_key = 'max_tokens' if (self._is_vllm or self._is_nim or self._is_azure) else 'max_completion_tokens'
                # NIM thinking models produce reasoning internally — no /no_think needed
                sys_content = system_prompt if self._is_nim else "/no_think\n" + system_prompt
                call_kwargs = {
                    'messages': [
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_prompt},
                    ],
                    'model': self.model_name,
                    tokens_key: self.max_tokens,
                }
                if self._is_vllm:
                    call_kwargs['extra_body'] = {"chat_template_kwargs": {"enable_thinking": False}}
                if temperature is not None:
                    call_kwargs['temperature'] = temperature
                if self._reasoning_effort:
                    call_kwargs['reasoning_effort'] = self._reasoning_effort
                response = self.ml_client.chat.completions.create(**call_kwargs)
                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
                response_data = response.choices[0].message.content or ""
                # Strip Kimi K2 / NIM-style <thinking>...</thinking> blocks
                response_data = re.sub(r'<thinking>.*?</thinking>', '', response_data, flags=re.DOTALL).strip()
                # Strip Qwen3 / DeepSeek-style <think>...</think> blocks
                if '</think>' in response_data:
                    response_data = response_data.split('</think>')[-1].strip()
                response_data = response_data.replace('```sql\n', '').replace('```json\n', '').replace('\n```', '')
                return response_data, (in_tok, out_tok)

            except Exception as e:
                msg = str(e)
                if 'VLLMValidationError' in msg or 'maximum context length' in msg \
                        or 'BadRequestError' in msg or getattr(e, 'status_code', None) == 400:
                    print(f"Skipping question: non-retryable error: {e}")
                    return "", (0, 0)
                is_rate_limit = '429' in msg or 'rate limit' in msg.lower() \
                    or 'concurrent capacity' in msg.lower() \
                    or getattr(e, 'status_code', None) == 429
                wait = 60 * (attempt + 1) if is_rate_limit else 10 * (attempt + 1)
                print(f"LLM call error: {e} (attempt {attempt + 1}/5, retrying in {wait}s)")
                time.sleep(wait)
        return "", (0, 0)
