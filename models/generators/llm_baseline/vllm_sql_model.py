import time
from openai import OpenAI
from typing import Tuple

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.kg_enriched_bm25_retriever import KGEnrichedBM25Retriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class VLLMBaselineSQLModel(BaseLLMGenerator):
    """Baseline SQL generator using a self-hosted vLLM server (OpenAI-compatible API).

    The vLLM server must be running and accessible at the given base_url.
    On Snellius, start it via: sbatch jobs/run_vllm_serve.job
    Then SSH port-forward: ssh -L 8000:localhost:8000 <node>

    Usage (BM25 retriever):
        env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_query_generation.py \\
            --model-path models/generators/llm_baseline/vllm_sql_model.py \\
            --query_type sql --task query-only \\
            --output_path evaluation/results/out.json \\
            --results_path evaluation/results/res.json \\
            --model Qwen/Qwen3.5-27B \\
            --base_url http://localhost:8000/v1

    Usage (ColBERT retriever):
        ... same as above, add:
            --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN
    """

    def __init__(self, model: str = 'Qwen/Qwen3.5-27B', base_url: str = 'http://localhost:8000/v1',
                 checkpoint: str = None, max_tokens: int = 2048):
        super().__init__()
        if checkpoint:
            self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table')
        else:
            self.retriever = KGEnrichedBM25Retriever()
        self.model_name = model
        self.ml_client = OpenAI(api_key='vllm', base_url=base_url, timeout=120)
        self.max_tokens = max_tokens

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        for attempt in range(5):
            try:
                response = self.ml_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "/no_think\n" + system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                num_tokens = (response.usage.prompt_tokens, response.usage.completion_tokens)
                response_data = response.choices[0].message.content
                # Strip chain-of-thought reasoning block (Qwen3.5 thinking mode)
                if '</think>' in response_data:
                    response_data = response_data.split('</think>')[-1].strip()
                response_data = response_data.replace('```sql\n', '').replace('\n```', '')
                return response_data, num_tokens
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"An error occurred while calling the vLLM server: {e} (attempt {attempt + 1}/5, retrying in {wait}s)")
                time.sleep(wait)
        return "", (0, 0)
