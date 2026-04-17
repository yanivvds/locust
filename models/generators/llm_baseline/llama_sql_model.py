import re
from openai import OpenAI, NOT_GIVEN
from typing import Tuple

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class GPTBaselineSQLModel(BaseLLMGenerator):
    def __init__(self, checkpoint: str, reasoning: str = 'False'):
        super().__init__()
        self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table')
        self.reasoning = True if reasoning == 'True' else False

        self.model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8'
        self.ml_client = OpenAI(
            base_url='<BASE_URL>',
            api_key='<API_KEY>',
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        try:
            response = self.ml_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                max_completion_tokens=20000,
                reasoning_effort="high" if self.reasoning else NOT_GIVEN
            )

            num_tokens = (response.usage.prompt_tokens, response.usage.completion_tokens)
            response_data = response.choices[0].message.content
            response_data = re.sub(r'^```\w*\s*\n?', '', response_data.strip())
            response_data = re.sub(r'\n?```\s*$', '', response_data)
            response_data = response_data.strip()
            return response_data, num_tokens
        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return "", (0, 0)
