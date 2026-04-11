import os
from openai import OpenAI, NOT_GIVEN
from typing import Tuple

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class OpenAIBaselineSQLModel(BaseLLMGenerator):
    """Baseline SQL generator using a personal OpenAI API key (non-Azure).
    Set OPENAI_API_KEY in your environment before running.
    """

    def __init__(self, checkpoint: str, model: str = 'gpt-5-mini', reasoning: str = 'False'):
        super().__init__()
        self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table')
        self.model_name = model
        self.reasoning = True if reasoning == 'True' else False
        self.ml_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        try:
            response = self.ml_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                max_completion_tokens=10000,
            )

            num_tokens = (response.usage.prompt_tokens, response.usage.completion_tokens)
            response_data = response.choices[0].message.content
            response_data = response_data.replace('```sql\n', '').replace('\n```', '')
            return response_data, num_tokens
        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return "", (0, 0)
