import re
from google import genai
from google.genai import types
from typing import Tuple

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class GPTBaselineSQLModel(BaseLLMGenerator):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table')

        self.model_name = "gemini-2.5-pro"
        self.ml_client = genai.Client(api_key='<API_KEY>')

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        try:
            response = self.ml_client.models.generate_content(
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=20000
                ),
                contents=user_prompt,
                model=self.model_name
            )

            num_tokens = (response.usage_metadata.prompt_token_count,
                          response.usage_metadata.candidates_token_count + response.usage_metadata.thoughts_token_count)
            response_data = re.sub(r'^```\w*\s*\n?', '', response.text.strip())
            response_data = re.sub(r'\n?```\s*$', '', response_data)
            response_data = response_data.strip()
            return response_data, num_tokens
        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return "", (0, 0)
