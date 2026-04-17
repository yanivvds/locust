import re
from anthropic import AnthropicFoundry, omit
from typing import Tuple

from models.generators.llm_baseline.base_llm_generator import BaseLLMGenerator
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class GPTBaselineSQLModel(BaseLLMGenerator):
    def __init__(self, checkpoint: str, reasoning: str = 'False'):
        super().__init__()
        self.retriever = ColBERTRetriever(checkpoint=checkpoint, mode='table')
        self.reasoning = True if reasoning == 'True' else False

        self.model_name = "claude-sonnet-4-5"
        self.ml_client = AnthropicFoundry(
            # api_version=config.AZURE_API_VERSION,
            base_url="<BASE_URL>",
            api_key="<API_KEY>",
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        try:
            response = self.ml_client.messages.create(
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                max_tokens=10000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 9000
                } if self.reasoning else omit,
            )

            num_tokens = (response.usage.input_tokens, response.usage.output_tokens)
            response_data = response.content[0 if not self.reasoning else 1].text
            # Strip Markdown code fences (```json ... ``` or ``` ... ```)
            response_data = re.sub(r'^```\w*\s*\n?', '', response_data.strip())
            response_data = re.sub(r'\n?```\s*$', '', response_data)
            response_data = response_data.strip()
            return response_data, num_tokens
        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return "", (0, 0)
