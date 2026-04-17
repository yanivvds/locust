from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from utils.custom_types import BaseModel, LLMResponse


class BaseGenerator(BaseModel, ABC):
    """Main abstract base model for doing query generation."""
    @abstractmethod
    def __init__(self):
        pass

    def generate_query(
            self,
            question: str,
            retrieved_tables: Optional[List[str]] = None
    ) -> LLMResponse:
        """
            Generate a logical query for the given question.

            :param question: natural language question
            :param retrieved_tables: optional full retriever output dict with table scores and optional node scores.
            :return: tuple containing LLM response and number of input & output tokens
        """
        raise NotImplementedError()
