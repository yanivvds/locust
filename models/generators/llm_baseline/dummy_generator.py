from typing import Tuple, Optional, List
from models.generators.base_generator import BaseGenerator


class DummyGenerator(BaseGenerator):
    """Returns empty string for any question. Used to evaluate pre-cached outputs."""

    def __init__(self):
        pass

    def generate_query(self, question: str, k: int = 5, golden_tables: Optional[List[str]] = None, query_type: str = 'sql') -> Tuple[str, Tuple[int, int]]:
        return "", (0, 0)
