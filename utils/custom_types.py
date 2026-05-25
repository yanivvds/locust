"""
This utility module contains custom types and dataclasses that can be used for validating
parameter input in functions when using beartype or general convenience in providing type hints.
"""
from abc import ABC, abstractmethod
from beartype.vale import Is
from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Tuple, Dict, TypeVar, Literal

T = TypeVar('T')

NonEmpty = Is[lambda lst: len(lst) > 0]
NonEmptyList = Annotated[List[T], NonEmpty]

ComparisonOperator = Literal['<', '>', '!=', '<=', '>=', '=']

QueryType = Literal['sexp', 'sql', 'simplified_sql']


class FormatWarning(Warning):
    """Raised when SQL output of a query is not correctly pivoted to our rules."""
    pass


class UnitCompatibilityError(TypeError):
    """Raised when units are not compatible for an aggregation operation."""
    pass


@dataclass
class QAPair(object):
    question: str
    sexp: str
    sql: str
    simplified_sql: str

    def __getitem__(self, attr: str):
        return getattr(self, attr)


class BaseModel(ABC):
    """Main abstract base model."""
    @abstractmethod
    def __init__(self):
        pass


@dataclass
class LLMResponse(object):
    query: str
    input_token_count: int
    output_token_count: int
    probe_hits: Optional[Dict[str, List[str]]] = None
    remarks: Optional[List[Tuple[str, str]]] = None
    elapsed_seconds: float = 0.0

    def to_dict(self):
        d = {
            'query': self.query,
            'input_token_count': self.input_token_count,
            'output_token_count': self.output_token_count,
            'elapsed_seconds': self.elapsed_seconds,
        }
        if self.probe_hits is not None:
            d['probe_hits'] = self.probe_hits
        if self.remarks is not None:
            d['remarks'] = self.remarks
        return d

    def get(self, attr: str, default=None):
        return getattr(self, attr, default)
