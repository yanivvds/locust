"""
This utility module contains custom types and dataclasses that can be used for validating
parameter input in functions when using beartype or general convenience in providing type hints.
"""
from abc import ABC, abstractmethod
from beartype.vale import Is
from dataclasses import dataclass
from typing import Annotated, List, TypeVar, Literal

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

    def to_dict(self):
        return {
            'query': self.query,
            'input_token_count': self.input_token_count,
            'output_token_count': self.output_token_count
        }

    def get(self, attr: str, default=None):
        return getattr(self, attr, default)
