from abc import ABC, abstractmethod
from collections import OrderedDict

from utils.custom_types import BaseModel


class BaseRetriever(BaseModel, ABC):
    """Main abstract base model for doing table retrieval."""
    @abstractmethod
    def __init__(self):
        pass

    def retrieve_tables(self, query: str, k: int) -> OrderedDict:
        """
            Retrieve the top K tables for the given question.

            :param query: natural language question
            :param k: number of tables to retrieve
            :return: dictionary of K tables, sorted on most to least relevant to the question.
                     dictionary should have the shape of:
                     {
                        'table_id': {
                            'score': 0.42
                            'nodes': {  ----- <This is optional and only when retrieving tables + nodes>
                                'node_id1': {'score': 0.42},
                                'node_id2': {'score': 0.39},
                                'node_id2': {'score': 0.36},
                            }
                        }
                    }
        """
        raise NotImplementedError()
