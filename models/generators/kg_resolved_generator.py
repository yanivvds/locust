from typing import Tuple, Optional, List

from models.generators.base_generator import BaseGenerator
from models.generators.code_resolver import SKOSCodeResolver
from models.generators.llm_baseline.openai_sql_model import OpenAIBaselineSQLModel
from odata_graph import engine


class KGResolvedGenerator(BaseGenerator):
    """
    Wraps OpenAIBaselineSQLModel and injects SKOS-resolved schema dimension
    codes into the question before prompting the LLM.

    The existing BaseLLMGenerator already resolves geo and time dimension codes
    via match_region() / extract_tc(). This wrapper adds resolution for all
    other (schema) dimensions by querying the Knowledge Graph.

    Usage in evaluation:
        --model-path models/generators/kg_resolved_generator.py
        --checkpoint "StatisticsNetherlands/GECKOv2-ColBERT-EN"
    """

    def __init__(self, checkpoint: str, model: str = 'gpt-5-mini',
                 reasoning: str = 'False', threshold: int = 80):
        """
        :param checkpoint: ColBERT checkpoint for the base generator's retriever
        :param model: OpenAI model name (default: gpt-5-mini)
        :param reasoning: pass 'True' to enable reasoning effort on supported models
        :param threshold: fuzzy match threshold for SKOS code resolver (0–100)
        """
        super().__init__()
        self.base = OpenAIBaselineSQLModel(
            checkpoint=checkpoint, model=model, reasoning=reasoning
        )
        self.resolver = SKOSCodeResolver(engine, threshold=int(threshold))

    def generate_query(self, question: str, golden_tables: Optional[List[str]] = None,
                       query_type: str = 'sql') -> Tuple[str, Tuple[int, int]]:
        """
        Resolve schema dim codes for each golden table, inject as hints into
        the question, then delegate to the base generator.
        """
        hints = {}
        for table_id in (golden_tables or []):
            hints.update(self.resolver.resolve(question, table_id))

        if hints:
            hint_str = ', '.join(f"{dim}='{code}'" for dim, code in hints.items())
            enriched_question = question + f"\n[Known CBS codes: {hint_str}]"
        else:
            enriched_question = question

        return self.base.generate_query(
            enriched_question, golden_tables=golden_tables, query_type=query_type
        )
