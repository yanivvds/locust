from typing import Tuple, Optional, List

from models.generators.base_generator import BaseGenerator
from models.generators.code_resolver import SKOSCodeResolver
from odata_graph import engine


class KGResolvedGenerator(BaseGenerator):
    """
    Wraps a baseline SQL generator and injects SKOS-resolved schema dimension
    codes into the question before prompting the LLM.

    Supports two backends via the `backend` parameter:
      - 'openai'  — uses OpenAIBaselineSQLModel (requires OPENAI_API_KEY)
      - 'vllm'    — uses VLLMBaselineSQLModel (requires a running vLLM server)

    The existing BaseLLMGenerator already resolves geo and time dimension codes
    via match_region() / extract_tc(). This wrapper adds resolution for all
    other (schema) dimensions by querying the Knowledge Graph.

    Usage (OpenAI):
        --model-path models/generators/kg_resolved_generator.py
        --backend openai --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN

    Usage (vLLM / Gemma-4 on Snellius reverse proxy):
        --model_path models/generators/kg_resolved_generator.py
        --backend vllm --model google/gemma-4-31B-it
        --base_url http://localhost:8000/v1
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN
    """

    def __init__(self, backend: str = 'openai', threshold: int = 80,
                 checkpoint: str = None, model: str = None,
                 reasoning: str = 'False', base_url: str = 'http://localhost:8000/v1',
                 max_tokens: int = 2048):
        """
        :param backend: 'openai' or 'vllm'
        :param threshold: fuzzy match threshold for SKOS code resolver (0–100)
        :param checkpoint: ColBERT checkpoint for table retrieval (optional for query-only task)
        :param model: LLM model name (defaults to backend's own default if omitted)
        :param reasoning: OpenAI backend only — pass 'True' to enable reasoning effort
        :param base_url: vLLM backend only — URL of the OpenAI-compatible vLLM server
        """
        super().__init__()

        if backend == 'vllm':
            from models.generators.llm_baseline.vllm_sql_model import VLLMBaselineSQLModel
            vllm_kwargs = {'base_url': base_url, 'max_tokens': int(max_tokens)}
            if model:
                vllm_kwargs['model'] = model
            if checkpoint:
                vllm_kwargs['checkpoint'] = checkpoint
            self.base = VLLMBaselineSQLModel(**vllm_kwargs)
        else:
            from models.generators.llm_baseline.openai_sql_model import OpenAIBaselineSQLModel
            openai_kwargs = {'reasoning': reasoning}
            if model:
                openai_kwargs['model'] = model
            if checkpoint:
                openai_kwargs['checkpoint'] = checkpoint
            self.base = OpenAIBaselineSQLModel(**openai_kwargs)

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
