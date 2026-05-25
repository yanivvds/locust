import duckdb
import os
import re
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

import config
from models.generators.code_resolver import SKOSCodeResolver
from models.generators.llm_baseline.vllm_sql_model import VLLMBaselineSQLModel
from odata_graph import engine
from s_expression import Table
from utils.custom_types import LLMResponse


class ExecutionLoopGenerator(VLLMBaselineSQLModel):
    """
    Two-phase execution-based generator combining ReFoRCE-style column exploration
    with MAC-SQL-style execution-feedback refinement, scoped by KG ontology.

      Phase 1 (hint injection, upfront):
        v7 hint layers:
          1. SKOS coverage floor on ALL retrieved tables — now injects (code, label)
             pairs so the model can verify which entity each code represents.
          2. Parquet probe on top-probe_top_k_tables table(s): ALL dim columns are
             probed. Ranked (code, label) pairs overwrite SKOS hints when available.
             col_label_threshold gates injection (not probing), so all probe results
             populate the correction cache regardless.
          3. Period code hints: year/quarter/month mentions parsed from the question
             and mapped to CBS period codes. Month/quarter codes claim their year so
             the annual JJ00 code is NOT also injected (prevents over-inclusion).
          Measure hints removed in v7 — they consistently caused extra_measures
          regression by making models treat suggestions as a shopping list.

      Phase 2 (correction evidence, on failure):
        Execute the generated SQL; on error or empty results, parse WHERE predicates,
        run targeted probes, and attach valid codes as evidence. Up to max_retries
        iterations. Failed SQL is truncated to avoid bloating context.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        max_retries: int = 2,
        probe_limit: int = 100,
        evidence_limit: int = 30,
        threshold: int = 80,
        col_label_threshold: int = 60,
        collection_variant: Optional[str] = None,
        max_tokens: int = 2048,
        max_nodes_per_table: int = 500,
        probe_top_k_tables: int = 1,
        use_bm25: bool = False,
        reasoning_effort: Optional[str] = None,
        # v11a — DIVER (Nan et al., SIGMOD 2026): verify SKOS codes exist in parquet before injecting
        use_existence_gate: bool = False,
        # v11b — RSL-SQL (Cao et al., 2024): on first retry, try hint-free SQL as binary alternative
        use_binary_selection: bool = False,
        # v11c — CHASE-SQL (Pourreza et al., ICLR 2025): N-best sampling + execution-guided selection
        n_candidates: int = 1,
        sample_temperature: float = 0.7,
        # v12 — post-generation pruning: remove dim filter values not found in probe_hits
        use_dim_pruning: bool = False,
        # v16 — on-demand DuckDB existence check: prune codes that don't exist in the parquet
        use_duckdb_verify: bool = False,
        # LLM reranker: use LLMTableReranker instead of ColBERT for table retrieval
        use_llm_reranker: bool = False,
        use_enriched_reranker: bool = False,
        retrieval_k: int = 20,
        reranker_model: str = 'gpt-5.4-mini',
        reranker_backend: str = 'openai',
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            checkpoint=checkpoint,
            max_tokens=int(max_tokens),
            max_nodes_per_table=int(max_nodes_per_table),
            collection_variant=collection_variant,
            reasoning_effort=reasoning_effort,
            use_llm_reranker=use_llm_reranker,
            use_enriched_reranker=use_enriched_reranker,
            retrieval_k=int(retrieval_k),
            reranker_model=reranker_model,
            reranker_backend=reranker_backend,
        )
        if use_bm25 or (isinstance(use_bm25, str) and use_bm25.lower() in ('true', '1')):
            from models.retrievers.kg_enriched_bm25_retriever import KGEnrichedBM25Retriever
            self.retriever = KGEnrichedBM25Retriever()
        self.resolver = SKOSCodeResolver(engine, threshold=int(threshold))
        self.max_retries = int(max_retries)
        self.probe_limit = int(probe_limit)
        self.evidence_limit = int(evidence_limit)
        self.col_label_threshold = int(col_label_threshold)
        self.probe_top_k_tables = int(probe_top_k_tables)
        self.use_existence_gate = use_existence_gate if isinstance(use_existence_gate, bool) else str(use_existence_gate).lower() in ('true', '1')
        self.use_binary_selection = use_binary_selection if isinstance(use_binary_selection, bool) else str(use_binary_selection).lower() in ('true', '1')
        self.n_candidates = int(n_candidates)
        self.sample_temperature = float(sample_temperature)
        self.use_dim_pruning = use_dim_pruning if isinstance(use_dim_pruning, bool) else str(use_dim_pruning).lower() in ('true', '1')
        self.use_duckdb_verify = use_duckdb_verify if isinstance(use_duckdb_verify, bool) else str(use_duckdb_verify).lower() in ('true', '1')

    @staticmethod
    def _parquet_path(table_id: str) -> str:
        return os.path.join('data', config.LANGUAGE, 'odata3', f'{table_id}.parquet')

    # CBS period codes: e.g. 2022JJ00, 2023KW01, 1993MM09
    _PERIOD_CODE_RE = re.compile(r'^\d{4}(JJ|KW|MM)\d{2}$')

    @staticmethod
    def _is_time_dim(dim_col: str, codes: list) -> bool:
        """True if this dimension column represents time periods (skip from pruning)."""
        if dim_col.lower() in ('periods', 'period'):
            return True
        # Probe codes that look like CBS period codes → treat as time dim
        sample = codes[:5] if codes else []
        return bool(sample) and all(
            ExecutionLoopGenerator._PERIOD_CODE_RE.match(c) for c in sample
        )

    @staticmethod
    def _prune_hallucinated_dim_filters(sql: str, probe_hits: dict) -> str:
        """v12/v14: strip IN-clause values for probed dim columns that weren't found in parquet.
        Skips time/period dimensions — their codes come from a separate hint mechanism and
        probing only returns top-N by frequency, which may miss the correct period code."""
        if not probe_hits or not sql:
            return sql
        result = sql
        for dim_col, valid_codes in probe_hits.items():
            if not valid_codes:
                continue
            if ExecutionLoopGenerator._is_time_dim(dim_col, valid_codes):
                continue  # never prune period codes
            valid_set = set(valid_codes)
            for col_pat in (f'"{re.escape(dim_col)}"', re.escape(dim_col)):
                pattern = re.compile(rf'{col_pat}\s+IN\s*\(([^)]+)\)', re.IGNORECASE)
                def _replacer(m, vs=valid_set):
                    values = re.findall(r"'([^']*)'", m.group(1))
                    kept = [v for v in values if v in vs]
                    if not kept:
                        return m.group(0)  # safety: nothing survived, keep original
                    full = m.group(0)
                    return full[:full.index('(') + 1] + ', '.join(f"'{v}'" for v in kept) + ')'
                result = pattern.sub(_replacer, result)
        return result

    @staticmethod
    def _verify_dim_codes_duckdb(sql: str) -> str:
        """v16: for each dimension IN-clause, check whether each code actually exists
        in the parquet the SQL references. Prune codes that return 0 rows (invented).

        Unlike _prune_hallucinated_dim_filters (which whitelists from pre-probed columns),
        this method covers ALL dimension columns in the generated SQL, reaching the ~78%
        of invented codes that appear in columns that were never pre-probed.

        Skips Measure, Value, and time/period dimensions."""
        if not sql:
            return sql

        path_match = re.search(r'"(data/[^"]+\.parquet)"', sql)
        if not path_match:
            path_match = re.search(r"'(data/[^']+\.parquet)'", sql)
        if not path_match:
            return sql

        parquet_path = path_match.group(1)
        if not os.path.exists(parquet_path):
            return sql

        # Match DIM_COL IN ('a', 'b', ...) — both quoted and unquoted column names
        in_clause_re = re.compile(
            r'(?:"(\w+)"|(\w+))\s+IN\s*\(([^)]+)\)', re.IGNORECASE
        )
        result = sql

        for m in in_clause_re.finditer(sql):
            dim_col = m.group(1) or m.group(2)
            if dim_col.upper() in ('MEASURE', 'VALUE', 'FOR'):
                continue
            codes_raw = re.findall(r"'([^']*)'", m.group(3))
            if not codes_raw:
                continue
            if ExecutionLoopGenerator._is_time_dim(dim_col, codes_raw):
                continue

            try:
                placeholders = ', '.join(f"'{c}'" for c in codes_raw)
                query = (
                    f'SELECT DISTINCT "{dim_col}" '
                    f"FROM read_parquet('{parquet_path}') "
                    f'WHERE "{dim_col}" IN ({placeholders})'
                )
                valid = {row[0] for row in duckdb.sql(query).fetchall()}
            except Exception:
                continue  # fail-safe: never prune if check errors

            if not valid or valid == set(codes_raw):
                continue  # nothing to prune or all codes are valid

            kept = [c for c in codes_raw if c in valid]
            if not kept:
                continue  # safety: don't wipe entire filter

            original = m.group(0)
            paren_start = original.index('(')
            new_clause = original[: paren_start + 1] + ', '.join(f"'{c}'" for c in kept) + ')'
            result = result.replace(original, new_clause, 1)

        return result

    @classmethod
    def _probe_dim(cls, table_id: str, dim_col: str, limit: int) -> List[str]:
        path = cls._parquet_path(table_id)
        if not os.path.exists(path):
            return []
        try:
            result = duckdb.sql(
                f'SELECT DISTINCT "{dim_col}" FROM read_parquet(\'{path}\') LIMIT {int(limit)}'
            ).fetchall()
            return [str(row[0]) for row in result if row[0] is not None]
        except Exception:
            return []

    @classmethod
    def _code_exists_in_parquet(cls, table_id: str, dim_col: str, code: str) -> bool:
        """
        DIVER-inspired (Nan et al., SIGMOD 2026): verify a candidate CBS code actually
        exists in this table's parquet before injecting it as a hint.

        Analogous to DIVER's value_in(table, col, value) tool — data-grounded rather
        than lexical. Codes absent from the actual data are silently dropped; the probe
        fallback (uniq_value equivalent) remains available via _probe_dim().
        Returns True when the parquet is unavailable (non-blocking fail-safe).
        """
        path = cls._parquet_path(table_id)
        if not os.path.exists(path):
            return True
        try:
            result = duckdb.sql(
                f"SELECT 1 FROM read_parquet('{path}') WHERE \"{dim_col}\" = '{code}' LIMIT 1"
            ).fetchall()
            return len(result) > 0
        except Exception:
            return True

    @classmethod
    def _probe_measure_cols(cls, table_id: str) -> List[str]:
        """Return exact measure column names from the parquet schema (MeasureName_N pattern)."""
        path = cls._parquet_path(table_id)
        if not os.path.exists(path):
            return []
        try:
            cols = duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
            ).fetchall()
            return [c[0] for c in cols if re.match(r'^[A-Za-z][A-Za-z0-9]*_\d+$', c[0])]
        except Exception:
            return []

    @staticmethod
    def _extract_period_codes(question: str) -> List[str]:
        """
        Parse year/quarter/month mentions and return CBS period codes (deduped).

        Month/quarter codes claim their year — the annual JJ00 is NOT also emitted
        for that year, preventing double-injection like (2023MM04, 2023JJ00).

        CBS formats: YYYYJJ00 (year), YYYYKW01-04 (quarter), YYYYMM01-12 (month).
        """
        seen: set = set()
        codes: List[str] = []
        claimed_years: set = set()

        def add(code: str) -> None:
            if code not in seen:
                seen.add(code)
                codes.append(code)

        q = question.lower()
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
        }

        # Month + year: "April 2023" → 2023MM04; year is claimed, no JJ00 added
        for month, num in months.items():
            for m in re.finditer(rf'\b{month}\s+((?:19|20)\d{{2}})\b', q):
                y = m.group(1)
                add(f'{y}MM{num}')
                claimed_years.add(y)

        # Quarters: "Q3 1996", "1997 Q1"
        for q_num, q_code in [('1', '01'), ('2', '02'), ('3', '03'), ('4', '04')]:
            for m in re.finditer(rf'\bq{q_num}\s*((?:19|20)\d{{2}})\b', q):
                y = m.group(1)
                add(f'{y}KW{q_code}')
                claimed_years.add(y)
            for m in re.finditer(rf'\b((?:19|20)\d{{2}})\s*q{q_num}\b', q):
                y = m.group(1)
                add(f'{y}KW{q_code}')
                claimed_years.add(y)

        # Year ranges: "2016-2019", "2016 to 2019" — skip claimed years
        for m in re.finditer(
            r'\b((?:19|20)\d{2})\s*(?:[-\u2013]|to)\s*((?:19|20)\d{2})\b',
            question, re.IGNORECASE,
        ):
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 0 < y2 - y1 <= 20:
                for y in range(y1, y2 + 1):
                    if str(y) not in claimed_years:
                        add(f'{y}JJ00')

        # Slash years: "2017/2018"
        for m in re.finditer(r'\b((?:19|20)\d{2})/((?:19|20)\d{2})\b', question):
            for ys in [m.group(1), m.group(2)]:
                if ys not in claimed_years:
                    add(f'{ys}JJ00')

        # Standalone years — skip claimed years
        for m in re.finditer(r'\b((?:19|20)\d{2})\b', question):
            ys = m.group(1)
            if ys not in claimed_years:
                add(f'{ys}JJ00')

        return codes

    def _collect_hints(
        self, question: str, retrieved_tables: Dict[str, dict]
    ) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[str]]]:
        """
        Two-layer hint collection (v7):

          1. SKOS coverage floor: resolve on EVERY retrieved table, collecting
             (code, label) pairs so the model can verify entity matches.
          2. Probe-validated overwrite: ALL dim columns of the top-probe_top_k_tables
             table(s) are probed. Ranked (code, label) pairs overwrite SKOS hints
             when found. All probe results populate probe_hits regardless of
             col_label_threshold — only injection is gated by the threshold.

        Returns (dim_hints, probe_hits).
          dim_hints: {dim_col: [(code, label), ...]}
          probe_hits: {dim_col: [raw_code, ...]} for correction evidence
        """
        from rapidfuzz import fuzz, utils as fuzz_utils

        dim_hints: Dict[str, List[Tuple[str, str]]] = {}
        probe_hits: Dict[str, List[str]] = {}

        sorted_tables = sorted(
            retrieved_tables.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True,
        )
        probe_table_ids = {t for t, _ in sorted_tables[: self.probe_top_k_tables]}

        for table_id in retrieved_tables.keys():
            # SKOS floor with labels
            try:
                skos_hits = self.resolver.resolve_with_labels(question, table_id)
            except Exception:
                skos_hits = {}
            for dim_col, (code, label) in skos_hits.items():
                if dim_col not in dim_hints:
                    # v11a — DIVER: skip codes not present in this table's actual data
                    if self.use_existence_gate and not self._code_exists_in_parquet(table_id, dim_col, code):
                        continue
                    dim_hints[dim_col] = [(code, label)]

            if table_id not in probe_table_ids:
                continue

            table = Table(table_id)
            try:
                dim_labels = self.resolver.get_schema_dim_labels(table)
            except Exception:
                dim_labels = {}

            for dim_col, dim_label in dim_labels.items():
                # Probe ALL dims (threshold only gates injection, not probing)
                probe_codes = self._probe_dim(table_id, dim_col, self.probe_limit)
                if not probe_codes:
                    continue

                probe_hits[dim_col] = probe_codes

                try:
                    value_label_map = self.resolver.engine.get_dimension_codes(table, dim_col) or {}
                except Exception:
                    value_label_map = {}
                probed_label_map = {c: value_label_map[c] for c in probe_codes if c in value_label_map}

                ranked = (
                    self.resolver.rank_codes(question, probed_label_map, top_k=3)
                    if probed_label_map else []
                )

                col_score = fuzz.partial_ratio(
                    question, dim_label, processor=fuzz_utils.default_process
                )
                if ranked and col_score >= self.col_label_threshold:
                    dim_hints[dim_col] = ranked  # list of (code, label) tuples

        return dim_hints, probe_hits

    @staticmethod
    def _format_hints(
        dim_hints: Dict[str, List[Tuple[str, str]]],
        period_codes: List[str],
    ) -> str:
        parts = []
        if period_codes:
            parts.append(
                "[For PIVOT Periods IN (...) — do NOT add to WHERE: "
                + ", ".join(f"'{c}'" for c in period_codes) + "]"
            )
        if dim_hints:
            hint_parts = []
            for dim, pairs in dim_hints.items():
                code_strs = []
                for entry in pairs:
                    if isinstance(entry, tuple):
                        code, label = entry
                        code_strs.append(f"'{code}' (\"{label}\")" if label else f"'{code}'")
                    else:
                        code_strs.append(f"'{entry}'")
                hint_parts.append(f"{dim}={', '.join(code_strs)}")
            parts.append("[Known CBS codes for WHERE filters: " + "; ".join(hint_parts) + "]")
        return ("\n" + "\n".join(parts)) if parts else ""

    @staticmethod
    def _execute_sql(sql: str) -> Optional[str]:
        try:
            rel = duckdb.sql(sql)
            df = rel.df()
            if df is None or len(df) == 0:
                return "Query executed but returned no results."
            return None
        except Exception as e:
            return str(e)

    def _extract_filter_cols(self, sql: str) -> List[str]:
        cols = set()
        try:
            tree = sqlglot.parse_one(sql, read='duckdb')
            for where in tree.find_all(exp.Where):
                for col in where.find_all(exp.Column):
                    cols.add(col.name)
        except Exception:
            for match in re.finditer(r'"?([A-Za-z_][A-Za-z0-9_]*)"?\s*(?:=|IN|LIKE)\s', sql, re.IGNORECASE):
                cols.add(match.group(1))
        return [c for c in cols if c.lower() not in {'value', 'measure', 'rnk'}]

    def _correction_evidence(
        self,
        failed_sql: str,
        retrieved_tables: Dict[str, dict],
        probe_hits_cache: Dict[str, List[str]],
    ) -> str:
        filter_cols = self._extract_filter_cols(failed_sql)
        if not filter_cols:
            return ""

        evidence_lines = []
        for col in filter_cols[:3]:
            if col in probe_hits_cache and probe_hits_cache[col]:
                codes = probe_hits_cache[col][: self.evidence_limit]
            else:
                codes = []
                for table_id in retrieved_tables.keys():
                    probed = self._probe_dim(table_id, col, self.evidence_limit)
                    if probed:
                        probe_hits_cache.setdefault(col, probed)
                        codes = probed
                        break

            if not codes:
                continue

            # Look up KG labels so the LLM can match codes to intent semantically
            label_map: dict = {}
            for table_id in retrieved_tables.keys():
                try:
                    candidate = self.resolver.engine.get_dimension_codes(Table(table_id), col) or {}
                    if candidate:
                        label_map = candidate
                        break
                except Exception:
                    continue

            if label_map:
                code_strs = [
                    f"'{c}' (\"{label_map[c]}\")" if c in label_map else f"'{c}'"
                    for c in codes
                ]
            else:
                code_strs = [f"'{c}'" for c in codes]

            evidence_lines.append(f"- Column {col} valid values: {', '.join(code_strs)}")

        if not evidence_lines:
            return ""
        return "Valid values from the parquet files:\n" + "\n".join(evidence_lines)

    def generate_query(
        self,
        question: str,
        k: int = 5,
        retrieved_tables: Optional[dict] = None,
        query_type: str = 'sql',
        remarks: Optional[List[Tuple[str, str]]] = None,
        on_event=None,
    ) -> LLMResponse:
        def emit(etype, **data):
            if on_event:
                try:
                    on_event({'type': etype, **data})
                except Exception:
                    pass

        if retrieved_tables is None:
            retrieved_tables = self.retriever.retrieve_tables(question, k=k)
        else:
            table_node_scores = self.retriever.retrieve_tables(question, k=max(k, 1_000))
            golden_tables = {}
            for t in retrieved_tables.keys():
                golden_tables[t] = table_node_scores.get(t, {})
            retrieved_tables = golden_tables

        if not retrieved_tables:
            return LLMResponse(query="", input_token_count=0, output_token_count=0)

        emit('hints_collecting', n_tables=len(retrieved_tables))
        dim_hints, probe_hits = self._collect_hints(question, retrieved_tables)
        period_codes = self._extract_period_codes(question)
        emit('hints_collected',
             dim_cols=list(dim_hints.keys()),
             n_period_codes=len(period_codes),
             period_codes=period_codes[:10])
        base_question = question  # hint-free, used by v11b binary selection
        enriched_question = question + self._format_hints(dim_hints, period_codes)

        accumulated_remarks: List[Tuple[str, str]] = list(remarks) if remarks else []
        probe_hits_cache: Dict[str, List[str]] = dict(probe_hits)
        total_input = 0
        total_output = 0
        last_parsed: Optional[LLMResponse] = None

        # v11c — CHASE-SQL (Pourreza et al., ICLR 2025): N-best sampling + execution-guided selection.
        # Generate N candidates at sample_temperature before entering the correction loop.
        # Execution short-circuit: if any candidate produces non-empty results, return immediately.
        if self.n_candidates > 1:
            candidates = []
            for _ in range(self.n_candidates):
                sys_p, usr_p = self._build_prompt(
                    enriched_question, retrieved_tables,
                    query_type=query_type, remarks=[],
                )
                raw_c, (in_c, out_c) = self._call_llm(sys_p, usr_p, temperature=self.sample_temperature)
                total_input += in_c
                total_output += out_c
                parsed_c = self._parse_response(raw_c, (total_input, total_output))
                if parsed_c.query:
                    candidates.append(parsed_c)

            winners = [p for p in candidates if self._execute_sql(p.query) is None]
            if winners:
                best = max(winners, key=lambda p: len(p.query))
                if self.use_dim_pruning and best.query and probe_hits:
                    best.query = self._prune_hallucinated_dim_filters(best.query, probe_hits)
                if self.use_duckdb_verify and best.query:
                    best.query = self._verify_dim_codes_duckdb(best.query)
                best.probe_hits = probe_hits or None
                best.remarks = None
                return best
            # All failed — fall through to correction loop with first candidate as seed
            if candidates:
                last_parsed = candidates[0]
                sql_snippet = (candidates[0].query[:400] + '...') if len(candidates[0].query) > 400 else candidates[0].query
                err_seed = self._execute_sql(candidates[0].query) or ""
                accumulated_remarks.append((sql_snippet, err_seed))

        for attempt in range(self.max_retries + 1):
            emit('llm_attempt', attempt=attempt + 1, total=self.max_retries + 1)
            system_prompt, user_prompt = self._build_prompt(
                enriched_question, retrieved_tables,
                query_type=query_type, remarks=accumulated_remarks,
            )
            raw_response, (in_tok, out_tok) = self._call_llm(system_prompt, user_prompt)
            total_input += in_tok
            total_output += out_tok
            parsed = self._parse_response(raw_response, (total_input, total_output))
            last_parsed = parsed

            sql = parsed.query
            if not sql:
                break

            emit('sql_executing')
            error_msg = self._execute_sql(sql)
            if error_msg is None:
                emit('sql_ok')
                break

            if attempt == self.max_retries:
                emit('sql_failed', error=(error_msg or '')[:300])
                break

            # v11b — RSL-SQL (Cao et al., 2024): binary hint selection.
            # On the first retry, also try a hint-free SQL as an alternative.
            # If the stripped version succeeds and the hinted version failed, return the stripped SQL.
            if self.use_binary_selection and attempt == 0:
                sys_stripped, usr_stripped = self._build_prompt(
                    base_question, retrieved_tables, query_type=query_type, remarks=[],
                )
                raw_stripped, (in_s, out_s) = self._call_llm(sys_stripped, usr_stripped)
                total_input += in_s
                total_output += out_s
                parsed_stripped = self._parse_response(raw_stripped, (total_input, total_output))
                if parsed_stripped.query and self._execute_sql(parsed_stripped.query) is None:
                    if self.use_dim_pruning and probe_hits:
                        parsed_stripped.query = self._prune_hallucinated_dim_filters(parsed_stripped.query, probe_hits)
                    if self.use_duckdb_verify:
                        parsed_stripped.query = self._verify_dim_codes_duckdb(parsed_stripped.query)
                    parsed_stripped.probe_hits = probe_hits or None
                    parsed_stripped.remarks = accumulated_remarks or None
                    return parsed_stripped

            emit('sql_error', error=(error_msg or '')[:300], attempt=attempt + 1)
            evidence = self._correction_evidence(sql, retrieved_tables, probe_hits_cache)
            directive = "Fix: change only the filter values in WHERE/UNPIVOT to match the valid codes below.\n"
            detailed_error = directive + error_msg + ("\n" + evidence if evidence else "")
            sql_snippet = (sql[:400] + '...') if len(sql) > 400 else sql
            accumulated_remarks.append((sql_snippet, detailed_error))

        if last_parsed is None:
            last_parsed = LLMResponse(
                query="", input_token_count=total_input, output_token_count=total_output
            )

        if self.use_dim_pruning and last_parsed.query and probe_hits:
            last_parsed.query = self._prune_hallucinated_dim_filters(last_parsed.query, probe_hits)
        if self.use_duckdb_verify and last_parsed.query:
            last_parsed.query = self._verify_dim_codes_duckdb(last_parsed.query)

        last_parsed.probe_hits = probe_hits if probe_hits else None
        last_parsed.remarks = accumulated_remarks if accumulated_remarks else None
        return last_parsed
