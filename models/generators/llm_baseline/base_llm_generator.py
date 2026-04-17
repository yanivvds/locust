import json
import os
from abc import ABC
from itertools import groupby
from operator import itemgetter
from rdflib import Literal
from rdflib.namespace import SKOS, DCTERMS as DCT, QB, RDF, XSD
from typing import Tuple, Optional, List

import config
from models.generators.base_generator import BaseGenerator
from models.retrievers.base_retriever import BaseRetriever
from odata_graph import engine
from odata_graph.sparql_controller import QUDT
from s_expression import Table, uri_to_code
from utils.custom_types import LLMResponse


class BaseLLMGenerator(BaseGenerator, ABC):
    """
        A baseline model for generating SQL queries using a LLM.

        :param retriever: table/node retriever that is used for populating the prompt going out to the LLM
        :param node_score_threshold: score threshold for node items in tables for adding to the prompt.
                                     If the retriever does not return node scores, this value is ignored.
                                     Set to 0 to add all nodes connected to a table in a prompt (might yield very large prompts!)
    """
    retriever: BaseRetriever
    node_score_threshold: float = 0.1

    def _build_prompt(self,
                      question: str,
                      tables: dict,
                      query_type: str = 'sql',
                      max_nodes_per_table: int = 500) -> Tuple[str, str]:
        """
            Builds a prompt for the LLM based on the question and retrieved tables.

            :param question: question string
            :param tables: dictionary of ranked tables and their nodes
            :param max_nodes_per_table: maximum number of nodes to include per table
        """
        table_info = []
        for table_id, data in tables.items():
            # Get graph elements for tables
            table_node = Table(table_id)
            table_graph = engine.get_table_graph(table_node, include_time_geo_dims=True)

            # Separate the dimension types into buckets for rule-based extraction of geo and time dims from the query
            dim_groups = {uri_to_code(s) for s in set(table_graph.objects(predicate=QB.dimension)) -
                          set(table_graph.subjects(predicate=SKOS.broader))}
            geo_dims = {uri_to_code(s) for s in
                table_graph.subjects(predicate=RDF.type, object=Literal("GeoDimension", datatype=XSD.string if config.LOCAL_GRAPH else None))
            } - dim_groups
            time_dims = {uri_to_code(s) for s in
                table_graph.subjects(predicate=RDF.type, object=Literal("TimeDimension", datatype=XSD.string if config.LOCAL_GRAPH else None))
            } - dim_groups
            schema_dims = {uri_to_code(s) for s in table_graph.objects(predicate=QB.dimension)} - dim_groups - geo_dims - time_dims

            units = dict(map(lambda x: (uri_to_code(x[0]), x[1].split('/')[-1]), table_graph.subject_objects(QUDT.unit)))
            concepts = set(map(lambda x: (x[1], x[0]), table_graph.subject_objects(QB.concept)))
            labels = set(table_graph.subject_objects(SKOS.prefLabel))

            # Inner join concepts with labels
            def inner_join(a, b):
                L = a + b
                L.sort(key=itemgetter(0))  # sort by the first column
                for _, group in groupby(L, itemgetter(0)):
                    row_a, row_b = next(group), next(group, None)
                    if row_b is not None:  # join
                        yield row_a[1:] + row_b[1:]  # cut 1st column from 2nd row

            concept_labels = dict(inner_join(list(concepts), list(labels)))

            # Setup prompt
            table_labels = set(table_graph.objects(table_node.uri, DCT.title) or {''})
            if len(table_labels) == 0:
                table_labels = {''}
            table_info.append(f"Table: {table_id} "
                              f"(label: \"{str(table_labels.pop())}\") "
                              f"(score: {data.get('score', '-')})")

            # Build a mapping from dimension codes to their parent group via skos:broader
            dim_to_group = {}
            for child, parent in table_graph.subject_objects(SKOS.broader):
                child_code = uri_to_code(child)
                parent_code = uri_to_code(parent)
                if parent_code in dim_groups:
                    dim_to_group[child_code] = parent_code

            # Build node entries, filtering out irrelevant dimensions
            node_scores = data.get('nodes', {})
            measures = []
            group_entries = {}  # group_code -> (group_entry, [child_entries])
            ungrouped = []

            for code, label in concept_labels.items():
                node_code = uri_to_code(code)

                if node_code in dim_groups:
                    node_type = "DimensionGroup"
                elif node_code in schema_dims | geo_dims | time_dims:
                    node_type = "Dimension"
                else:
                    node_type = "Measure"

                score = node_scores.get(node_code, {}).get('score')
                if node_scores and score and score < self.node_score_threshold:
                    continue
                entry = (node_code, node_type, label, score)

                if node_type == "Measure":
                    measures.append(entry)
                elif node_type == "DimensionGroup":
                    if node_code not in group_entries:
                        group_entries[node_code] = (entry, [])
                    else:
                        group_entries[node_code] = (entry, group_entries[node_code][1])
                elif node_code in dim_to_group:
                    parent = dim_to_group[node_code]
                    if parent not in group_entries:
                        group_entries[parent] = (None, [])
                    group_entries[parent][1].append(entry)
                else:
                    ungrouped.append(entry)

            # Sort each category: scored first (highest score on top), then unscored
            def _sort_key(e):
                return (e[3] is not None, e[3] or 0)

            measures.sort(key=_sort_key, reverse=True)
            ungrouped.sort(key=_sort_key, reverse=True)

            # Build ordered list: measures first, then groups with their children, then ungrouped
            ordered_entries = list(measures)
            # Sort groups by their best child score (or own score)
            def _group_sort_key(item):
                group_code, (group_entry, children) = item
                scores = [c[3] for c in children if c[3] is not None]
                if group_entry and group_entry[3] is not None:
                    scores.append(group_entry[3])
                return (len(scores) > 0, max(scores) if scores else 0)

            for group_code, (group_entry, children) in sorted(group_entries.items(),
                                                              key=_group_sort_key, reverse=True):
                if group_entry:
                    ordered_entries.append(group_entry)
                children.sort(key=_sort_key, reverse=True)
                ordered_entries.extend(children)

            ordered_entries.extend(ungrouped)

            nodes_info = []
            for node_code, node_type, label, score in ordered_entries[:max_nodes_per_table]:
                line = (f"- {node_type}: {node_code} "
                        f"(label: \"{label}\")"
                        f"{' (unit: ' + units[node_code] + ')' if node_code in units else ''}"
                        f"{' (relevance: ' + f'{score:.4f}' + ')' if score is not None else ''}")
                nodes_info.append(line)

            if nodes_info:
                table_info.append("Nodes:")
                table_info.extend(nodes_info)
            table_info.append("")

        table_info_str = chr(10).join(table_info)

        file_name = f"{config.LANGUAGE}_prompt{'_simplified' if query_type == 'simplified_sql' else ''}.txt"
        prompt_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        system_prompt = prompt_template.split('[SYSTEM]')[1].split('[USER]')[0].strip()
        user_prompt_template = prompt_template.split('[USER]')[1].strip()

        user_prompt = user_prompt_template.format(question=question, table_info=table_info_str)
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Tuple[int, int]]:
        """Calls the LLM endpoint and returns the generated query."""
        raise NotImplementedError()

    def generate_query(
        self,
        question: str,
        k: int = 5,
        retrieved_tables: Optional[dict] = None,
        query_type: str = 'sql',
        remarks: Optional[List[Tuple[str, str]]] = None
    ) -> LLMResponse:
        """
            Generates a query for the given question.

            :param question: natural language question
            :param k: number of tables to retrieve for the LLM to process
            :param retrieved_tables: full retriever output dict with table scores and optional node scores.
            :param query_type: type of query to generate (sexp or sql)
            :param remarks: optional list of (previous_response_json, error_message) tuples from failed attempts,
                            appended to the conversation so the model can correct itself.
            :returns: tuple of (parsed JSON response dict, token counts).
                      The dict has keys: query (str), status (OK|ALTERNATIVE|UNANSWERABLE), remarks (str).
        """
        if retrieved_tables is None:
            retrieved_tables = self.retriever.retrieve_tables(question, k=k)
        else:
            table_node_scores = self.retriever.retrieve_tables(question, k=max(k, 1_000))  # get scored nodes for golden tables
            golden_tables = {}
            for t in retrieved_tables.keys():
                golden_tables[t] = table_node_scores.get(t, {})
            retrieved_tables = golden_tables

        if not retrieved_tables:
            return LLMResponse(query="", input_token_count=0, output_token_count=0)

        system_prompt, user_prompt = self._build_prompt(question, retrieved_tables, query_type=query_type)
        raw_response, num_tokens = self._call_llm(system_prompt, user_prompt)
        parsed = self._parse_response(raw_response, num_tokens)
        return parsed

    @staticmethod
    def _parse_response(raw: str, token_counts: Tuple[int, int]) -> LLMResponse:
        """Parse the LLM's JSON response into a structured dict."""
        try:
            parsed = json.loads(raw)
            return LLMResponse(
                query=parsed.get("query", ""),
                input_token_count=token_counts[0],
                output_token_count=token_counts[1],
            )
        except (json.JSONDecodeError, AttributeError, ValueError, KeyError):
            return LLMResponse(query="", input_token_count=0, output_token_count=0)
