"""
This module contains a simple rule-based node retrieval module using BM25+, for indexing and
retrieving the top-scoring tables and their corresponding dimensions and measures for a given query.
"""
import itertools
import json
import nltk
import os
import string
from collections import OrderedDict
from rdflib.namespace import QB
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer, EnglishStemmer
from nltk.tokenize import word_tokenize
from operator import itemgetter
from rank_bm25 import BM25Plus
from tqdm import tqdm
from typing import Dict, List

import config
from models.retrievers.base_retriever import BaseRetriever
from odata_graph import engine

NODE_LABEL_BASE_PATH = f"{config.PATH_DIR_DATA}/bm25_retriever/bm25_{config.GRAPH_DB_REPO}_node_labels{{}}.json"


class BM25Retriever(BaseRetriever):
    """
        :var indices: A dictionary of BM25+ index objects for each component.
        :var doc_ids_map: A dictionary of document ID lists for each component.
    """
    indices: Dict[str, BM25Plus]
    doc_ids_map: Dict[str, List[str]]

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')

        super().__init__()

        graph_nodes = get_graph_node_labels(preprocess_text=True, include_time_geo_dims=False)
        self.index_nodes(graph_nodes)

    def index_nodes(self, nodes: Dict[str, Dict[str, str]]):
        """
            Indexes the provided nodes into separate BM25+ indices for tables, dimensions, and measures.

            :param nodes: A dictionary of nodes from get_graph_node_labels.
            :return: A tuple containing a dictionary of BM25+ index objects and a dictionary of document IDs.
        """
        components = {'table': {}, 'dimension': {}, 'measure': {}}
        for node_id, node_data in nodes.items():
            node_type = node_data.get('type')
            if node_type in components:
                components[node_type][node_id.strip()] = node_data

        indices = {}
        doc_ids_map = {}
        for component_type, component_nodes in components.items():
            if not component_nodes:
                print(f"No nodes found for type '{component_type}', skipping index creation.")
                indices[component_type] = None
                doc_ids_map[component_type] = []
                continue

            doc_ids = list(component_nodes.keys())
            corpus = [component_nodes[doc_id]['body'] for doc_id in doc_ids]
            tokenized_corpus = [doc.split(" ") for doc in corpus]

            print(f"Creating BM25+ index for {component_type}s...")
            bm25 = BM25Plus(tokenized_corpus)
            indices[component_type] = bm25
            doc_ids_map[component_type] = doc_ids

        self.indices = indices
        self.doc_ids_map = doc_ids_map

    def retrieve_tables(self, question: str, k: int = 5, candidate_pool_size: int = 10) -> OrderedDict:
        """
            Queries the BM25+ indices, groups results by table and returns a sorted,
            nested list of the top tables and their nodes.

            :param question: The query string.
            :param k: The final number of top tables to return.
            :param candidate_pool_size: The number of initial candidate dimensions and measures to retrieve.
            :return: A sorted list of dictionaries with top tables and their nodes.
        """
        processed_query = process_text(question)
        tokenized_query = processed_query.split()

        top_candidates = []

        # Add all tables with a score > 0 to the candidate pool
        if 'table' in self.indices and self.indices['table']:
            scores = self.indices['table'].get_scores(tokenized_query)
            for i, score in enumerate(scores):
                if score > 0:
                    top_candidates.append({'id': self.doc_ids_map['table'][i], 'score': score, 'type': 'table'})
        else:
            raise RuntimeError("No table index found.")

        # Get top N candidates for dimensions and measures
        top_k_table_ids = {c['id'] for c in sorted(top_candidates, key=lambda x: x['score'], reverse=True)[:k]}

        for node_type in ['dimension', 'measure']:
            if node_type in self.indices and self.indices[node_type]:
                scores = self.indices[node_type].get_scores(tokenized_query)

                # Create a list of nodes and their scores corresponding with the top-k tables by score
                all_nodes_of_type = [
                    {'id': self.doc_ids_map[node_type][i], 'score': scores[i], 'type': node_type}
                    for i in range(len(scores))
                    if self.doc_ids_map[node_type][i].split('#')[0] in top_k_table_ids
                ]

                all_nodes_of_type.sort(key=lambda x: x['score'], reverse=True)
                top_candidates.extend(all_nodes_of_type[:candidate_pool_size])
            else:
                raise RuntimeError(f"No {node_type} index found.")

        # Group nodes by table and store their scores
        table_candidates = {}
        for node in top_candidates:
            node_id = node['id']
            score = node['score']
            node_type = node['type']

            if node_type == 'table':
                table_id = node_id
            else:  # dimension or measure
                table_id, child_id = node_id.split('#', 1)

            if table_id not in table_candidates:
                # Initialize with dimensions and measures keys
                table_candidates[table_id] = {"table_score": 0.0, "dimensions": {}, "measures": {}}

            if node_type == 'table':
                table_candidates[table_id]["table_score"] = score
            elif node_type == 'dimension':
                table_candidates[table_id]["dimensions"][child_id] = {"score": score}
            elif node_type == 'measure':
                table_candidates[table_id]["measures"][child_id] = {"score": score}

        # Calculate combined table scores and format for sorting
        sorted_tables = []
        for table_id, data in table_candidates.items():
            dim_scores = sum(node["score"] for node in data["dimensions"].values())
            msr_scores = sum(node["score"] for node in data["measures"].values())
            combined_score = data["table_score"] + dim_scores + msr_scores
            data['combined_score'] = combined_score
            sorted_tables.append((table_id, data))

        # Sort tables by the new combined score in descending order
        sorted_tables.sort(key=lambda item: item[1]["combined_score"], reverse=True)

        # Format the final output structure
        final_results = OrderedDict()
        for table_id, data in sorted_tables[:k]:
            # Sort child nodes by score for cleaner output
            sorted_dims = sorted(data["dimensions"].items(), key=lambda item: item[1]["score"], reverse=True)
            sorted_msrs = sorted(data["measures"].items(), key=lambda item: item[1]["score"], reverse=True)

            final_results[table_id] = {
                "score": data["combined_score"],
                "dimensions": OrderedDict([(node_id, node_data) for node_id, node_data in sorted_dims]),
                "measures": OrderedDict([(node_id, node_data) for node_id, node_data in sorted_msrs])
            }

        return final_results


def get_graph_node_labels(include_time_geo_dims: bool = False, preprocess_text: bool = True) -> Dict[str, Dict[str, str]]:
    """Return a dictionary with all the textual elements for every table and measure/dimension in the graph."""
    node_labels_path = NODE_LABEL_BASE_PATH.format('_including_time_geo' if include_time_geo_dims else '')
    if os.path.isfile(node_labels_path):
        print(f"Loading node labels from {node_labels_path}")
        with open(node_labels_path) as f:
            node_labels = json.load(f)
            return node_labels

    tables = {}
    nodes = {}

    table_query = ("""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT DISTINCT ?id ?label WHERE {{ 
            ?s a dcat:Dataset .
            ?s dct:identifier ?id .
            OPTIONAL {{ ?s dct:title|dct:abstract|dct:description ?label }}
        }}
    """)

    try:
        fetch_tables = engine.select(table_query)
        props = [(r['id']['value'], (r.get('label', False) or {'value': ''})['value']) for r in fetch_tables]
        table_it = itertools.groupby(props, itemgetter(0))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch table IDs: {e}")

    for table, table_props in tqdm(table_it, total=len(set(map(itemgetter(0), props))),
                                   desc="Indexing graph nodes", bar_format=config.TQDM_BAR_FMT):
        val = ' '.join(f"{prop[1].strip()}{'' if prop[1][-1] == '.' else '.'}" for prop in table_props)
        tables[table] = {'body': process_text(val) if preprocess_text else val, 'type': 'table'}

        # Get all measures and dimensions for a table
        query = (f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX qb: <http://purl.org/linked-data/cube#>
            
            SELECT ?node_id ?type ?label WHERE {{
                ?table ?type ?node .
                FILTER (?type = qb:measure || ?type = qb:dimension) .
                {'''
                FILTER NOT EXISTS {
                    VALUES ?timegeo {'TimeDimension' 'GeoDimension'}
                    ?node a ?timegeo.
                }
                ''' if not include_time_geo_dims else ''}
                ?table dct:identifier ?table_id .
                ?node qb:concept ?concept ;
                      dct:identifier ?node_id .
                ?concept dct:isPartOf ?table .
                OPTIONAL {{ ?concept skos:prefLabel|skos:altLabel|skos:definition|dct:description|dct:subject ?label }}
                FILTER (?table_id = "{table}")
            }}
        """)

        try:
            result = engine.select(query)
            props = [(r['node_id']['value'], r['type']['value'], (r.get('label', False) or {'value': ''})['value'])
                     for r in result]
            node_it = itertools.groupby(props, itemgetter(0))
            for node_id, node_props in node_it:
                node_props = list(node_props)
                node_type = node_props[0][1]
                if node_type == str(QB.measure):
                    type_ = 'measure'
                elif node_type == str(QB.dimension):
                    type_ = 'dimension'
                else:
                    raise ValueError(f"Unknown node type: {node_type}")

                val = ' '.join(f"{prop[2].strip()}{'' if prop[2][-1] == '.' else '.'}" for prop in node_props)
                # Because nodes can have different concepts per table, store them using unique identifiers per table
                nodes[f"{table}#{node_id}"] = {'body': process_text(val) if preprocess_text else val,
                                               'type': type_, 'table': table}
        except Exception as e:
            print(f"Failed to fetch table nodes: {e}")

    nodes.update(tables)
    os.makedirs(os.path.dirname(node_labels_path), exist_ok=True)
    with open(node_labels_path, 'w') as f:
        json.dump(nodes, f)
    print(f"Successfully saved node properties to {node_labels_path}")
    return nodes


def stem_words(words: List[str]) -> List[str]:
    """Stem given words using basic Porter's stemming."""
    if config.LANGUAGE == 'nl':
        stemmer = DutchStemmer()
    else:
        stemmer = EnglishStemmer()

    return [stemmer.stem(w) for w in words]


def process_text(text: str) -> str:
    """Process a given text using basic tokenization, stemming and removing punctuation and stop words."""
    text = text.lower()

    translation = str.maketrans('', '', string.punctuation)
    stripped_words = [w.translate(translation) for w in word_tokenize(text)]

    stop_words = set(stopwords.words('dutch' if config.LANGUAGE == 'nl' else 'english'))
    relevant_words = [w for w in stripped_words if w not in stop_words]

    processed_text = stem_words(relevant_words)
    return ' '.join(processed_text)
