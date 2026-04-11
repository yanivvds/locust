import httpx
import config

from typing import List
from odata_graph import engine
from s_expression import Table


class AzureCohereReranker(object):
    def __init__(self):
        self.model_name = "Cohere-rerank-v4.0-pro"

    def rerank(self, question: str, documents: List, k: int) -> httpx.Response:
        with httpx.Client(timeout=None, headers={
            "api-key": config.AZURE_FOUNDRY_KEY,
            "Content-Type": "application/json"
        }) as client:
            response = client.post(
                "https://tzv-meta.services.ai.azure.com/providers/cohere/v2/rerank",
                json={
                    "model": self.model_name,
                    "query": question,
                    "documents": documents,
                    "top_n": k
                }
            )

        return response

    def rerank_tables(self, question: str, tables: dict, k: int = 5) -> dict:
        """
        Rerank retrieved tables using Cohere.

        :param question: natural language question
        :param tables: retriever output dict {table_id: {score, nodes}}
        :param k: number of tables to return after reranking
        :return: dict in the same {table_id: {score, nodes}} format, reranked and trimmed to top k
        """
        if not tables:
            return {}

        table_ids = list(tables.keys())
        table_titles = engine.get_table_titles([str(Table(tid).uri) for tid in table_ids])

        documents = []
        for tid in table_ids:
            title = table_titles.get(tid, {}).get('title', tid)
            description = table_titles.get(tid, {}).get('description', '')
            documents.append(f"{title}: {description}" if description else title)

        response = self.rerank(question=question, documents=documents, k=k)

        reranked = {}
        for result in response.json()['results'][:k]:
            tid = table_ids[result['index']]
            original_data = tables[tid]
            reranked[tid] = {
                **original_data,
                'score': result['relevance_score'],
            }

        return reranked


if __name__ == "__main__":
    # Test the Cohere reranker pro hosted within Azure AI Foundry
    import pprint

    reranker = AzureCohereReranker()
    rank_response = reranker.rerank(
        question="What are the health benefits of green tea?",
        documents=[
                "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
                "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
                "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
                "Basketball is one of the most popular sports in the United States.",
                "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
                "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
            ],
        k=3
    )

    pprint.pp(rank_response.json(), indent=4)
