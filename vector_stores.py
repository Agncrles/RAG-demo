import json

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings

import config_data as config
from konwledge import connect_db


class VectorStores(object):
    def __init__(self, embedding):
        self.embedding = embedding
        conn = connect_db()
        conn.close()

    def similarity_search(self, query, k):
        query_vector = self.embedding.embed_query(query)
        conn = connect_db()
        try:
            rows = conn.execute(
                "SELECT source, content, embedding, create_time, operator FROM chunks"
            ).fetchall()
        finally:
            conn.close() 

        ranked = []
        for row in rows:
            vector = json.loads(row["embedding"])
            distance = sum((a - b) ** 2 for a, b in zip(query_vector, vector))
            ranked.append((distance, row))
        ranked.sort(key=lambda item: item[0])

        return [
            Document(
                page_content=row["content"],
                metadata={
                    "source": row["source"],
                    "create_time": row["create_time"],
                    "operator": row["operator"],
                },
            )
            for _, row in ranked[:k]
        ]

    def get_retriever(self):
        k = config.similarity_threshold
        return RunnableLambda(lambda query: self.similarity_search(query, k))


if __name__ == "__main__":
    embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_stores = VectorStores(embedding)
    retriever = vector_stores.get_retriever()
    question = "我的体重180斤，尺码推荐？"
    docs = retriever.invoke(question)
    print(f"问题: {question}\n")
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        print(f"[{index}] {source}")
        print(doc.page_content.strip())
        print("-" * 40)
