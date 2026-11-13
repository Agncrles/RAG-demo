from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings

import config_data as config
from konwledge import connect_db, get_collection


class VectorStores(object):
    def __init__(self, embedding):
        self.embedding = embedding
        conn = connect_db()
        conn.close()
        get_collection()

    def similarity_search(self, query, k):
        query_vector = self.embedding.embed_query(query)
        collection = get_collection()
        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, count),
        )
        documents = result["documents"][0] or []
        metadatas = result["metadatas"][0] or []
        return [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(documents, metadatas)
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
