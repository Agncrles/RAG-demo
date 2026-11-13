'''
向量检索与重排序。
先按相似度从 Chroma 取出候选片段，再用本地重排序模型留下最相关的几段。
分数低于配置门槛的片段直接丢掉，供问答使用。
'''
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings

import config_data as config
from konwledge import connect_db, get_collection
from sentence_transformers import CrossEncoder

class VectorStores(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.reranker = CrossEncoder(
            config.rerank_model_path,
            local_files_only=True,
        )
        conn = connect_db()
        conn.close()
        get_collection()

    def rerank(self,query,docs,k):
        if not docs:    #如果文档为空，则返回空列表
            return []
        pairs=[(query,doc.page_content) for doc in docs]   #把查询问题和文档内容打包成pairs
        scores=self.reranker.predict(pairs)   #对pairs进行重排序
        ranked=sorted(zip(docs,scores),key=lambda item:float(item[1]),reverse=True)   #把文档和得分打包成ranked
        kept=[doc for doc,score in ranked if float(score) >= config.rerank_score_min]
        return kept[:k]   #丢掉低分片段后，返回前k个文档

    def similarity_search(self, query, k):
        query_vector = self.embedding.embed_query(query)#把用户问题转换为向量
        collection = get_collection()#获取向量库
        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, count),
        )
        documents = result["documents"][0] or []#获取查询结果的文档内容
        metadatas = result["metadatas"][0] or []#获取查询结果的相关数据
        return [        #把文档内容和相关数据打包成Document对象返回
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(documents, metadatas)  #把文档内容和相关数据打包成Document对象返回
        ]

    def get_retriever(self):
        def retrieve(query):
            candidates=self.similarity_search(query,config.retrieve_k)
            return self.rerank(query,candidates,config.rerank_k)
        return RunnableLambda(retrieve)


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

