'''
集中存放本案例的配置。
包括数据库路径、嵌入模型和对话模型、提示词、文本切块规则、向量库参数，以及检索和重排序数量。
'''
import os

from langchain_ollama import ChatOllama, OllamaEmbeddings

_base_dir = os.path.dirname(os.path.abspath(__file__))
sqlite_path = os.path.join(_base_dir, "knowledge.db")
embedding_model_name = "qwen3-embedding:0.6b"
embedding_function = OllamaEmbeddings(model=embedding_model_name)

chat_model_name = "qwen2.5:3b"
chat_model = ChatOllama(model=chat_model_name, temperature=0)

prompt_template = """你是通用知识库助手。
优先根据参考资料回答。资料里有多条相关内容时，逐条回答，不要漏掉。
参考资料里没有答案时，用你自己的知识回答。这时第一行只能写“【模型回答】”，从第二行开始写答案。
根据资料回答时，不要写“【模型回答】”，也不要写参考资料编号或文件名。

参考资料：
{context}

问题：{question}
"""

direct_prompt = """直接回答下面的问题。

问题：{question}
"""

chunk_size = 300
chunk_overlap = 40
separators = ["\n\n", "\n", " ","。","！","？","\n\n\n"]
max_split_char_number = 10000


chroma_path = os.path.join(_base_dir, "chroma_db")
chroma_collection = "knowledge"
chroma_space = "cosine"

retrieve_k = 10          # 向量检索先拿 10 段
rerank_k = 3             # 重排后只留 3 段给对话模型
rerank_score_min = 0.35  # 低于这个分数的片段视为不相关，不送给对话模型
rerank_model_name = "BAAI/bge-reranker-base"
_rerank_root = os.path.join(_base_dir, "models", "models--BAAI--bge-reranker-base")
with open(os.path.join(_rerank_root, "refs", "main"), encoding="utf-8") as _rerank_ref:
    _rerank_revision = _rerank_ref.read().strip()
rerank_model_path = os.path.join(_rerank_root, "snapshots", _rerank_revision)