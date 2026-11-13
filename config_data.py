from langchain_ollama import ChatOllama, OllamaEmbeddings

sqlite_path = "./knowledge.db"
embedding_model_name = "qwen3-embedding:0.6b"
embedding_function = OllamaEmbeddings(model=embedding_model_name)

chat_model_name = "qwen2.5:3b"
chat_model = ChatOllama(model=chat_model_name, temperature=0)

prompt_template = """你是服装知识库助手。请只根据参考资料回答问题。
资料里没有答案时，直接说“不知道”，不要编造。

参考资料：
{context}

问题：{question}
"""

chunk_size = 50
chunk_overlap = 0
separators = ["\n\n", "\n", " ","。","！","？","\n\n\n"]
max_split_char_number = 10000

similarity_threshold = 3