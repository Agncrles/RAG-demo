from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config_data as config
import vector_stores as vs
import itertools
import sys
import threading

def _format_docs(docs):             # 将文档转换为字符串
    return "\n\n".join(doc.page_content for doc in docs)


class RagService(object):           # 定义RAG服务
    def __init__(self):
        self.vector_service = vs.VectorStores(config.embedding_function)
        self.prompt_template = PromptTemplate.from_template(config.prompt_template)
        self.chat_model = config.chat_model
        self.chain = self._get_chain()

    def _get_chain(self):            # 获取链
        retriever = self.vector_service.get_retriever()    # 获取检索器
        return (
            {
                "context": retriever | _format_docs,   #把检索结果转换为字符串
                "question": RunnablePassthrough(),#把无
            }
            | self.prompt_template   # 填好prompt模版，返回的是StringPromptValue类型
            | self.chat_model     #传给模型，模型返回AIMessage类型
            | StrOutputParser()   # 将模型输出转换为字符串
        )

    def ask(self, question):
        return self.chain.invoke(question)   # 调用链，返回AIMessage类型



if __name__ == "__main__":
    service = RagService()
    print(service.ask("我的体重180斤，尺码推荐什么？选哪些面料合适"))