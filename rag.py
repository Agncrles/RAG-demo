'''
RAG 问答入口。
根据问题从向量库取出相关片段，填进提示词后交给对话模型回答。
资料里有答案时在末尾附上参考资料；没有对应资料时使用模型自己的回答，不再附来源。
'''
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import config_data as config
import vector_stores as vs

_MODEL_MARK = "【模型回答】"


def _own_answer(answer):#如果答案以【模型回答】开头，则返回答案，否则返回None   
    text = answer.lstrip()
    if not text.startswith(_MODEL_MARK):
        return None
    return text[len(_MODEL_MARK):].lstrip(" ：:\n")


def _format_docs(docs):
    parts=[]
    for doc in docs:        #从文档里提取来源和内容
        source=doc.metadata.get("source","未知来源")     
        parts.append(f"来源：{source}\n{doc.page_content}")        # 将文档转换为字符串
    return "\n\n".join(parts)   #返回的文档字符串，每个文档占一行，每行包含来源和内容


class RagService(object):           # 定义RAG服务
    def __init__(self):
        self.vector_service = vs.VectorStores(config.embedding_function)
        self.prompt_template = PromptTemplate.from_template(config.prompt_template)
        self.direct_template = PromptTemplate.from_template(config.direct_prompt)
        self.chat_model = config.chat_model
        self.chain = self._get_chain()
        self.direct_chain = self.direct_template | self.chat_model | StrOutputParser()

    def _get_chain(self):            # 获取链
        return (
            self.prompt_template   # 填好prompt模版，返回的是StringPromptValue类型
            | self.chat_model     #传给模型，模型返回AIMessage类型
            | StrOutputParser()   # 将模型输出转换为字符串
        )

    def ask(self, question):
        docs = self.vector_service.get_retriever().invoke(question)#从向量库取出与问题相关片段

        if not docs: #如果相关片段为空，直接返回模型回答
            return self.direct_chain.invoke({"question": question})
        answer = self.chain.invoke(#填好prompt模版，传给模型，模型返回AIMessage类型
            {"context": _format_docs(docs), "question": question}
        )
        own_answer = _own_answer(answer)#如果答案以【模型回答】开头，则返回答案，否则返回None
        if own_answer is not None:
            return own_answer
        sources = []
        for doc in docs: #遍历片段
            source = doc.metadata.get("source", "未知来源") #获取片段来源
            if source not in sources: #如果片段来源不在来源列表中，则添加到列表中
                sources.append(source)
        if not sources: #如果来源列表为空，直接返回模型回答
            return answer
        return answer.rstrip() + "\n参考资料：" + "、".join(sources)



if __name__ == "__main__":
    service = RagService()
    print(service.ask("怎么清理油烟机?"))
