'''
基于streamlit的文件上传
'''
import streamlit as st
from konwledge import KnowledgeBaseService

st.title("知识库更新服务")

#添加文件上传服务
uploader_file=st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False,
)

service=KnowledgeBaseService()

if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()

if uploader_file is not None:
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024

    st.subheader(f"成功上传文件: {file_name}")
    st.write(f"格式: {file_type}, 大小: {file_size} ")

    text=uploader_file.getvalue().decode("utf-8")
    st.write(text)

    result=st.session_state["service"].upload_by_str(text,file_name)
    st.write(result)