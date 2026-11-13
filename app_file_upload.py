'''
基于streamlit的文件上传。
只把文件保存到 data 目录，不做解析和入库。
'''
import os

import streamlit as st

st.title("知识库更新服务")

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

uploaded_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    file_name = os.path.basename(uploaded_file.name)
    save_path = os.path.join(data_dir, file_name)
    if uploaded_file.size < 1024:
        size_text = f"{uploaded_file.size} B"
    else:
        size_text = f"{uploaded_file.size / 1024:.1f} KB"

    st.subheader(f"已选择文件: {file_name}")
    st.write(f"格式: {uploaded_file.type}，大小: {size_text}")

    if st.button("保存到 data"):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("文件已保存")
        st.text(save_path)
