import hashlib
import os
import sqlite3
from datetime import datetime

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
import itertools
import sys
import threading

def connect_db():           #连接数据库
    conn = sqlite3.connect(config.sqlite_path)    #连接数据库文件，数据库文件的路径写在config_data.py文件中
    conn.row_factory = sqlite3.Row    #把数据库里的数据按照行进行处理，可以按列名进行访问
    conn.execute(       #执行sql语句，创建documents表，如果表不存在则创建，表结构如下：
        """
        CREATE TABLE IF NOT EXISTS documents (    
            source TEXT PRIMARY KEY,    
            content_md5 TEXT NOT NULL,    
            chunk_count INTEGER NOT NULL,    
            create_time TEXT NOT NULL,    
            operator TEXT NOT NULL   
        )
        """
    )
    conn.commit()       #提交事务
    return conn


def get_collection():
    client = chromadb.PersistentClient(path=config.chroma_path)
    return client.get_or_create_collection(
        name=config.chroma_collection,
        metadata={"hnsw:space": "l2"},
    )


def get_string_md5(input_str: str, encoding="utf-8") -> str:
    str_bytes = input_str.encode(encoding=encoding)     #将输入的字符串转为字节数组
    md5_obj = hashlib.md5()    #创建一个md5对象
    md5_obj.update(str_bytes)    #更新md5对象
    return md5_obj.hexdigest() #返回一个md5值，用来校验文件


class KnowledgeBaseService(object):
    def __init__(self):
        self.spliter = RecursiveCharacterTextSplitter(   #创建一个RecursiveCharacterTextSplitter对象，用来分割文本
            chunk_size=config.chunk_size,    #设置分割的块大小
            chunk_overlap=config.chunk_overlap,    #设置分割的块重叠大小
            separators=config.separators,    #设置分割的分割符
            length_function=len,    #设置分割的长度函数
        )
        conn = connect_db()   #连接数据库
        conn.close()   #关闭数据库连接

    def upload_by_str(self, data, filename):   #上传文本
        md5_hex = get_string_md5(data)   #获取文本的md5值，写入数据库，用于校验文件
        conn = connect_db()
        try:
            row = conn.execute(
                "SELECT content_md5 FROM documents WHERE source = ?",
                (filename,),   #查询数据库中是否存在该文件
            ).fetchone()   #获取查询结果
            if row and row["content_md5"] == md5_hex:
                return "[已存在]"   #如果文件存在，则返回已存在

            knowledge_chunks = [   #分割文本
                chunk.strip()
                for chunk in self.spliter.split_text(data)
                if chunk.strip()   #如果文本不为空，则分割
            ]
            if not knowledge_chunks:
                return "[成功] 0段"   #如果文本为空，则返回成功，0段

            vectors = config.embedding_function.embed_documents(knowledge_chunks)   #把分割好的chunk进行向量化
            create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   #获取当前时间，会一并写入数据库
            operator = "admin"   #操作人，写入数据库
            collection = get_collection()
            collection.delete(where={"source": filename})   #删掉这个文件在索引里的旧片段
            collection.add(
                ids=[f"{filename}-{i}" for i in range(len(knowledge_chunks))],
                documents=knowledge_chunks,
                embeddings=vectors,
                metadatas=[
                    {
                        "source": filename,
                        "create_time": create_time,
                        "operator": operator,
                    }
                    for _ in knowledge_chunks
                ],
            )
            with conn:
                conn.execute(    #执行sql语句，插入数据到documents表
                    """
                    INSERT INTO documents (
                        source, content_md5, chunk_count, create_time, operator
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        content_md5 = excluded.content_md5,
                        chunk_count = excluded.chunk_count,
                        create_time = excluded.create_time,
                        operator = excluded.operator
                    """,
                    (filename, md5_hex, len(knowledge_chunks), create_time, operator),
                )
            return f"[成功] {len(knowledge_chunks)}段"
        finally:
            conn.close()

    def upload_by_file(self, file_path):        #上传文件
        with open(file_path, "r", encoding="utf-8") as f:   #以读取模式打开文件，编码方式为utf-8
            data = f.read()
        filename = os.path.basename(file_path)   #获取文件名
        return self.upload_by_str(data, filename)   #把解析好的文件名和内容交给upload_by_str函数处理

def _spin(stop_event, label):   #定义一个旋转动画函数，label是前面的文字，用来显示进度
    frames = itertools.cycle("|/-\\")    #循环动画，|/-\\四个符号循环显示
    while not stop_event.is_set():   #如果停止事件没有被设置，则继续显示
        sys.stdout.write(f"\r{label} {next(frames)}")   #显示进度，frames是循环帧，next(frames)是获取下一个帧
        sys.stdout.flush()   #刷新缓冲区，确保显示的内容立即显示
        stop_event.wait(0.1)   #等待0.1秒，确保动画不会太快
    sys.stdout.write("\r" + " " * (len(label) + 2) + "\r")   #清空进度，确保动画不会残留
    sys.stdout.flush()   #刷新缓冲区，确保清空的内容立即显示


if __name__ == "__main__":
    knowledge_base_service = KnowledgeBaseService()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    names = [
        name for name in sorted(os.listdir(data_dir))
        if os.path.isfile(os.path.join(data_dir, name))
    ]
    total = len(names)
    for index, name in enumerate(names, start=1):
        file_path = os.path.join(data_dir, name)
        stop_event = threading.Event()
        label = f"[{index}/{total}] 正在入库 {name}"
        worker = threading.Thread(target=_spin, args=(stop_event, label), daemon=True)
        worker.start()
        try:
            result = knowledge_base_service.upload_by_file(file_path)
        finally:
            stop_event.set()
            worker.join()
        print(f"[{index}/{total}] {file_path} {result}")
