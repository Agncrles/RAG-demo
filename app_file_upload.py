'''
基于 Streamlit 的知识库聊天页。
左侧列出历史对话，并可新开一段；上传放在左下角。
历史对话写入本地数据库，关掉页面后下次打开还在。
右侧只显示当前对话。每次提问仍单独检索，模型看不到更早的内容。
'''
import json
import os
import sqlite3
import uuid
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="知识库助手", layout="wide")

# 资料 txt、对话记录和向量索引的登记都放在脚本所在目录，不跟工作目录走
_base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_base_dir, "data")
sqlite_path = os.path.join(_base_dir, "knowledge.db")
os.makedirs(data_dir, exist_ok=True)

# 页面本身不滚动：左右两栏撑满视口，滚动只发生在历史列表和消息区
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 0.6rem;
    }
    /* 两栏拉到视口高度，给内部滚动留出固定空间 */
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
        height: calc(100vh - 5.5rem) !important;
        min-height: 0;
    }
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
        height: 100%;
        min-height: 0;
    }
    /* stretch 才能让子项被格子限制住；用 start 时内容会撑破外层并被裁掉，没有滚动条 */
    .stVerticalBlock.st-key-left_pane,
    .stVerticalBlock.st-key-chat_pane {
        display: grid !important;
        align-items: stretch !important;
        height: 100% !important;
        max-height: 100%;
        overflow: hidden;
    }
    /* 左：新对话、标题、历史列表、底部上传。右：标题、说明、消息、输入框 */
    .stVerticalBlock.st-key-left_pane {
        grid-template-rows: auto auto minmax(0, 1fr) auto;
    }
    .stVerticalBlock.st-key-chat_pane {
        grid-template-rows: auto auto minmax(0, 1fr) auto;
    }
    /* 第三行是唯一滚动的区域：历史列表和消息区 */
    .stVerticalBlock.st-key-left_pane > :nth-child(3),
    .stVerticalBlock.st-key-chat_pane > :nth-child(3) {
        min-height: 0 !important;
        height: auto !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
    }
    /* 左栏第四行是上传，不要被裁切 */
    .stVerticalBlock.st-key-left_pane > :nth-child(4) {
        min-height: 0;
        overflow: visible;
    }
    /* 真正滚动的是外层，里面的历史和消息按内容高度排，避免再套一层滚不动的盒子 */
    .stVerticalBlock.st-key-history,
    .stVerticalBlock.st-key-messages {
        height: auto !important;
        overflow: visible !important;
    }
    /* 贴底脚本注入的 html 不占消息区高度 */
    .st-key-messages [data-testid="stHtml"] {
        display: none;
    }
    .st-key-upload {
        gap: 0.25rem !important;
        padding: 0.45rem 0.6rem 0.35rem !important;
    }
    /* 去掉拖拽大框，只留上传按钮 */
    .st-key-upload [data-testid="stFileUploaderDropzone"] {
        min-height: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    .st-key-upload [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    /* 上传区里的分栏会继承上面的视口高度，这里改回按内容撑开 */
    .st-key-upload [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
    .st-key-upload [data-testid="stLayoutWrapper"],
    .st-key-upload [data-testid="stHorizontalBlock"] {
        height: auto !important;
        min-height: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_service():
    """创建并缓存问答服务，整个进程只初始化一次。

    RagService 会加载对话模型、向量库和重排序模型，第一次调用比较慢。
    用 cache_resource 是为了避免每次页面重跑都重新加载。
    延迟导入 rag，让页面在模型还没准备好时也能先画出来。

    返回:
        RagService: 提供 ask(问题) 的检索问答对象。只传入当前这一句，不带历史。
    """
    # 放在函数里导入：页面先画出来，模型相关依赖等到第一次提问再加载
    from rag import RagService

    return RagService()


@st.cache_resource
def get_knowledge_service():
    """创建并缓存知识库服务，整个进程只初始化一次。

    首次调用时先 sync_index()：切块长度或向量模型变了就清空旧向量再重建，
    避免新旧索引混在一起。之后的上传、切块、写入 Chroma 都走这个对象。

    返回:
        KnowledgeBaseService: 负责把本地 txt 解析并写入向量库。
    """
    from konwledge import KnowledgeBaseService, sync_index

    # 切块或向量模型配置变了会清掉旧索引，必须在服务创建前做完
    sync_index()
    return KnowledgeBaseService()


def list_documents():
    """读取已经入库的资料清单，供左下角「已入库」展开列表使用。

    数据来自 knowledge.db 的 documents 表，按入库时间从新到旧排列。
    数据库文件还不存在，或表还没建好（第一次打开、还没上传过），都返回空列表，
    不让页面因为缺表而报错。

    返回:
        list[dict]: 每项含 source（文件名）、chunk_count（切成多少段）、create_time。
    """
    # 库文件都还没有，说明从未入库
    if not os.path.exists(sqlite_path):
        return []
    conn = sqlite3.connect(sqlite_path)
    # 用列名取值，不靠下标
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT source, chunk_count, create_time
            FROM documents
            ORDER BY create_time DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        # 只建过对话表、documents 还没有时，查询会失败，按空库处理
        return []
    finally:
        conn.close()


def save_and_index(uploaded_file, knowledge_service):
    """把浏览器里选中的一个 txt 存到 data 目录，并交给知识库服务建索引。

    只保留文件名，丢掉上传路径，避免把本机目录写进库。文件按原始字节写入，
    编码检查放在调用方：不是 UTF-8 时 upload_by_file 会抛 UnicodeDecodeError。
    内容与已入库文件相同时，知识库服务返回「[已存在]」，不会重复切块。

    参数:
        uploaded_file: Streamlit 的 UploadedFile。
        knowledge_service: get_knowledge_service() 返回的对象。

    返回:
        tuple[str, str]: (保存后的文件名, 入库结果说明)。
    """
    # 丢掉客户端路径，只留文件名，避免写出 data 目录
    file_name = os.path.basename(uploaded_file.name)
    save_path = os.path.join(data_dir, file_name)
    # 按原始字节保存，是不是 UTF-8 交给 upload_by_file 判断
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    # 内容没变时返回「[已存在]」，不会重复切块
    result = knowledge_service.upload_by_file(save_path)
    return file_name, result


def _title_from(text):
    """用用户第一句话生成左侧历史按钮上的标题。

    连续空白压成一个空格。不超过 16 个字符就整句显示；更长则截断并加省略号。
    去掉空白后如果什么都不剩，退回「新对话」，避免出现空白按钮。

    参数:
        text: 当前对话里的第一条用户消息。

    返回:
        str: 侧栏按钮文案。
    """
    # 换行和连续空格收成一个空格，按钮文案里不换行
    compact = " ".join(text.split())
    if len(compact) <= 16:
        # 全是空白时 compact 为空，按钮不能没有字
        return compact or "新对话"
    return compact[:16] + "…"


def _blank_conversation():
    """造一段还没有消息的新对话，只存在于这次调用的返回值里。

    id 用去掉横线的 uuid，作为数据库主键和按钮 key，避免两次新建撞在一起。
    标题先写「新对话」，等用户发出第一句再由 _title_from 换掉。
    这里不写数据库，调用方决定要不要立刻保存。

    返回:
        dict: id、title、messages（空列表）。messages 只给界面用，不会喂给模型。
    """
    # hex 去掉横线，直接拿来当主键和按钮 key
    return {"id": uuid.uuid4().hex, "title": "新对话", "messages": []}


def _chat_db():
    """打开 knowledge.db，并保证对话相关的两张表存在。

    conversations 存每一段对话：id、标题、消息 JSON、最后更新时间。
    chat_state 存当前选中的对话 id（key 固定为 current_id），刷新后还能停在同一段。
    这两张表和资料表 documents 在同一个库里，互不覆盖。
    调用方用完必须自己 close；本函数不关闭连接。

    返回:
        sqlite3.Connection: 已设置 row_factory，可以用列名取字段。
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    # 一段对话一行。messages 是整段消息的 JSON，不是单独的消息表
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            messages TEXT NOT NULL,
            updated_time TEXT NOT NULL
        )
        """
    )
    # 只记当前选中哪一段，切换时不用改整段消息
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _save_conversation(conv):
    """把一段对话整份写入数据库；同一 id 已存在则覆盖标题、消息和更新时间。

    messages 用 JSON 存成一条文本，中文不转义。updated_time 取调用这一刻，
    所以刚发过消息的对话在 _load_conversations 里会排到最前。
    只保存界面上的往来记录，不参与检索，模型也读不到这里的内容。

    参数:
        conv: 含 id、title、messages 的字典，结构与 _blank_conversation 相同。
    """
    conn = _chat_db()
    try:
        conn.execute(
            """
            INSERT INTO conversations (id, title, messages, updated_time)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                messages = excluded.messages,
                updated_time = excluded.updated_time
            """,
            (
                conv["id"],
                conv["title"],
                # ensure_ascii=False 让中文原样落库，而不是写成 \uXXXX
                json.dumps(conv["messages"], ensure_ascii=False),
                # 用保存时刻排序，后发的对话在左侧排到前面
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    finally:
        # 写入失败也要关掉连接，避免占着数据库
        conn.close()


def _save_current_id(current_id):
    """记下当前正在看的对话 id，下次打开页面时默认回到这一段。

    chat_state 里只有 current_id 这一行，重复调用会改掉原来的值。
    切换历史、新开对话、发出消息时都要调用，否则刷新后会停在别的对话上。

    参数:
        current_id: conversations.id。
    """
    conn = _chat_db()
    try:
        # 表里永远只有 current_id 这一行，再次保存就是改掉原来的值
        conn.execute(
            """
            INSERT INTO chat_state (key, value)
            VALUES ('current_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (current_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _load_conversations():
    """从数据库读出全部历史对话，以及上次选中的那一段。

    按 updated_time 从新到旧排列，左侧列表直接按这个顺序画。
    messages 从 JSON 还原成列表。current_id 在库里没有、或指向已删除的对话时，
    改用最新的一段，避免界面选中一个不存在的 id。库是空的则 current_id 为 None。

    返回:
        tuple[list[dict], str | None]: (对话列表, 应选中的 id)。
    """
    conn = _chat_db()
    try:
        # 新更新的在前，左侧列表按这个顺序直接画
        rows = conn.execute(
            """
            SELECT id, title, messages
            FROM conversations
            ORDER BY updated_time DESC
            """
        ).fetchall()
        current = conn.execute(
            "SELECT value FROM chat_state WHERE key = 'current_id'"
        ).fetchone()
    finally:
        conn.close()
    conversations = []
    for row in rows:
        conversations.append(
            {
                "id": row["id"],
                "title": row["title"],
                # 库存的是 JSON 字符串，界面要的是消息列表
                "messages": json.loads(row["messages"]),
            }
        )
    # 还没记过选中项时 current 是 None
    current_id = current["value"] if current else None
    # 选中项被删掉或不在列表里时，改看最新的一段
    if conversations and current_id not in {item["id"] for item in conversations}:
        current_id = conversations[0]["id"]
    return conversations, current_id


def _ensure_state():
    """保证本次会话的内存里已经有对话列表和当前 id。

    Streamlit 每次交互都会从头执行脚本，但 session_state 会留下来。
    因此只在本会话第一次进入、键还不存在时才读库。读到的内容一直留在内存里，
    后面的按钮和发消息都改这份内存，再由保存函数写回数据库。
    库里一条对话都没有时，新建一段空白对话并立刻落库，左侧至少有一个「新对话」。
    已经在内存里的旧列表不会在这里补写；那些内容若没经过 _save_conversation，关掉服务就没了。
    """
    # 同一浏览器会话里键已经在，说明读过库，不要每次重跑都覆盖内存里的对话
    if "conversations" not in st.session_state:
        conversations, current_id = _load_conversations()
        if not conversations:
            # 库是空的，先落一段空白对话，左侧至少有一个按钮
            conv = _blank_conversation()
            _save_conversation(conv)
            _save_current_id(conv["id"])
            conversations = [conv]
            current_id = conv["id"]
        st.session_state.conversations = conversations
        st.session_state.current_id = current_id


def _current():
    """返回当前选中的那段对话，右侧消息区画的就是它。

    按 current_id 在内存列表里查找，返回的是列表中的原字典，改 messages 或 title
    会直接反映到 session_state。id 对不上任何一段时（例如状态被改乱），
    退回列表第一段并改写 current_id，保证后面的渲染一定拿到一段对话。

    返回:
        dict: 当前对话，含 id、title、messages。
    """
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_id:
            # 返回列表里的原字典，改 title 或 messages 会直接反映到 session_state
            return conv
    # 选中 id 找不到时用列表第一段，并改掉失效的 id
    conv = st.session_state.conversations[0]
    st.session_state.current_id = conv["id"]
    return conv


def _start_new():
    """在左侧最上面新开一段空白对话，并把它设为当前对话。

    当前这段还没有任何消息时直接返回，避免连续点「新对话」堆出一串空记录。
    新对话插到列表开头，同时写入 conversations 和 chat_state，刷新后仍是这一段。
    """
    current = _current()
    # 当前还是空的，再开一段只会多一个空按钮
    if not current["messages"]:
        return
    conv = _blank_conversation()
    # 插到开头，和「最近更新在最前」一致
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_id = conv["id"]
    _save_conversation(conv)
    _save_current_id(conv["id"])


def _promote(conv):
    """把刚更新过的对话挪到内存列表最前面，和数据库按更新时间倒序一致。

    只改 session_state 里的顺序，不写数据库。排序持久化靠 _save_conversation
    刷新 updated_time，下次从库加载时自然排到第一位。

    参数:
        conv: 已经在 st.session_state.conversations 里的那一段。
    """
    conversations = st.session_state.conversations
    # 先摘掉再插到开头，避免列表里出现两份同一段对话
    conversations.remove(conv)
    conversations.insert(0, conv)


def _render_messages(messages):
    """把一段对话里已经有的消息画成聊天气泡。

    role 为 user 或 assistant，对应 Streamlit 左右两种气泡。
    内容按 Markdown 显示。这里不发请求，只回放内存或数据库里存好的文本。

    参数:
        messages: [{"role": "user"|"assistant", "content": str}, ...]。
    """
    for message in messages:
        # role 为 user 或 assistant，决定气泡在右还是在左
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _install_chat_pin():
    """往页面注入一段脚本，让右侧消息区在需要时自己滚到最底部。

    真正可滚动的是 chat_pane 里 overflow-y 为 auto 的那层布局容器，不是整个页面。
    脚本包在立即执行函数里：经典 script 标签的顶层 return 会直接语法错误，后面全不执行。
    window.__chatPinInstalled 保证多次重跑只装一个定时器。定时器每 200 毫秒看一次
    window.__forcePin，为真就把 scrollTop 拉到 scrollHeight。
    用户向上滚轮时清掉这个标记，不再把视口拽回去；程序自己设置的滚动不取消标记。
    注入的 html 容器被样式藏起来，不占消息区高度。
    """
    st.html(
        """
        <script>
        // 必须包在立即执行函数里，顶层 return 会让整段脚本语法错误
        (() => {
          // 页面每次重跑都会再注入这段脚本，定时器只装一次
          if (window.__chatPinInstalled) return;
          window.__chatPinInstalled = true;
          const scroller = () => {
            const pane = document.querySelector(".st-key-chat_pane");
            if (!pane) return null;
            // 真正滚动的是 overflow-y 为 auto 的布局层，找不到就用第三个子节点
            return [...pane.querySelectorAll('[data-testid="stLayoutWrapper"]')]
              .find((el) => getComputedStyle(el).overflowY === "auto")
              || pane.children[2];
          };
          setInterval(() => {
            const box = scroller();
            // __forcePin 为假时不跟到底部，留给用户自己翻上面的消息
            if (!box || !window.__forcePin) return;
            if (!box.__pinBound) {
              box.__pinBound = true;
              // 只在向上滚轮时取消贴底，程序设置 scrollTop 不当作用户离开底部
              box.addEventListener("wheel", (event) => {
                if (event.deltaY < 0) window.__forcePin = false;
              }, {passive: true});
            }
            box.scrollTop = box.scrollHeight;
          }, 200);
        })();
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )


def _pin_chat_to_bottom():
    """打开自动贴底。发出消息后、以及回答完成重新渲染后各调用一次。

    只把 window.__forcePin 设为真，真正滚动由 _install_chat_pin 里的定时器完成。
    回答生成要几秒，定时器会在这段时间里持续把新出现的内容拉进视野。
    """
    # 只打开标记，滚动由上面的定时器执行；回答生成的几秒里会一直贴住底部
    st.html(
        "<script>(() => { window.__forcePin = true; })();</script>",
        width="content",
        unsafe_allow_javascript=True,
    )


# 每次重跑先保证内存里有对话列表；已有则不会重新读库
_ensure_state()

# 左窄右宽：历史和上传在左，当前对话在右
left, right = st.columns([1, 2.6], gap="medium")

with left:
    with st.container(key="left_pane"):
        if st.button("新对话", width="stretch"):
            _start_new()
        st.caption("历史对话")
        with st.container(key="history"):
            for conv in st.session_state.conversations:
                selected = conv["id"] == st.session_state.current_id
                if st.button(
                    conv["title"],
                    key=f"conv-{conv['id']}",
                    width="stretch",
                    # 当前这段用主色，其余用次要按钮
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.current_id = conv["id"]
                    # 选中项写入库，刷新后还停在这一段
                    _save_current_id(conv["id"])
        with st.container(key="upload", border=True):
            file_col, action_col = st.columns([1, 1], gap="small", vertical_alignment="center")
            with file_col:
                uploaded_files = st.file_uploader(
                    "上传 txt",
                    type=["txt"],
                    accept_multiple_files=True,
                    # 标签收起，上传区只留一个按钮，避免左下角被拖拽框撑高
                    label_visibility="collapsed",
                )
            with action_col:
                index_clicked = st.button("解析并入库", width="stretch")
            if index_clicked:
                if not uploaded_files:
                    st.warning("请先选择 txt 文件")
                else:
                    # 重排序和向量模型只在真正入库时加载
                    knowledge_service = get_knowledge_service()
                    for uploaded_file in uploaded_files:
                        with st.spinner(f"正在解析 {uploaded_file.name}"):
                            try:
                                file_name, result = save_and_index(
                                    uploaded_file, knowledge_service
                                )
                            except UnicodeDecodeError:
                                # 解码失败的文件不入库，继续处理下一个
                                st.error(f"{uploaded_file.name} 不是 UTF-8 文本，没有入库")
                                continue
                            except OSError as exc:
                                st.error(f"{uploaded_file.name} 保存失败：{exc}")
                                continue
                        if result == "[已存在]":
                            st.info(f"{file_name}：内容没变，沿用已有索引")
                        else:
                            st.success(f"{file_name}：{result}")
            documents = list_documents()
            with st.expander(f"已入库（{len(documents)}）"):
                if not documents:
                    st.write("还没有资料")
                else:
                    for doc in documents:
                        st.write(f"{doc['source']} · {doc['chunk_count']} 段")
                        st.caption(doc["create_time"])

current = _current()

with right:
    with st.container(key="chat_pane"):
        st.markdown("### 知识库助手")
        st.caption("资料里有答案就按资料回答，没有就用模型自己的知识。历史对话只显示在左侧，模型每次只看当前这句话。")
        # 消息容器先建出来，内容放在下面的 with message_box 里画
        message_box = st.container(key="messages")
        # 输入框放在消息区后面，样式把它固定在右侧底部
        prompt = st.chat_input("输入问题，例如：手冲咖啡水温怎么控制？")
        # 每次重跑都注入贴底脚本；脚本内部会保证定时器只装一次
        _install_chat_pin()

with message_box:
    # 还没消息、这一轮也没输入时，给一句空状态提示
    if not current["messages"] and not prompt:
        st.caption("从这里开始提问。")
    # 先回放已经保存的消息，新的一句画在它们下面
    _render_messages(current["messages"])
    if prompt:
        # 这段对话的第一条用户消息用来当左侧标题
        if not any(item["role"] == "user" for item in current["messages"]):
            current["title"] = _title_from(prompt)
        current["messages"].append({"role": "user", "content": prompt})
        # 回答还没出来先落库，中途刷新也不会丢掉刚发的问题
        _save_conversation(current)
        _save_current_id(current["id"])
        with st.chat_message("user"):
            st.markdown(prompt)
        # 检索要几秒，先打开贴底，生成过程中视口跟着往下走
        _pin_chat_to_bottom()
        with st.chat_message("assistant"):
            with st.spinner("正在检索并回答"):
                # 只把当前这句交给模型，不附带更早的 messages
                answer = get_rag_service().ask(prompt)
            st.markdown(answer)
        current["messages"].append({"role": "assistant", "content": answer})
        # 刚聊过的这段挪到历史列表最前
        _promote(current)
        _save_conversation(current)
        _save_current_id(current["id"])
        # rerun 会清掉这一轮的 prompt，用标记让下一次渲染继续贴底
        st.session_state.pin_chat = True
        st.rerun()
    elif st.session_state.pop("pin_chat", False):
        # 回答已经画进历史气泡后，再贴一次底，然后清掉标记
        _pin_chat_to_bottom()
