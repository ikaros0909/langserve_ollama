import os
import re
import tempfile
import opendataloader_pdf
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ⭐️ Embedding 설정
# USE_BGE_EMBEDDING = True 로 설정시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB 다운로드 시간 걸릴 수 있습니다)
# USE_BGE_EMBEDDING = False 로 설정시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
USE_BGE_EMBEDDING = True

import os  # os 모듈 임포트
api_key_set = os.getenv("OPENAI_API_KEY")

if not USE_BGE_EMBEDDING:
    # OPENAI API KEY 입력
    # Embedding 을 무료 한글 임베딩으로 대체하면 필요 없음!
    os.environ["OPENAI_API_KEY"] = api_key_set

# ⭐️ LangServe 모델 설정(EndPoint)
# 1) REMOTE 접속: 본인의 REMOTE LANGSERVE 주소 입력
# (예시)
# LANGSERVE_ENDPOINT = "https://poodle-deep-marmot.ngrok-free.app/llm/"
# LANGSERVE_ENDPOINT = "https://4a7d-203-251-156-66.ngrok-free.app/llm/"
LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"

# 2) LocalHost 접속: 끝에 붙는 N4XyA 는 각자 다르니
# http://localhost:8000/llm/playground 에서 python SDK 에서 확인!
# LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트를 자유롭게 수정해 보세요!
#다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 답변하세요
RAG_PROMPT_TEMPLATE = """당신은 주어진 문서 내용만을 기반으로 답변하는 AI 입니다.
반드시 아래 [Context]에 제공된 내용만을 사용하여 답변하세요.
[Context]에 답이 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다."라고만 답하세요.
절대로 문서에 없는 내용을 추측하거나 지어내지 마세요.

[Question]
{question}

[Context]
{context}

[Answer]"""

st.set_page_config(page_title="JInhak Local 모델 테스트", page_icon="💬")
st.title("JInhak Local 모델 테스트")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]


def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def _build_retriever(file_path, cache_dir):
    """file_path의 문서를 청크로 분리하고 벡터 retriever를 반환하는 공통 함수"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    docs = []

    if file_path.endswith(".pdf"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            opendataloader_pdf.convert(
                input_path=file_path,
                output_dir=tmp_dir,
                format="markdown",
                table_method="cluster",
                reading_order="xycut",
                quiet=True,
            )
            md_file = os.path.join(tmp_dir, os.path.splitext(os.path.basename(file_path))[0] + ".md")
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_text = f.read()

        blocks = re.split(r'(?=^#{1,4} )', markdown_text, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            table_match = re.search(r'(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)', block)
            if table_match:
                context = block[:table_match.start()].strip()
                table_text = table_match.group(0)
                rows = [r for r in table_text.strip().split("\n") if r.strip()]
                if len(rows) >= 3:
                    headers = [c.strip() for c in rows[0].split("|") if c.strip()]
                    for row in rows[2:]:
                        cells = [c.strip() for c in row.split("|") if c.strip()]
                        if not any(cells):
                            continue
                        parts = []
                        for h, v in zip(headers, cells):
                            if v and v != "-":
                                parts.append(f"{h}: {v}" if h else v)
                        if parts:
                            content = (f"{context}\n" if context else "") + " | ".join(parts)
                            docs.append(Document(page_content=content, metadata={"source_type": "table"}))
                after = block[table_match.end():].strip()
                if after:
                    docs.extend(text_splitter.create_documents([after]))
            else:
                docs.extend(text_splitter.create_documents([block]))
    else:
        loader = UnstructuredFileLoader(file_path)
        raw_docs = loader.load()
        docs = text_splitter.split_documents(raw_docs)

    if USE_BGE_EMBEDDING:
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # FAISS 인덱스가 이미 저장되어 있으면 로드, 없으면 생성 후 저장
    faiss_path = str(cache_dir.root_path) + "_faiss"
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, cached_embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
        vectorstore.save_local(faiss_path)

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})


@st.cache_resource(show_spinner="Embedding file...")
def embed_cached_file(filename):
    """이미 .cache/files에 저장된 파일을 임베딩 (다른 브라우저에서 선택 시 사용)"""
    file_path = f"./.cache/files/{filename}"
    cache_dir = LocalFileStore(f"./.cache/embeddings/{filename}")
    return _build_retriever(file_path, cache_dir)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    return _build_retriever(file_path, cache_dir)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


import shutil

with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    # 저장된 파일 목록 — 접속 즉시 모두 사용, X 버튼으로 삭제
    cached_files = sorted([f for f in os.listdir(".cache/files") if not f.startswith(".")]) if os.path.exists(".cache/files") else []

    if cached_files:
        st.markdown("**사용 중인 파일**")
        for fname in cached_files:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("✕", key=f"del_{fname}"):
                st.session_state["confirm_delete"] = fname

    # 삭제 확인 다이얼로그
    if "confirm_delete" in st.session_state:
        target = st.session_state["confirm_delete"]
        st.warning(f"**'{target}'** 을 삭제할까요?")
        c1, c2 = st.columns(2)
        if c1.button("삭제 확인", type="primary"):
            for path in [
                f"./.cache/files/{target}",
            ]:
                if os.path.exists(path):
                    os.remove(path)
            for dirpath in [
                f"./.cache/embeddings/{target}",
                f"./.cache/embeddings/{target}_faiss",
            ]:
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
            del st.session_state["confirm_delete"]
            st.rerun()
        if c2.button("취소"):
            del st.session_state["confirm_delete"]
            st.rerun()

active_files = file if file else []

if active_files or cached_files:
    retrievers = []
    for f in active_files:
        retrievers.append(embed_file(f))
    for filename in cached_files:
        retrievers.append(embed_cached_file(filename))

    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        # 여러 벡터스토어 병합
        merged = retrievers[0].vectorstore
        for r in retrievers[1:]:
            merged.merge_from(r.vectorstore)
        retriever = merged.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})

print_history()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT, headers={"ngrok-skip-browser-warning": "1"})
        # LM Studio 모델 설정
        # ollama = ChatOpenAI(
        #     base_url="http://localhost:1234/v1",
        #     api_key="lm-studio",
        #     model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],  # 스트리밍 콜백 추가
        # )
        chat_container = st.empty()
        if active_files or cached_files:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
