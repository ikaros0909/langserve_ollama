import os
import sys
import re
import tempfile
import opendataloader_pdf
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
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
API_BASE_URL = "http://localhost:8000"

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

import requests

# --- 케밥 메뉴 (상단 오른쪽) ---
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

    from hashlib import sha256
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir,
        key_encoder=lambda x: sha256(x).hexdigest(),
    )

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
from langchain_ollama import ChatOllama

# 사용 가능한 모델
GUI_MODELS = {
    "exaone3.5:32b": "EXAONE 3.5 32B (기본)",
    "huihui_ai/kanana-nano-abliterated": "Kanana Nano (Kakao)",
    "gemma4:31b": "Gemma 4 31B Dense (Google)",
    "gemma4:26b": "Gemma 4 26B MoE (Google)",
}
MULTIMODAL_MODELS = {"gemma4:26b", "gemma4:31b"}

with st.sidebar:
    st.markdown("**모델 선택**")
    selected_model = st.selectbox(
        "LLM 모델",
        options=list(GUI_MODELS.keys()),
        format_func=lambda x: GUI_MODELS[x],
        index=0,
    )

    # 멀티모달 모델 선택 시 이미지 첨부
    if selected_model in MULTIMODAL_MODELS:
        img_upload = st.file_uploader(
            "이미지 첨부",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            key="chat_images",
            help="이미지를 첨부하면 멀티모달 모델이 분석합니다",
        )
        if img_upload:
            image_data = []
            for img in img_upload:
                img_bytes = img.read()
                image_data.append({
                    "bytes": img_bytes,
                    "type": img.type.split("/")[-1],
                    "name": img.name,
                })
            st.session_state["pending_images"] = image_data
        else:
            st.session_state["pending_images"] = []

        # 동영상 첨부
        video_upload = st.file_uploader(
            "동영상 첨부",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="chat_video",
            help="동영상을 업로드하면 음성을 텍스트로 변환하고 주요 장면을 분석합니다",
        )
        if video_upload:
            st.session_state["pending_video"] = {
                "bytes": video_upload.read(),
                "name": video_upload.name,
            }
        else:
            st.session_state["pending_video"] = None

    file = st.file_uploader(
        "RAG 파일 업로드",
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

    # --- API 키 관리 (사이드바 하단) ---
    st.markdown("---")
    with st.expander("API 키 관리"):
        # 키 생성
        with st.form("create_key_form"):
            key_name = st.text_input("키 이름", placeholder="예: 모바일 앱")
            submitted = st.form_submit_button("새 API 키 생성")
            if submitted and key_name:
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/keys/create", json={"name": key_name})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("**Secret Key는 지금만 확인 가능!**")
                        st.code(f"API Key:    {data['api_key']}\nSecret Key: {data['secret_key']}", language="text")
                    else:
                        st.error("키 생성 실패")
                except requests.ConnectionError:
                    st.error("서버에 연결할 수 없습니다.")

        # Endpoint
        st.caption("API Endpoint")
        st.code(f"{API_BASE_URL}/api/chat", language="text")

        # 키 목록
        try:
            resp = requests.get(f"{API_BASE_URL}/api/keys/list")
            if resp.status_code == 200:
                keys = resp.json()
                if not keys:
                    st.info("등록된 키 없음")
                for k in keys:
                    status = "활성" if k["is_active"] else "비활성"
                    st.markdown(f"**{k['name']}** `{k['api_key'][:12]}...` ({status})")
                    if k["is_active"]:
                        if st.button("비활성화", key=f"revoke_{k['api_key']}"):
                            requests.post(f"{API_BASE_URL}/api/keys/revoke/{k['api_key']}")
                            st.rerun()
                    else:
                        if st.button("삭제", key=f"delete_{k['api_key']}"):
                            requests.delete(f"{API_BASE_URL}/api/keys/delete/{k['api_key']}")
                            st.rerun()
        except requests.ConnectionError:
            st.error("서버에 연결할 수 없습니다.")

        st.caption("**API 사용 가이드**")
        st.markdown(f"""
**Endpoint**
`POST {API_BASE_URL}/api/chat`

**요청 필드**

| 필드 | 필수 | 설명 |
|------|:----:|------|
| `message` | O | 사용자 질문 |
| `system_prompt` | - | AI 역할/지시 설정 |
| `model` | - | 모델 선택 (기본: exaone3.5:32b) |
| `temperature` | - | 창의성 0~1 (기본: 0.5) |
| `images` | - | base64 이미지 배열 (gemma4만) |

**`message` vs `system_prompt`**
- `system_prompt`: AI의 역할과 행동 규칙을 지시 (시스템 프롬프트)
- `message`: 실제 사용자 질문 (사용자 프롬프트)
        """)
        st.code('''{
  "system_prompt": "당신은 연세대 입학처 AI 상담원입니다. 모집요강 기반으로만 답변하세요.",
  "message": "수시 추천형 지원 자격이 뭐야?",
  "model": "exaone3.5:32b",
  "temperature": 0.5
}''', language="json")
        st.markdown(f"""
**모델 목록**

| model | 멀티모달 | 설명 |
|-------|:--------:|------|
| `exaone3.5:32b` | - | LG AI 한국어 32B (기본) |
| `huihui_ai/kanana-nano-abliterated` | - | Kakao 이중언어 |
| `gemma4:31b` | O | Google Dense 32B 멀티모달 |
| `gemma4:26b` | O | Google MoE 26B 멀티모달 |

**curl 예시 - 기본**
```bash
curl -X POST {API_BASE_URL}/api/chat \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: jk-..." \\
  -H "X-Secret-Key: sk-..." \\
  -d '{{"message": "안녕하세요"}}'
```

**curl 예시 - 시스템 프롬프트 + 모델 지정**
```bash
curl -X POST {API_BASE_URL}/api/chat \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: jk-..." \\
  -H "X-Secret-Key: sk-..." \\
  -d '{{"message": "장학금 종류 알려줘", "system_prompt": "당신은 입학처 상담원입니다.", "model": "gemma4:26b", "temperature": 0.3}}'
```

**curl 예시 - 이미지 분석 (gemma4 전용, 파일 직접 업로드)**
`POST {API_BASE_URL}/api/chat/upload` (multipart/form-data)
```bash
# 이미지 1장 분석
curl -X POST {API_BASE_URL}/api/chat/upload \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "message=이 이미지를 설명해줘" \\
  -F "model=gemma4:26b" \\
  -F "images=@사진.jpg"
```

```bash
# 여러 이미지 동시 분석
curl -X POST {API_BASE_URL}/api/chat/upload \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "message=두 이미지를 비교해줘" \\
  -F "model=gemma4:26b" \\
  -F "images=@사진1.jpg" \\
  -F "images=@사진2.jpg"
```

**응답 형식**
```json
{{"answer": "답변", "model": "exaone3.5:32b", "usage": {{"input_tokens": 150, "output_tokens": 200, "total_tokens": 350}}}}
```

---

**자막/스크립트 생성**
`POST {API_BASE_URL}/api/transcribe` (multipart/form-data)

| 필드 | 필수 | 설명 |
|------|:----:|------|
| `file` | O | 영상 또는 음성 파일 |
| `whisper_model` | - | tiny, base(기본), small, medium, large |
| `format` | - | json(기본), srt, vtt, text |
| `language` | - | ko, en, 빈값=자동감지 |
| `prompt` | - | Whisper 힌트 (전문 용어, 고유명사, 화자 등) |

지원 파일: mp4, avi, mov, mkv, webm, mp3, wav, m4a, ogg, flac

`prompt` 활용 예시:
- 전문 용어 힌트: `"LangChain, RAG, Ollama, Streamlit"`
- 화자 표기 힌트: `"화자: 김교수, 이학생. 김교수가 강의하고 이학생이 질문한다."`
- 표기법 지정: `"연세대학교, 2026학년도, 수시모집"`

```bash
# 영상 → SRT 자막
curl -X POST {API_BASE_URL}/api/transcribe \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "file=@강의.mp4" \\
  -F "format=srt" \\
  -F "language=ko" \\
  -F "prompt=LangChain, RAG, Ollama" \\
  --max-time 1800 -o result.srt
```

```bash
# 음성 → 텍스트만
curl -X POST {API_BASE_URL}/api/transcribe \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "file=@녹음.mp3" \\
  -F "format=text"
```

---

**화자 분리 자막 (Whisper + pyannote)**
`POST {API_BASE_URL}/api/transcribe/diarize` (multipart/form-data)

| 필드 | 필수 | 설명 |
|------|:----:|------|
| `file` | O | 영상 또는 음성 파일 |
| `hf_token` | O | HuggingFace 토큰 (pyannote 모델 접근용) |
| `whisper_model` | - | tiny, base(기본), small, medium, large |
| `format` | - | json(기본), srt, text |
| `language` | - | ko, en, 빈값=자동감지 |
| `prompt` | - | Whisper 힌트 |
| `speaker_names` | - | 화자 이름 매핑 JSON |

```bash
# 화자 분리 자막 생성
curl -X POST {API_BASE_URL}/api/transcribe/diarize \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "file=@강의.mp4" \\
  -F "format=srt" \\
  -F "language=ko" \\
  -F "hf_token=hf_..." \\
  -F 'speaker_names={{"SPEAKER_00": "김교수", "SPEAKER_01": "이학생"}}' \\
  --max-time 1800 -o result.srt
```

SRT 결과 예시:
```
1
00:00:00,000 --> 00:00:03,500
[김교수] 오늘은 미분방정식에 대해 알아보겠습니다.

2
00:00:03,500 --> 00:00:07,200
[이학생] 교수님, 질문이 있습니다.
```

---

**동영상 분석 (Whisper + Gemma 4)**
`POST {API_BASE_URL}/api/video` (multipart/form-data)

| 필드 | 필수 | 설명 |
|------|:----:|------|
| `file` | O | 동영상 파일 |
| `message` | - | 질문 (기본: 동영상 내용 분석) |
| `model` | - | 분석 모델 (기본: gemma4:26b) |
| `whisper_model` | - | STT 모델 (기본: base) |
| `max_frames` | - | 분석할 장면 수 (기본: 5) |
| `language` | - | ko, en, 빈값=자동감지 |

```bash
curl -X POST {API_BASE_URL}/api/video \\
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \\
  -F "file=@강의.mp4" \\
  -F "message=이 강의의 핵심 내용을 요약해줘" \\
  -F "language=ko" \\
  --max-time 1800
```

---

**기타 엔드포인트**
- `GET {API_BASE_URL}/api/health` — 서버 상태
- `GET {API_BASE_URL}/api/models` — 모델 목록

**오류 코드**: `400` 모델/파일 오류 | `401` 인증 누락 | `403` 키 무효 | `429` 한도 초과(분당 30회)
        """)

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

import base64
from langchain_core.messages import HumanMessage as HMsg, SystemMessage as SMsg

# 첨부 상태 관리
if "pending_images" not in st.session_state:
    st.session_state["pending_images"] = []
if "pending_video" not in st.session_state:
    st.session_state["pending_video"] = None

# 첨부 미리보기
if st.session_state["pending_images"]:
    preview_cols = st.columns(min(len(st.session_state["pending_images"]), 4))
    for i, img_data in enumerate(st.session_state["pending_images"]):
        preview_cols[i % 4].image(img_data["bytes"], width=80)
if st.session_state.get("pending_video"):
    st.info(f"동영상 첨부됨: {st.session_state['pending_video']['name']}")

# chat_input은 항상 단독 사용 (하단 고정 유지)
user_input = st.chat_input("메시지를 입력하세요")

uploaded_images = []
pending_video = None
if user_input:
    if st.session_state["pending_images"]:
        uploaded_images = st.session_state["pending_images"]
        st.session_state["pending_images"] = []
    if st.session_state.get("pending_video"):
        pending_video = st.session_state["pending_video"]
        st.session_state["pending_video"] = None

if user_input:
    add_history("user", user_input)
    st.chat_message("user").write(user_input)

    # 이미지 미리보기
    if uploaded_images:
        with st.chat_message("user"):
            cols = st.columns(min(len(uploaded_images), 4))
            for i, img in enumerate(uploaded_images):
                cols[i % 4].image(img["bytes"], width=120)

    with st.chat_message("assistant"):
        model_label = GUI_MODELS[selected_model]
        ollama = ChatOllama(model=selected_model, temperature=0.5)
        chat_container = st.empty()

        if pending_video and selected_model in MULTIMODAL_MODELS:
            # 동영상 처리: Whisper STT + Gemma 4 프레임 분석
            import tempfile as _tmpfile
            chat_container.markdown("동영상 분석 중...")

            # 임시 파일에 저장
            tmp_dir = _tmpfile.mkdtemp()
            video_path = os.path.join(tmp_dir, pending_video["name"])
            with open(video_path, "wb") as vf:
                vf.write(pending_video["bytes"])

            try:
                from video_processor import process_video
                result = process_video(video_path, whisper_model="base", max_frames=5)
                transcript = result["transcript"]["text"]
                frames = result["frames"]

                # 프레임 + 음성 텍스트를 Gemma 4에 전달
                if frames:
                    content_blocks = []
                    for frame in frames:
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{frame['base64']}",
                        })
                    content_blocks.append({
                        "type": "text",
                        "text": f"동영상 음성 내용:\n{transcript}\n\n질문: {user_input}",
                    })

                    messages = [
                        SMsg(content="You are a helpful AI assistant that analyzes videos. Answer in Korean."),
                        HMsg(content=content_blocks),
                    ]
                    chunks = []
                    for chunk in ollama.stream(messages):
                        text = chunk.content if hasattr(chunk, "content") else str(chunk)
                        chunks.append(text)
                        chat_container.markdown("".join(chunks))
                else:
                    # 프레임 없으면 음성 텍스트만으로 답변
                    prompt = ChatPromptTemplate.from_template(
                        "다음은 동영상의 음성을 텍스트로 변환한 내용입니다:\n{transcript}\n\n질문: {input}"
                    )
                    chain = prompt | ollama | StrOutputParser()
                    chunks = []
                    for chunk in chain.stream({"transcript": transcript, "input": user_input}):
                        chunks.append(chunk)
                        chat_container.markdown("".join(chunks))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        elif uploaded_images and selected_model in MULTIMODAL_MODELS:
            # 멀티모달: 이미지 + 텍스트 함께 전송
            content_blocks = []
            for img in uploaded_images:
                img_b64 = base64.b64encode(img["bytes"]).decode()
                content_blocks.append({
                    "type": "image_url",
                    "image_url": f"data:image/{img['type']};base64,{img_b64}",
                })
            content_blocks.append({"type": "text", "text": user_input})

            messages = [
                SMsg(content="You are a helpful AI assistant. Answer in Korean."),
                HMsg(content=content_blocks),
            ]

            chunks = []
            for chunk in ollama.stream(messages):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                chunks.append(text)
                chat_container.markdown("".join(chunks))

        elif active_files or cached_files:
            # RAG 모드
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            answer = rag_chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
        else:
            # 일반 텍스트 채팅
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))

        # 답변 완료 후 모델명 뱃지 표시
        full_answer = "".join(chunks)
        badge = f"\n\n`{model_label}`"
        chat_container.markdown(full_answer + badge)
        add_history("ai", full_answer + badge)
