"""
RAG 컬렉션 관리 모듈
- 컬렉션별로 분리된 FAISS 벡터 저장소
- PDF/문서 업로드 → 임베딩 → 컬렉션에 저장
- 컬렉션 지정하여 검색
"""
import os
import re
import json
import shutil
import tempfile
from hashlib import sha256
from typing import List, Dict, Optional

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS

import opendataloader_pdf

# 기본 경로
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rag_collections")
METADATA_FILE = os.path.join(BASE_DIR, "collections.json")


def _ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)


def _load_metadata() -> Dict:
    _ensure_dirs()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(data: Dict):
    _ensure_dirs()
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_embeddings():
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


def _extract_text_from_image(file_path: str) -> str:
    """Gemma 4로 이미지에서 텍스트/내용 추출. RAG용이므로 원본 그대로 사용 (전처리 없음)."""
    import base64
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        ext = os.path.splitext(file_path)[1].lstrip(".") or "png"
        img_b64 = base64.b64encode(raw_bytes).decode()

        llm = ChatOllama(model="gemma4:26b", temperature=0, timeout=600)
        print(f"[RAG] Gemma 4 텍스트 추출 요청 중: {os.path.basename(file_path)}", flush=True)

        result = llm.invoke([
            SystemMessage(content="이미지의 모든 텍스트와 내용을 정확하게 추출하여 한국어로 정리해주세요. 손글씨도 최대한 정확히 읽어주세요. 표가 있으면 표 형식을 유지하세요."),
            HumanMessage(content=[
                {"type": "image_url", "image_url": f"data:image/{ext};base64,{img_b64}"},
                {"type": "text", "text": "이 이미지의 모든 내용을 텍스트로 추출해주세요. 손글씨 포함."},
            ]),
        ])
        text = result.content if hasattr(result, "content") else str(result)
        print(f"[RAG] 이미지 텍스트 추출 완료: {os.path.basename(file_path)} → {len(text)}자", flush=True)
        return text
    except Exception as e:
        print(f"[RAG] 이미지 텍스트 추출 실패: {os.path.basename(file_path)} → {e}", flush=True)
        return ""


def _build_docs(file_path: str) -> List[Document]:
    """파일을 청크로 분리하여 Document 리스트 반환."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    ext = os.path.splitext(file_path)[1].lower()

    # 이미지: Gemma 4로 내용 추출 후 청크 분할
    if ext in IMAGE_EXTENSIONS:
        text = _extract_text_from_image(file_path)
        if not text:
            return []
        docs = text_splitter.create_documents(
            [text], metadatas=[{"source_type": "image", "source_file": os.path.basename(file_path)}]
        )
        return docs

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
            md_file = os.path.join(
                tmp_dir,
                os.path.splitext(os.path.basename(file_path))[0] + ".md",
            )
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

    return docs


def create_collection(name: str, description: str = "") -> Dict:
    """새 RAG 컬렉션 생성."""
    meta = _load_metadata()
    if name in meta:
        return {"error": f"컬렉션 '{name}'이 이미 존재합니다."}

    col_dir = os.path.join(BASE_DIR, name)
    os.makedirs(os.path.join(col_dir, "files"), exist_ok=True)
    os.makedirs(os.path.join(col_dir, "embeddings"), exist_ok=True)

    meta[name] = {
        "description": description,
        "files": [],
    }
    _save_metadata(meta)
    print(f"[RAG] 컬렉션 생성: {name} — {description}", flush=True)
    return {"name": name, "description": description}


def delete_collection(name: str) -> bool:
    """컬렉션 삭제."""
    meta = _load_metadata()
    if name not in meta:
        return False
    col_dir = os.path.join(BASE_DIR, name)
    if os.path.exists(col_dir):
        shutil.rmtree(col_dir)
    del meta[name]
    _save_metadata(meta)
    print(f"[RAG] 컬렉션 삭제: {name}", flush=True)
    return True


def list_collections() -> List[Dict]:
    """모든 컬렉션 목록."""
    meta = _load_metadata()
    return [
        {"name": k, "description": v["description"], "file_count": len(v["files"])}
        for k, v in meta.items()
    ]


def add_file_to_collection(name: str, file_path: str, filename: str) -> Dict:
    """파일을 컬렉션에 추가 (임베딩 후 FAISS에 저장). 같은 파일명이면 덮어쓰기."""
    meta = _load_metadata()
    if name not in meta:
        return {"error": f"컬렉션 '{name}'이 존재하지 않습니다."}

    col_dir = os.path.join(BASE_DIR, name)

    # 같은 파일명이 이미 있으면 파일만 교체 (인덱스 재구축 안 함, 중복 허용)
    if filename in meta[name]["files"]:
        print(f"[RAG] 기존 파일 교체: {filename} → 컬렉션 '{name}' (파일만 교체, 기존 청크 유지)", flush=True)
        # 기존 파일만 삭제 (FAISS 재구축 안 함)
        old_path = os.path.join(col_dir, "files", filename)
        if os.path.exists(old_path):
            os.remove(old_path)

    # 파일 복사
    dest_path = os.path.join(col_dir, "files", filename)
    shutil.copy2(file_path, dest_path)

    # 문서 청크 생성
    print(f"[RAG] 파일 임베딩 중: {filename} → 컬렉션 '{name}'", flush=True)
    docs = _build_docs(dest_path)

    # 빈 문서 제거
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        print(f"[RAG] 경고: {filename}에서 텍스트를 추출하지 못했습니다. 건너뜀.", flush=True)
        # 파일은 저장하되 임베딩은 건너뜀
        if filename not in meta[name]["files"]:
            meta[name]["files"].append(filename)
        _save_metadata(meta)
        return {"collection": name, "filename": filename, "chunks": 0, "warning": "텍스트 추출 실패"}

    # 파일명을 메타데이터에 추가
    for doc in docs:
        doc.metadata["source_file"] = filename
        doc.metadata["collection"] = name

    # 임베딩 + FAISS
    embeddings = _get_embeddings()
    faiss_path = os.path.join(col_dir, "faiss_index")

    if os.path.exists(faiss_path):
        # 기존 인덱스에 추가
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    vectorstore.save_local(faiss_path)

    # 메타데이터 업데이트
    if filename not in meta[name]["files"]:
        meta[name]["files"].append(filename)
    _save_metadata(meta)

    print(f"[RAG] 완료: {len(docs)}개 청크 추가 → '{name}'", flush=True)
    return {"collection": name, "filename": filename, "chunks": len(docs)}


def list_files_in_collection(name: str) -> List[str]:
    """컬렉션의 파일 목록."""
    meta = _load_metadata()
    if name not in meta:
        return []
    return meta[name]["files"]


def delete_file_from_collection(name: str, filename: str) -> bool:
    """컬렉션에서 파일 삭제 (FAISS 인덱스 재구축)."""
    meta = _load_metadata()
    if name not in meta or filename not in meta[name]["files"]:
        return False

    col_dir = os.path.join(BASE_DIR, name)

    # 파일 삭제
    file_path = os.path.join(col_dir, "files", filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    meta[name]["files"].remove(filename)
    _save_metadata(meta)

    # FAISS 인덱스 재구축 (남은 파일들로)
    _rebuild_collection_index(name)
    return True


def _rebuild_collection_index(name: str):
    """컬렉션의 FAISS 인덱스를 남은 파일들로 재구축."""
    col_dir = os.path.join(BASE_DIR, name)
    faiss_path = os.path.join(col_dir, "faiss_index")

    # 기존 인덱스 삭제
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)

    files_dir = os.path.join(col_dir, "files")
    all_docs = []
    for fname in os.listdir(files_dir):
        fpath = os.path.join(files_dir, fname)
        docs = _build_docs(fpath)
        for doc in docs:
            doc.metadata["source_file"] = fname
            doc.metadata["collection"] = name
        all_docs.extend(docs)

    if all_docs:
        embeddings = _get_embeddings()
        vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
        vectorstore.save_local(faiss_path)
    print(f"[RAG] 인덱스 재구축: '{name}' — {len(all_docs)}개 청크", flush=True)


def get_retriever(name: str, k: int = 8):
    """컬렉션의 retriever 반환."""
    col_dir = os.path.join(BASE_DIR, name)
    faiss_path = os.path.join(col_dir, "faiss_index")

    if not os.path.exists(faiss_path):
        return None

    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 30})


def get_all_documents(name: str) -> List[Document]:
    """컬렉션의 모든 문서를 반환 (전체 참조 모드)."""
    col_dir = os.path.join(BASE_DIR, name)
    faiss_path = os.path.join(col_dir, "faiss_index")

    if not os.path.exists(faiss_path):
        return []

    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    # FAISS docstore에서 전체 문서 추출
    all_docs = list(vectorstore.docstore._dict.values())
    print(f"[RAG] 컬렉션 '{name}' 전체 문서 로드: {len(all_docs)}개", flush=True)
    return all_docs
