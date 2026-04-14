from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Union, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from chain import chain
from chat import chain as chat_chain
from translator import chain as EN_TO_KO_chain
from langchain_ollama import ChatOllama
from llm import llm as model
# from xionic import chain as xionic_chain
import api_keys
import video_processor


app = FastAPI()


# --- API 인증 미들웨어 ---
import logging
logger = logging.getLogger("api_auth")
logging.basicConfig(level=logging.INFO)


class APIAuthMiddleware(BaseHTTPMiddleware):
    """
    /api/ 경로에 대해 X-API-Key, X-Secret-Key 헤더 검증.
    다른 경로(playground, langserve 등)는 통과.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"

        # /api/ 경로가 아니면 통과
        if not path.startswith("/api/"):
            return await call_next(request)

        # CORS preflight (OPTIONS) 요청은 인증 없이 통과
        if method == "OPTIONS":
            return await call_next(request)

        # 헬스체크는 인증 없이 허용
        if path == "/api/health":
            return await call_next(request)

        # 키 관리 엔드포인트는 로컬에서만 접근 허용
        if path.startswith("/api/keys"):
            if client_host in ("127.0.0.1", "::1", "localhost"):
                return await call_next(request)
            logger.warning(f"[AUTH] 키 관리 접근 거부: client={client_host}, path={path}")
            return JSONResponse(
                status_code=403,
                content={"error": "키 관리는 로컬에서만 접근 가능합니다."},
            )

        api_key = (request.headers.get("X-API-Key") or "").strip()
        secret_key = (request.headers.get("X-Secret-Key") or "").strip()

        logger.info(f"[AUTH] {method} {path} | client={client_host} | X-API-Key={api_key[:12]}... | X-Secret-Key={'(있음)' if secret_key else '(없음)'}")

        if not api_key or not secret_key:
            logger.warning(f"[AUTH] 인증 헤더 누락: api_key={'있음' if api_key else '없음'}, secret_key={'있음' if secret_key else '없음'}")
            return JSONResponse(
                status_code=401,
                content={"error": "X-API-Key 와 X-Secret-Key 헤더가 필요합니다."},
            )

        if not api_keys.validate_key(api_key, secret_key):
            logger.warning(f"[AUTH] 키 검증 실패: api_key={api_key[:12]}...")
            return JSONResponse(
                status_code=403,
                content={"error": "유효하지 않거나 비활성화된 API 키입니다."},
            )

        if not api_keys.check_rate_limit(api_key):
            logger.warning(f"[AUTH] Rate limit 초과: api_key={api_key[:12]}...")
            return JSONResponse(
                status_code=429,
                content={"error": f"요청 한도 초과 (분당 {api_keys.RATE_LIMIT_PER_MINUTE}회)."},
            )

        return await call_next(request)


app.add_middleware(APIAuthMiddleware)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/chat/playground")


add_routes(app, chain, path="/prompt")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

add_routes(app, EN_TO_KO_chain, path="/translate")

add_routes(app, model, path="/llm")

# add_routes(
#     app,
#     xionic_chain.with_types(input_type=InputChat),
#     path="/xionic",
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True,
#     playground_type="chat",
# )


# ============================================================
# API 키 관리 엔드포인트 (Streamlit UI에서 호출)
# ============================================================

class KeyCreateRequest(BaseModel):
    name: str


@app.post("/api/keys/create")
async def create_api_key(req: KeyCreateRequest):
    if not req.name or not req.name.strip():
        raise HTTPException(400, "키 이름을 입력하세요.")
    api_key, secret_key = api_keys.generate_key(req.name.strip())
    return {"api_key": api_key, "secret_key": secret_key}


@app.get("/api/keys/list")
async def list_api_keys():
    return api_keys.list_keys()


@app.post("/api/keys/revoke/{api_key}")
async def revoke_api_key(api_key: str):
    if api_keys.revoke_key(api_key):
        return {"message": "키가 비활성화되었습니다."}
    raise HTTPException(404, "키를 찾을 수 없습니다.")


@app.delete("/api/keys/delete/{api_key}")
async def delete_api_key(api_key: str):
    if api_keys.delete_key(api_key):
        return {"message": "키가 삭제되었습니다."}
    raise HTTPException(404, "키를 찾을 수 없습니다.")


# ============================================================
# 사용 가능한 모델 정의
# ============================================================

AVAILABLE_MODELS = {
    "exaone3.5:32b": {
        "name": "EXAONE 3.5 32B",
        "description": "LG AI 한국어 특화 대형 모델",
        "multimodal": False,
    },
    "huihui_ai/kanana-nano-abliterated": {
        "name": "Kanana Nano",
        "description": "Kakao 한국어/영어 이중언어 모델",
        "multimodal": False,
    },
    "gemma4:26b": {
        "name": "Gemma 4 26B MoE",
        "description": "Google MoE 모델 (4B active, 256K context, 멀티모달)",
        "multimodal": True,
    },
}

DEFAULT_MODEL = "exaone3.5:32b"


# ============================================================
# 외부 API 엔드포인트 (인증 필요: X-API-Key, X-Secret-Key)
# ============================================================

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    images: Optional[List[str]] = None  # base64 인코딩 이미지 리스트


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    answer: str
    model: str
    usage: Optional[UsageInfo] = None


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """일반 채팅 API — 모델을 선택하여 질문하고 답변을 받습니다. 멀티모달 모델은 이미지도 처리 가능."""
    from langchain_core.messages import HumanMessage as HMsg, SystemMessage as SMsg

    # 모델 선택
    model_name = req.model or DEFAULT_MODEL
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            400,
            f"지원하지 않는 모델입니다: {model_name}. "
            f"사용 가능한 모델: {list(AVAILABLE_MODELS.keys())}",
        )

    # 이미지가 포함된 요청인데 멀티모달 미지원 모델인 경우
    if req.images and not AVAILABLE_MODELS[model_name]["multimodal"]:
        raise HTTPException(
            400,
            f"'{model_name}'은 이미지를 지원하지 않습니다. "
            f"멀티모달 모델: {[k for k, v in AVAILABLE_MODELS.items() if v['multimodal']]}",
        )

    system = req.system_prompt or "You are a helpful AI assistant. Answer in Korean."
    temp = req.temperature if req.temperature is not None else 0.5

    llm = ChatOllama(model=model_name, temperature=temp)

    if req.images:
        # 멀티모달: 이미지 + 텍스트를 HumanMessage content 블록으로 구성
        content_blocks = []
        for img_b64 in req.images:
            content_blocks.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_b64}",
            })
        content_blocks.append({"type": "text", "text": req.message})

        messages = [
            SMsg(content=system),
            HMsg(content=content_blocks),
        ]
        result = llm.invoke(messages)
    else:
        # 텍스트 전용
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
        ])
        chain_api = prompt | llm
        result = chain_api.invoke({"input": req.message})

    answer = result.content if hasattr(result, "content") else str(result)

    # 토큰 사용량 추출
    usage = None
    if hasattr(result, "response_metadata"):
        meta = result.response_metadata
        if "prompt_eval_count" in meta or "eval_count" in meta:
            input_tokens = meta.get("prompt_eval_count", 0)
            output_tokens = meta.get("eval_count", 0)
            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

    return ChatResponse(answer=answer, model=model_name, usage=usage)


# --- 모델 목록 ---
@app.get("/api/models")
async def list_models():
    """사용 가능한 모델 목록 반환."""
    return {
        "default": DEFAULT_MODEL,
        "models": {
            k: {**v, "id": k} for k, v in AVAILABLE_MODELS.items()
        },
    }


# --- 동영상 처리 API ---
import tempfile
import shutil
from fastapi import UploadFile, File, Form


@app.post("/api/video")
async def api_video(
    file: UploadFile = File(...),
    message: str = Form(default="이 동영상의 내용을 분석해줘"),
    model: str = Form(default="gemma4:26b"),
    whisper_model: str = Form(default="base"),
    max_frames: int = Form(default=5),
):
    """
    동영상 업로드 → 음성 텍스트 변환(Whisper) + 주요 프레임 이미지 분석(Gemma 4).
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"지원하지 않는 모델: {model}")

    # 임시 파일에 동영상 저장
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 동영상 처리: 음성 추출 + STT + 프레임 추출
        result = video_processor.process_video(video_path, whisper_model, max_frames)
        transcript = result["transcript"]["text"]
        frames = result["frames"]

        # Gemma 4로 프레임 분석 (멀티모달 모델인 경우)
        frame_analysis = ""
        if frames and AVAILABLE_MODELS.get(model, {}).get("multimodal"):
            from langchain_core.messages import HumanMessage as HMsg, SystemMessage as SMsg

            content_blocks = []
            for frame in frames:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{frame['base64']}",
                })
            content_blocks.append({
                "type": "text",
                "text": f"다음은 동영상에서 추출한 주요 장면입니다. 음성 내용: {transcript}\n\n질문: {message}",
            })

            llm = ChatOllama(model=model, temperature=0.5)
            llm_result = llm.invoke([
                SMsg(content="You are a helpful AI assistant that analyzes videos. Answer in Korean."),
                HMsg(content=content_blocks),
            ])
            frame_analysis = llm_result.content if hasattr(llm_result, "content") else str(llm_result)

        return {
            "transcript": transcript,
            "segments": result["transcript"]["segments"],
            "language": result["transcript"]["language"],
            "frame_count": len(frames),
            "analysis": frame_analysis,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- 자막/스크립트 생성 API ---
@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    whisper_model: str = Form(default="base"),
    format: str = Form(default="json"),
):
    """
    영상 또는 음성 파일 → 자막/스크립트 생성.

    - 지원 포맷: mp4, avi, mov, mkv, webm, mp3, wav, m4a, ogg, flac
    - whisper_model: tiny, base, small, medium, large (클수록 정확, 느림)
    - format: json (기본), srt, vtt, text
    """
    SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_VIDEO | SUPPORTED_AUDIO:
        raise HTTPException(
            400,
            f"지원하지 않는 파일 형식: {ext}. "
            f"지원: {sorted(SUPPORTED_VIDEO | SUPPORTED_AUDIO)}",
        )

    if format not in ("json", "srt", "vtt", "text"):
        raise HTTPException(400, "format은 json, srt, vtt, text 중 하나여야 합니다.")

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 영상이면 음성 추출, 음성이면 그대로 사용
        audio_path = file_path
        if ext in SUPPORTED_VIDEO:
            audio_path = os.path.join(tmp_dir, "audio.wav")
            if not video_processor.extract_audio(file_path, audio_path):
                raise HTTPException(500, "음성 추출에 실패했습니다.")

        # Whisper 변환
        result = video_processor.transcribe_audio(audio_path, whisper_model)

        if format == "text":
            return {"text": result["text"], "language": result["language"]}

        if format == "srt":
            srt_lines = []
            for i, seg in enumerate(result["segments"], 1):
                start = _format_timestamp_srt(seg["start"])
                end = _format_timestamp_srt(seg["end"])
                srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
            return {"srt": "\n".join(srt_lines), "language": result["language"]}

        if format == "vtt":
            vtt_lines = ["WEBVTT\n"]
            for seg in result["segments"]:
                start = _format_timestamp_vtt(seg["start"])
                end = _format_timestamp_vtt(seg["end"])
                vtt_lines.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
            return {"vtt": "\n".join(vtt_lines), "language": result["language"]}

        # json (기본)
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# --- 헬스체크 ---
@app.get("/api/health")
async def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL, "available_models": list(AVAILABLE_MODELS.keys())}


if __name__ == "__main__":
    import os
    import uvicorn

    # uvicorn.run(app, host="10.2.2.44", port=8000)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        limit_max_request_size=500 * 1024 * 1024,
        timeout_keep_alive=600,  # 10분
    )
