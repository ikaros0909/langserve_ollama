from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Union, Optional
from pydantic import BaseModel as PydanticBaseModel
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from chain import chain
from chat import chain as chat_chain
from translator import chain as EN_TO_KO_chain
from llm import llm as model
# from xionic import chain as xionic_chain
import api_keys


app = FastAPI()


# --- API 인증 미들웨어 ---
class APIAuthMiddleware(BaseHTTPMiddleware):
    """
    /api/ 경로에 대해 X-API-Key, X-Secret-Key 헤더 검증.
    다른 경로(playground, langserve 등)는 통과.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # /api/ 경로만 인증 필요
        if not path.startswith("/api/"):
            return await call_next(request)

        # 키 관리 엔드포인트는 로컬에서만 접근 허용
        if path.startswith("/api/keys"):
            client_host = request.client.host if request.client else ""
            if client_host in ("127.0.0.1", "::1", "localhost"):
                return await call_next(request)
            return JSONResponse(
                status_code=403,
                content={"error": "키 관리는 로컬에서만 접근 가능합니다."},
            )

        api_key = request.headers.get("X-API-Key")
        secret_key = request.headers.get("X-Secret-Key")

        if not api_key or not secret_key:
            return JSONResponse(
                status_code=401,
                content={"error": "X-API-Key 와 X-Secret-Key 헤더가 필요합니다."},
            )

        if not api_keys.validate_key(api_key, secret_key):
            return JSONResponse(
                status_code=403,
                content={"error": "유효하지 않거나 비활성화된 API 키입니다."},
            )

        if not api_keys.check_rate_limit(api_key):
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

class KeyCreateRequest(PydanticBaseModel):
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
# 외부 API 엔드포인트 (인증 필요: X-API-Key, X-Secret-Key)
# ============================================================

class ChatRequest(PydanticBaseModel):
    message: str
    system_prompt: Optional[str] = None


class ChatResponse(PydanticBaseModel):
    answer: str


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """일반 채팅 API — 모델에 질문하고 답변을 받습니다."""
    system = req.system_prompt or "You are a helpful AI assistant. Answer in Korean."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
    ])
    chain_api = prompt | model | StrOutputParser()
    answer = chain_api.invoke({"input": req.message})
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="10.2.2.44", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
