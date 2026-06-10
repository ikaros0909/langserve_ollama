# JInhak Local LLM 서버 설치 매뉴얼

로컬 환경에서 LLM 모델을 실행하고, 웹 채팅 UI + REST API를 제공하는 서버입니다.

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [프로젝트 클론](#2-프로젝트-클론)
3. [Python 환경 설정](#3-python-환경-설정)
4. [Ollama 설치 및 모델 다운로드](#4-ollama-설치-및-모델-다운로드)
5. [환경 변수 설정](#5-환경-변수-설정)
6. [서버 실행](#6-서버-실행)
7. [웹 채팅 UI 실행](#7-웹-채팅-ui-실행)
8. [API 사용법](#8-api-사용법)
9. [GPU 모니터링](#9-gpu-모니터링)
10. [문제 해결](#10-문제-해결)

---

## 1. 사전 준비

### 필수 사양
| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Ubuntu 22.04 | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 3080 (10GB VRAM) | RTX 3090 (24GB VRAM) |
| RAM | 32GB | 64GB |
| 디스크 | 100GB 여유 | 200GB+ 여유 |
| Python | 3.9+ | 3.10 |

### 필수 소프트웨어 설치

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Java 11 이상 (PDF 파싱용)
sudo apt-get install -y default-jre
java -version

# ffmpeg (동영상/음성 처리용)
sudo apt-get install -y ffmpeg
ffmpeg -version

# Git
sudo apt-get install -y git
```

---

## 2. 프로젝트 클론

```bash
cd ~
mkdir -p chatbot && cd chatbot
git clone git@github.com:ikaros0909/langserve_ollama.git
cd langserve_ollama
```

---

## 3. Python 환경 설정

### 3-1. 가상환경 생성

```bash
python3 -m venv ollama-env
source ollama-env/bin/activate
```

### 3-2. 패키지 설치

```bash
pip install --upgrade pip

# 핵심 패키지
pip install langchain langchain-community langchain-core langchain-ollama langchain-openai langchain-text-splitters
pip install langserve fastapi uvicorn
pip install faiss-cpu
pip install streamlit
pip install opendataloader-pdf
pip install python-dotenv

# 임베딩 모델
pip install sentence-transformers

# 멀티미디어 처리
pip install openai-whisper ffmpeg-python
pip install opencv-python-headless

# 호환성 맞추기 (중요!)
pip install "numpy<2.0"
pip install "transformers<4.52"
```

> numpy 2.x는 transformers/torch와 충돌하므로 반드시 `<2.0`으로 설치

### 3-3. 설치 확인

```bash
python -c "
import langchain, langserve, faiss, streamlit, whisper, cv2
print('langchain:', langchain.__version__)
print('langserve:', langserve.__version__)
print('모든 패키지 정상!')
"
```

---

## 4. Ollama 설치 및 모델 다운로드

### 4-1. Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl restart ollama
ollama --version
```

### 4-2. Ollama 최적화 설정

GPU 메모리를 효율적으로 사용하기 위해 설정합니다:

```bash
sudo systemctl edit ollama
```

편집기에 아래 내용 입력 후 저장:

```
[Service]
Environment="OLLAMA_KEEP_ALIVE=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

적용:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

> `OLLAMA_KEEP_ALIVE=0`: 응답 후 즉시 모델 언로드 (VRAM 절약)
> `OLLAMA_MAX_LOADED_MODELS=1`: 동시에 1개 모델만 로드

### 4-3. 기본 모델 다운로드

**필수 (기본 모델)**
```bash
ollama pull exaone3.5:32b          # LG AI 한국어 32B (19GB)
```

**선택 (추가 모델)**
```bash
ollama pull gemma4:26b             # Google MoE 멀티모달 (17GB)
ollama pull gemma4:31b             # Google Dense 멀티모달 (19GB)
ollama pull huihui_ai/kanana-nano-abliterated  # Kakao 경량 모델 (2.2GB)
ollama pull 0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M  # 무검열 (16GB)
```

### 4-4. EXAONE 4.5 33B 설치 (HuggingFace GGUF)

EXAONE 4.5는 Ollama 공식 라이브러리에 없어서 수동 설치가 필요합니다:

```bash
# 1. GGUF 파일 다운로드 (약 22GB)
pip install huggingface-hub
huggingface-cli download LGAI-EXAONE/EXAONE-4.5-33B-GGUF \
  EXAONE-4.5-33B-Q4_K_M.gguf \
  mmproj-EXAONE-4.5-33B-BF16.gguf \
  --local-dir ~/chatbot/langserve_ollama/gguf

# 2. Ollama에 등록
ollama create EXAONE-4.5-33B -f ~/chatbot/langserve_ollama/ollama-modelfile/EXAONE-4.5-33B/Modelfile

# 3. 확인
ollama list
```

### 4-5. 모델 목록 확인

```bash
ollama list
```

설치 완료 시 예시:
```
NAME                                                SIZE
exaone3.5:32b                                       19 GB
EXAONE-4.5-33B:latest                               22 GB
gemma4:26b                                          17 GB
gemma4:31b                                          19 GB
huihui_ai/kanana-nano-abliterated:latest            2.2 GB
0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M   16 GB
```

---

## 5. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```bash
cd ~/chatbot/langserve_ollama
nano .env
```

내용:
```
HF_TOKEN=hf_여기에_허깅페이스_토큰
```

> HuggingFace 토큰은 https://huggingface.co/settings/tokens 에서 발급
> 화자 분리(pyannote) 기능 사용 시 필요합니다

### pyannote 라이선스 동의 (화자 분리 사용 시)

아래 두 페이지에서 라이선스에 동의해야 합니다:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

---

## 6. 서버 실행

### 6-1. LangServe API 서버

```bash
cd ~/chatbot/langserve_ollama/app
source ../ollama-env/bin/activate
python server.py
```

정상 실행 시:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
LANGSERVE: Playground for chain "/llm/" is live at /llm/playground/
LANGSERVE: Playground for chain "/chat/" is live at /chat/playground/
```

> 서버는 `http://서버IP:8000` 으로 접근 가능

### 6-2. 서버 종료 / 재시작

```bash
# 종료
kill $(lsof -t -i :8000)

# 재시작
cd ~/chatbot/langserve_ollama/app
python server.py
```

---

## 7. 웹 채팅 UI 실행

**별도 터미널**에서:

```bash
cd ~/chatbot/langserve_ollama/example
source ../ollama-env/bin/activate
streamlit run main.py
```

정상 실행 시:
```
Local URL: http://localhost:8501
Network URL: http://서버IP:8501
```

브라우저에서 `http://서버IP:8501` 접속

### UI 기능
- 사이드바에서 **모델 선택** (EXAONE, Gemma, Kanana 등)
- **RAG 파일 업로드** (PDF, TXT, DOCX → 문서 기반 질의응답)
- **멀티모달 모델** 선택 시 이미지 첨부 가능
- **API 키 관리** (사이드바 하단)

---

## 8. API 사용법

### 8-1. API 키 발급

웹 UI 사이드바 하단의 **API 키 관리** 에서 생성하거나:

```bash
curl -X POST http://localhost:8000/api/keys/create \
  -H "Content-Type: application/json" \
  -d '{"name": "내 앱"}'
```

> Secret Key는 생성 시 한 번만 표시됩니다!

### 8-2. 채팅 API

```bash
curl -X POST http://서버IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: jk-발급받은키" \
  -H "X-Secret-Key: sk-발급받은시크릿키" \
  -d '{
    "message": "안녕하세요",
    "model": "exaone3.5:32b",
    "system_prompt": "당신은 친절한 AI 상담원입니다.",
    "temperature": 0.5
  }'
```

### 8-3. RAG (문서 기반 질의응답)

```bash
# 1. 문서 업로드
curl -X POST http://서버IP:8000/api/rag/upload \
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \
  -F "files=@모집요강.pdf" \
  -F "collection=모집요강"

# 2. RAG 참조하여 질문
curl -X POST http://서버IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \
  -d '{
    "message": "수시 추천형 지원 자격은?",
    "rag_collection": "모집요강"
  }'
```

### 8-4. 이미지 분석 (멀티모달)

```bash
curl -X POST http://서버IP:8000/api/chat-upload \
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \
  -F "message=이 이미지를 설명해줘" \
  -F "model=gemma4:26b" \
  -F "images=@사진.jpg"
```

### 8-5. 영상/음성 자막 생성

```bash
curl -X POST http://서버IP:8000/api/transcribe \
  -H "X-API-Key: jk-..." -H "X-Secret-Key: sk-..." \
  -F "file=@강의.mp4" \
  -F "format=srt" \
  -F "language=ko" \
  --max-time 1800 -o subtitle.srt
```

### 8-6. 헬스체크

```bash
curl http://서버IP:8000/api/health
```

### 8-7. 전체 API 목록

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/chat` | POST | 텍스트 채팅 (JSON) |
| `/api/chat-upload` | POST | 이미지+텍스트 채팅 (multipart) |
| `/api/models` | GET | 사용 가능한 모델 목록 |
| `/api/health` | GET | 서버 상태 |
| `/api/rag/collections` | GET/POST | RAG 컬렉션 관리 |
| `/api/rag/upload` | POST | RAG 문서 업로드 |
| `/api/rag/collections/{name}/files` | GET | 컬렉션 파일 목록 |
| `/api/transcribe` | POST | 자막/스크립트 생성 |
| `/api/transcribe/diarize` | POST | 화자 분리 자막 |
| `/api/video` | POST | 동영상 분석 (Whisper+Gemma) |
| `/api/keys/create` | POST | API 키 생성 |
| `/api/keys/list` | GET | API 키 목록 |

---

## 9. GPU 모니터링

```bash
# 실시간 GPU 상태
watch -d -n 1 nvidia-smi
```

---

## 10. 문제 해결

### Ollama 서버가 안 뜰 때
```bash
sudo systemctl restart ollama
sudo systemctl status ollama
```

### 모델 로딩 실패 (GPU 메모리 부족)
```bash
# 실행 중인 모델 모두 종료
ollama stop exaone3.5:32b
ollama stop gemma4:26b
# 이후 다시 시도
```

### CUDA 드라이버 오류
```bash
# 드라이버 버전 확인
nvidia-smi
# "Driver/library version mismatch" 나오면 재부팅
sudo reboot
```

### LangServe 서버 포트 충돌
```bash
# 8000번 포트 사용 중인 프로세스 종료
kill $(lsof -t -i :8000)
```

### Streamlit 업로드 크기 제한
`~/.streamlit/config.toml` 파일:
```toml
[server]
maxUploadSize = 500
maxMessageSize = 500
```

---

## 프로젝트 구조

```
langserve_ollama/
├── app/                          # 서버 코드
│   ├── server.py                 # FastAPI + LangServe 서버
│   ├── api_keys.py               # API 키 관리 (SQLite)
│   ├── rag_collections.py        # RAG 컬렉션 관리
│   ├── video_processor.py        # 동영상/음성 처리 (Whisper)
│   ├── image_preprocess.py       # 이미지 전처리 (OpenCV)
│   ├── llm.py                    # 기본 LLM 설정
│   ├── chain.py                  # LangChain 체인
│   └── chat.py                   # 채팅 체인
├── example/
│   └── main.py                   # Streamlit 웹 UI
├── ollama-modelfile/
│   └── EXAONE-4.5-33B/Modelfile  # EXAONE 4.5 Ollama 등록용
├── gguf/                         # GGUF 모델 파일 (gitignore)
├── data/                         # DB, RAG 저장소 (gitignore)
├── tests/                        # 테스트 스크립트
├── .env                          # 환경 변수 (gitignore)
└── .gitignore
```

---

## 빠른 시작 요약

```bash
# 1. 클론
git clone git@github.com:ikaros0909/langserve_ollama.git
cd langserve_ollama

# 2. Python 환경
python3 -m venv ollama-env && source ollama-env/bin/activate
pip install langchain langchain-ollama langserve fastapi uvicorn faiss-cpu streamlit opendataloader-pdf python-dotenv sentence-transformers openai-whisper opencv-python-headless "numpy<2.0" "transformers<4.52"

# 3. Ollama + 모델
curl -fsSL https://ollama.com/install.sh | sh
ollama pull exaone3.5:32b

# 4. 환경 변수
echo "HF_TOKEN=hf_your_token" > .env

# 5. 서버 실행 (터미널 1)
cd app && python server.py

# 6. 웹 UI 실행 (터미널 2)
cd example && streamlit run main.py

# 7. 브라우저에서 http://localhost:8501 접속!
```

---

## License

```
MIT License
Copyright (c) 2024, 테디노트
```
