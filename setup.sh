#!/bin/bash
set -e

# ============================================================
# JInhak Local LLM 서버 자동 설치 스크립트
# 사용법: chmod +x setup.sh && ./setup.sh
# ============================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/ollama-env"
GGUF_DIR="$PROJECT_DIR/gguf"

step() { echo -e "\n${GREEN}[$(date +%H:%M:%S)] ========== $1 ==========${NC}\n"; }
warn() { echo -e "${YELLOW}[경고] $1${NC}"; }
fail() { echo -e "${RED}[실패] $1${NC}"; exit 1; }

# ============================================================
# 1. 시스템 패키지
# ============================================================
step "1/8. 시스템 패키지 설치"

if ! command -v java &>/dev/null; then
    echo "Java 설치 중..."
    sudo apt-get update -qq
    sudo apt-get install -y default-jre
else
    echo "Java 이미 설치됨: $(java -version 2>&1 | head -1)"
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg 설치 중..."
    sudo apt-get install -y ffmpeg
else
    echo "ffmpeg 이미 설치됨: $(ffmpeg -version 2>&1 | head -1)"
fi

if ! command -v nvidia-smi &>/dev/null; then
    warn "NVIDIA 드라이버가 감지되지 않습니다. GPU 없이 CPU 모드로 실행됩니다."
else
    echo "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '확인 불가')"
fi

# ============================================================
# 2. Python 가상환경
# ============================================================
step "2/8. Python 가상환경 설정"

if [ ! -d "$VENV_DIR" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv "$VENV_DIR"
else
    echo "가상환경 이미 존재: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ============================================================
# 3. Python 패키지 설치
# ============================================================
step "3/8. Python 패키지 설치 (2~3분 소요)"

pip install -q \
    langchain langchain-community langchain-core langchain-ollama \
    langchain-openai langchain-text-splitters \
    langserve fastapi uvicorn \
    faiss-cpu \
    streamlit \
    opendataloader-pdf \
    python-dotenv \
    sentence-transformers \
    openai-whisper ffmpeg-python \
    opencv-python-headless \
    pyannote.audio \
    requests \
    "numpy<2.0" \
    "transformers<4.52"

echo "패키지 설치 완료"

# 설치 검증
python -c "
import langchain, langserve, faiss, streamlit
print(f'  langchain: {langchain.__version__}')
print(f'  langserve: {langserve.__version__}')
print('  핵심 패키지 검증 완료')
" || fail "Python 패키지 검증 실패"

# ============================================================
# 4. Ollama 설치
# ============================================================
step "4/8. Ollama 설치"

if ! command -v ollama &>/dev/null; then
    echo "Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama 이미 설치됨: $(ollama --version 2>&1 | head -1)"
fi

# Ollama 서비스 시작
sudo systemctl enable ollama 2>/dev/null || true
sudo systemctl start ollama 2>/dev/null || true
sleep 2

# ============================================================
# 5. Ollama 최적화 설정
# ============================================================
step "5/8. Ollama GPU 메모리 최적화 설정"

OLLAMA_OVERRIDE="/etc/systemd/system/ollama.service.d/override.conf"
if [ ! -f "$OLLAMA_OVERRIDE" ]; then
    echo "Ollama 메모리 최적화 설정 적용 중..."
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee "$OLLAMA_OVERRIDE" > /dev/null <<'CONF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
CONF
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    sleep 2
    echo "설정 완료 (KEEP_ALIVE=0, MAX_LOADED=1)"
else
    echo "Ollama 최적화 이미 설정됨"
fi

# ============================================================
# 6. 모델 다운로드
# ============================================================
step "6/8. Ollama 모델 다운로드"

download_model() {
    local model=$1
    local desc=$2
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo "  [건너뜀] $model — 이미 다운로드됨"
    else
        echo "  [다운로드] $model ($desc)..."
        ollama pull "$model" || warn "$model 다운로드 실패 — 나중에 수동 다운로드 가능"
    fi
}

echo "필수 모델:"
download_model "exaone3.5:32b" "LG AI 한국어 32B, 19GB"

echo ""
echo "선택 모델 (전부 다운로드합니다. Ctrl+C로 중단 가능):"
download_model "gemma4:26b" "Google MoE 멀티모달, 17GB"
download_model "gemma4:31b" "Google Dense 멀티모달, 19GB"
download_model "huihui_ai/kanana-nano-abliterated" "Kakao 경량 2.2GB"
download_model "0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M" "무검열 16GB"

# ============================================================
# 7. EXAONE 4.5 설치 (HuggingFace GGUF)
# ============================================================
step "7/8. EXAONE 4.5 33B 설치 (HuggingFace)"

if ollama list 2>/dev/null | grep -q "EXAONE-4.5-33B"; then
    echo "EXAONE 4.5 이미 등록됨"
else
    mkdir -p "$GGUF_DIR"
    GGUF_FILE="$GGUF_DIR/EXAONE-4.5-33B-Q4_K_M.gguf"
    MMPROJ_FILE="$GGUF_DIR/mmproj-EXAONE-4.5-33B-BF16.gguf"

    if [ ! -f "$GGUF_FILE" ] || [ ! -f "$MMPROJ_FILE" ]; then
        echo "EXAONE 4.5 GGUF 다운로드 중 (약 22GB)..."
        pip install -q huggingface-hub
        huggingface-cli download LGAI-EXAONE/EXAONE-4.5-33B-GGUF \
            EXAONE-4.5-33B-Q4_K_M.gguf \
            mmproj-EXAONE-4.5-33B-BF16.gguf \
            --local-dir "$GGUF_DIR" || warn "EXAONE 4.5 다운로드 실패"
    fi

    if [ -f "$GGUF_FILE" ]; then
        echo "Ollama에 EXAONE 4.5 등록 중..."
        MODELFILE="$PROJECT_DIR/ollama-modelfile/EXAONE-4.5-33B/Modelfile"
        if [ -f "$MODELFILE" ]; then
            ollama create EXAONE-4.5-33B -f "$MODELFILE" || warn "EXAONE 4.5 등록 실패"
        else
            warn "Modelfile 없음: $MODELFILE"
        fi
    fi
fi

# ============================================================
# 8. 환경 변수 설정
# ============================================================
step "8/8. 환경 변수 설정"

ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "HF_TOKEN=" > "$ENV_FILE"
    warn ".env 파일이 생성되었습니다. HuggingFace 토큰을 입력해 주세요:"
    warn "  nano $ENV_FILE"
    warn "  HF_TOKEN=hf_여기에토큰입력"
else
    echo ".env 파일 이미 존재"
fi

# Streamlit 설정
STREAMLIT_CONFIG="$HOME/.streamlit/config.toml"
if [ ! -f "$STREAMLIT_CONFIG" ]; then
    mkdir -p "$HOME/.streamlit"
    cat > "$STREAMLIT_CONFIG" <<'TOML'
[server]
maxUploadSize = 500
maxMessageSize = 500
TOML
    echo "Streamlit 설정 완료 (업로드 500MB)"
else
    echo "Streamlit 설정 이미 존재"
fi

# ============================================================
# 완료
# ============================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  설치 완료!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "설치된 모델:"
ollama list 2>/dev/null || echo "  (Ollama 서비스 확인 필요)"
echo ""
echo "다음 단계:"
echo ""
echo "  1. 서버 실행 (터미널 1):"
echo "     cd $PROJECT_DIR/app"
echo "     source $VENV_DIR/bin/activate"
echo "     python server.py"
echo ""
echo "  2. 웹 UI 실행 (터미널 2):"
echo "     cd $PROJECT_DIR/example"
echo "     source $VENV_DIR/bin/activate"
echo "     streamlit run main.py"
echo ""
echo "  3. 브라우저에서 http://localhost:8501 접속"
echo ""
if [ ! -s "$ENV_FILE" ] || grep -q "HF_TOKEN=$" "$ENV_FILE"; then
    echo -e "${YELLOW}  [!] .env에 HF_TOKEN을 설정하면 화자 분리 기능을 사용할 수 있습니다.${NC}"
fi
