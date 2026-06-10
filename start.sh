#!/bin/bash
# ============================================================
# 서버 + 웹 UI 동시 실행 스크립트
# 사용법: ./start.sh
# 종료: Ctrl+C
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_DIR/ollama-env/bin/activate"

echo "LangServe API 서버 시작 중..."
cd "$PROJECT_DIR/app"
python server.py &
SERVER_PID=$!
sleep 3

echo "Streamlit 웹 UI 시작 중..."
cd "$PROJECT_DIR/example"
streamlit run main.py &
UI_PID=$!

echo ""
echo "==============================="
echo "  서버 실행 완료"
echo "  API:  http://localhost:8000"
echo "  웹UI: http://localhost:8501"
echo "  종료: Ctrl+C"
echo "==============================="
echo ""

# Ctrl+C 시 둘 다 종료
trap "echo '종료 중...'; kill $SERVER_PID $UI_PID 2>/dev/null; exit" INT TERM
wait
