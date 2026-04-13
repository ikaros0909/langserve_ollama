from openai import AsyncOpenAI
import os  # os 모듈 임포트
api_key = os.getenv("OPENAI_API_KEY")
aclient = AsyncOpenAI(api_key=api_key)
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt


# 재시도 로직 추가
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def openai_request(payload):
    response = await aclient.chat.completions.create(model=payload["model"],
    messages=payload["messages"])
    return response

async def main():
    # 환경 변수에서 API 키 가져오기
    
    if not api_key:
        raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

      # API 키 설정

    payload = {
        "model": "gpt-3.5-turbo",  # 최신 모델 사용
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
    }
    response = await openai_request(payload)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
