"""
API 키 관리 + 인증 교차 테스트 & 시나리오 테스트
서버가 localhost:8000에서 실행 중이어야 합니다.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import requests

BASE = "http://localhost:8000"


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_result(name, passed, detail=""):
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def test_cross():
    """교차 테스트: 유효/무효 키로 API 호출"""
    header("교차 테스트: 유효/무효 키 검증")
    results = []

    # 1) 헤더 없이 호출 → 401
    r = requests.post(f"{BASE}/api/chat", json={"message": "hello"})
    results.append(test_result(
        "헤더 없이 호출 → 401",
        r.status_code == 401,
        f"status={r.status_code}"
    ))

    # 2) 잘못된 키로 호출 → 403
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "hello"},
        headers={"X-API-Key": "jk-fake", "X-Secret-Key": "sk-fake"},
    )
    results.append(test_result(
        "잘못된 키로 호출 → 403",
        r.status_code == 403,
        f"status={r.status_code}"
    ))

    # 3) 키 생성
    r = requests.post(f"{BASE}/api/keys/create", json={"name": "테스트용"})
    assert r.status_code == 200, f"키 생성 실패: {r.text}"
    data = r.json()
    api_key = data["api_key"]
    secret_key = data["secret_key"]
    results.append(test_result(
        "키 생성 성공",
        api_key.startswith("jk-") and secret_key.startswith("sk-"),
        f"api_key={api_key[:16]}..."
    ))

    # 4) 유효한 키로 호출 → 200
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "1+1은?"},
        headers={"X-API-Key": api_key, "X-Secret-Key": secret_key},
        timeout=120,
    )
    results.append(test_result(
        "유효한 키로 호출 → 200",
        r.status_code == 200,
        f"status={r.status_code}, answer={r.json().get('answer', '')[:50]}..." if r.status_code == 200 else f"status={r.status_code}"
    ))

    # 5) api_key는 맞지만 secret_key 틀림 → 403
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "hello"},
        headers={"X-API-Key": api_key, "X-Secret-Key": "sk-wrongkey"},
    )
    results.append(test_result(
        "api_key 맞고 secret_key 틀림 → 403",
        r.status_code == 403,
        f"status={r.status_code}"
    ))

    # 6) 키 목록에서 확인
    r = requests.get(f"{BASE}/api/keys/list")
    keys = r.json()
    found = any(k["api_key"] == api_key for k in keys)
    results.append(test_result(
        "키 목록에서 생성한 키 확인",
        found,
    ))

    # 정리: 테스트 키 삭제
    requests.delete(f"{BASE}/api/keys/delete/{api_key}")

    return results


def test_scenario():
    """시나리오 테스트: 생성 → 사용 → 비활성화 → 재호출 차단 → 삭제"""
    header("시나리오 테스트: 전체 라이프사이클")
    results = []

    # 1) 키 생성
    r = requests.post(f"{BASE}/api/keys/create", json={"name": "시나리오 테스트"})
    data = r.json()
    api_key, secret_key = data["api_key"], data["secret_key"]
    results.append(test_result("1. 키 생성", r.status_code == 200))

    # 2) 정상 호출
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "안녕"},
        headers={"X-API-Key": api_key, "X-Secret-Key": secret_key},
        timeout=120,
    )
    results.append(test_result(
        "2. 생성된 키로 API 호출 → 200",
        r.status_code == 200,
        f"status={r.status_code}"
    ))

    # 3) 키 비활성화
    r = requests.post(f"{BASE}/api/keys/revoke/{api_key}")
    results.append(test_result("3. 키 비활성화", r.status_code == 200))

    # 4) 비활성화 후 호출 → 403
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "안녕"},
        headers={"X-API-Key": api_key, "X-Secret-Key": secret_key},
    )
    results.append(test_result(
        "4. 비활성화 키로 호출 → 403",
        r.status_code == 403,
        f"status={r.status_code}"
    ))

    # 5) 키 삭제
    r = requests.delete(f"{BASE}/api/keys/delete/{api_key}")
    results.append(test_result("5. 키 삭제", r.status_code == 200))

    # 6) 삭제 후 호출 → 403
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "안녕"},
        headers={"X-API-Key": api_key, "X-Secret-Key": secret_key},
    )
    results.append(test_result(
        "6. 삭제된 키로 호출 → 403",
        r.status_code == 403,
        f"status={r.status_code}"
    ))

    # 7) 키 목록에서 삭제 확인
    r = requests.get(f"{BASE}/api/keys/list")
    keys = r.json()
    not_found = not any(k["api_key"] == api_key for k in keys)
    results.append(test_result("7. 키 목록에서 삭제 확인", not_found))

    return results


if __name__ == "__main__":
    all_results = []

    try:
        all_results.extend(test_cross())
    except Exception as e:
        print(f"\n  [ERROR] 교차 테스트 실패: {e}")

    try:
        all_results.extend(test_scenario())
    except Exception as e:
        print(f"\n  [ERROR] 시나리오 테스트 실패: {e}")

    # 최종 결과
    header("최종 결과")
    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    print(f"  {passed}/{total} 통과")
    if passed == total:
        print("  모든 테스트 통과!")
    else:
        print(f"  {total - passed}개 실패")
