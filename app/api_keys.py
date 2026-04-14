"""
API Key 관리 모듈
- SQLite 기반 키 저장소
- api_key(공개): 요청 시 헤더에 포함
- secret_key: 생성 시 한 번만 표시, DB에는 SHA-256 해시로 저장
- Rate limiting: 키별 분당 요청 수 제한
"""
import hashlib
import secrets
import sqlite3
import time
import os
from datetime import datetime
from typing import Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "api_keys.db")


def _get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            secret_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_request_log_key_ts
        ON request_log (api_key, timestamp)
    """)
    conn.commit()
    return conn


def _hash_secret(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode()).hexdigest()


def generate_key(name: str) -> Tuple[str, str]:
    """새 API 키 쌍 생성. (api_key, secret_key) 반환. secret_key는 이때만 볼 수 있음."""
    api_key = "jk-" + secrets.token_hex(16)
    secret_key = "sk-" + secrets.token_hex(32)
    secret_hash = _hash_secret(secret_key)

    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO api_keys (name, api_key, secret_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, api_key, secret_hash, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    return api_key, secret_key


def validate_key(api_key: str, secret_key: str) -> bool:
    """api_key + secret_key 조합이 유효하고 활성 상태인지 검증."""
    api_key = api_key.strip()
    secret_key = secret_key.strip()
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT secret_hash, is_active FROM api_keys WHERE api_key = ?",
            (api_key,),
        ).fetchone()
        if not row:
            return False
        secret_hash, is_active = row
        if not is_active:
            return False
        return secret_hash == _hash_secret(secret_key)
    finally:
        conn.close()


def list_keys():
    """모든 API 키 목록 반환 (secret_hash는 포함하지 않음)."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT name, api_key, created_at, is_active FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        return [
            {"name": r[0], "api_key": r[1], "created_at": r[2], "is_active": bool(r[3])}
            for r in rows
        ]
    finally:
        conn.close()


def revoke_key(api_key: str) -> bool:
    """키 비활성화."""
    conn = _get_db()
    try:
        cur = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE api_key = ?", (api_key,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_key(api_key: str) -> bool:
    """키 완전 삭제."""
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM api_keys WHERE api_key = ?", (api_key,))
        conn.execute("DELETE FROM request_log WHERE api_key = ?", (api_key,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# --- Rate Limiting ---
RATE_LIMIT_PER_MINUTE = 30


def check_rate_limit(api_key: str) -> bool:
    """분당 요청 수 체크. 초과 시 False 반환."""
    conn = _get_db()
    try:
        now = time.time()
        one_minute_ago = now - 60
        # 오래된 로그 정리
        conn.execute("DELETE FROM request_log WHERE timestamp < ?", (one_minute_ago,))
        count = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE api_key = ? AND timestamp > ?",
            (api_key, one_minute_ago),
        ).fetchone()[0]
        if count >= RATE_LIMIT_PER_MINUTE:
            conn.commit()
            return False
        conn.execute(
            "INSERT INTO request_log (api_key, timestamp) VALUES (?, ?)",
            (api_key, now),
        )
        conn.commit()
        return True
    finally:
        conn.close()
