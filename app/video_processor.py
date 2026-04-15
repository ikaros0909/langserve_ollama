"""
동영상 처리 모듈
- ffmpeg로 음성 추출
- Whisper로 음성 → 텍스트 변환 (STT)
- ffmpeg로 주요 프레임 추출 → Gemma 4 이미지 분석
"""
import os
import base64
import tempfile
import subprocess
from typing import List, Dict, Optional


def extract_audio(video_path: str, output_path: str) -> bool:
    """동영상에서 음성을 WAV로 추출."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn",  # 비디오 제거
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # Whisper 최적 샘플레이트
                "-ac", "1",  # 모노
                "-y",  # 덮어쓰기
                output_path,
            ],
            capture_output=True,
            check=True,
        )
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.CalledProcessError:
        return False


def extract_keyframes(video_path: str, output_dir: str, max_frames: int = 5) -> List[str]:
    """동영상에서 주요 프레임을 추출. 균등 간격으로 max_frames장."""
    # 먼저 동영상 길이 확인
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        duration = 10.0  # 기본값

    # 균등 간격으로 프레임 추출
    interval = max(duration / (max_frames + 1), 0.5)
    frame_paths = []
    for i in range(max_frames):
        timestamp = interval * (i + 1)
        if timestamp >= duration:
            break
        output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-ss", str(timestamp),
                    "-vframes", "1",
                    "-q:v", "2",
                    "-y",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
            if os.path.exists(output_path):
                frame_paths.append(output_path)
        except subprocess.CalledProcessError:
            continue

    return frame_paths


def get_audio_duration(audio_path: str) -> float:
    """음성 파일의 길이(초)를 반환."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def split_audio(audio_path: str, output_dir: str, chunk_seconds: int = 600) -> List[str]:
    """긴 음성을 chunk_seconds(기본 10분) 단위로 분할."""
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return [audio_path]

    chunk_paths = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = os.path.join(output_dir, f"chunk_{idx:03d}.wav")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", audio_path,
                    "-ss", str(start),
                    "-t", str(chunk_seconds),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    chunk_path,
                ],
                capture_output=True,
                check=True,
            )
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 44:
                chunk_paths.append(chunk_path)
        except subprocess.CalledProcessError:
            pass
        start += chunk_seconds
        idx += 1

    return chunk_paths if chunk_paths else [audio_path]


def _transcribe_single(model, audio_path: str, language: Optional[str] = None) -> Dict:
    """단일 파일을 Whisper로 변환. language=None이면 자동 감지."""
    kwargs = {
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "verbose": False,
    }
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)
    return result


def transcribe_audio(audio_path: str, model_size: str = "base", language: Optional[str] = None) -> Dict:
    """
    Whisper로 음성을 텍스트로 변환.
    항상 5분 단위 청크 분할로 처리 (긴 음성 안정성 보장).
    """
    import whisper
    import sys

    duration = get_audio_duration(audio_path)
    lang_label = language or "자동감지"
    print(f"[STT] 시작 — 길이: {duration:.1f}초 ({duration/60:.1f}분), 모델: {model_size}, 언어: {lang_label}", flush=True)

    if duration <= 0:
        print("[STT] 오류: 음성 길이를 확인할 수 없습니다.", flush=True)
        return {"text": "", "segments": [], "language": ""}

    model = whisper.load_model(model_size)
    print(f"[STT] Whisper {model_size} 모델 로드 완료", flush=True)

    # 항상 5분 단위 청크 분할 (안정성)
    chunk_seconds = 300
    total_chunks = int(duration // chunk_seconds) + (1 if duration % chunk_seconds > 0 else 0)
    all_text = []
    all_segments = []

    with tempfile.TemporaryDirectory() as chunk_dir:
        for chunk_idx in range(total_chunks):
            # 원본 영상 기준 정확한 시작 시간 (고정 오프셋)
            chunk_start = chunk_idx * chunk_seconds
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:03d}.wav")

            # ffmpeg: -ss를 -i 앞에 배치 → 정확한 프레임 탐색
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-ss", str(chunk_start),
                        "-i", audio_path,
                        "-t", str(chunk_seconds),
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        "-y",
                        chunk_path,
                    ],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[STT] 청크 {chunk_idx+1}/{total_chunks} ffmpeg 실패: {e}", flush=True)
                continue

            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) <= 44:
                print(f"[STT] 청크 {chunk_idx+1}/{total_chunks} 빈 파일, 건너뜀", flush=True)
                continue

            chunk_duration = get_audio_duration(chunk_path)
            end_time = chunk_start + chunk_duration
            print(f"[STT] 청크 {chunk_idx+1}/{total_chunks} 처리 중: {chunk_start/60:.1f}분 ~ {end_time/60:.1f}분", flush=True)

            try:
                result = _transcribe_single(model, chunk_path, language)
                chunk_text = result["text"]
                chunk_segs = result["segments"]
                all_text.append(chunk_text)
                for seg in chunk_segs:
                    # 원본 기준 고정 오프셋으로 타임스탬프 계산
                    all_segments.append({
                        "start": round(seg["start"] + chunk_start, 2),
                        "end": round(seg["end"] + chunk_start, 2),
                        "text": seg["text"],
                    })
                print(f"[STT] 청크 {chunk_idx+1}/{total_chunks} 완료: {len(chunk_segs)}개 세그먼트, {len(chunk_text)}자", flush=True)
            except Exception as e:
                print(f"[STT] 청크 {chunk_idx+1}/{total_chunks} Whisper 실패: {e}", flush=True)

    full_text = " ".join(all_text)
    print(f"[STT] 전체 완료: {len(all_segments)}개 세그먼트, {len(full_text)}자", flush=True)

    return {
        "text": full_text,
        "segments": all_segments,
        "language": language or "auto",
    }


def frames_to_base64(frame_paths: List[str]) -> List[Dict]:
    """프레임 이미지를 base64로 변환."""
    results = []
    for path in frame_paths:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        results.append({
            "base64": img_b64,
            "filename": os.path.basename(path),
        })
    return results


def process_video(video_path: str, whisper_model: str = "base", max_frames: int = 5, language: Optional[str] = None) -> Dict:
    """
    동영상 처리 통합 함수.
    반환: {"transcript": {...}, "frames": [...]}
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = {}

        # 1) 음성 추출 + STT
        audio_path = os.path.join(tmp_dir, "audio.wav")
        if extract_audio(video_path, audio_path):
            result["transcript"] = transcribe_audio(audio_path, whisper_model, language)
        else:
            result["transcript"] = {"text": "(음성을 추출할 수 없습니다)", "segments": [], "language": ""}

        # 2) 주요 프레임 추출
        frame_paths = extract_keyframes(video_path, tmp_dir, max_frames)
        result["frames"] = frames_to_base64(frame_paths)

        return result
