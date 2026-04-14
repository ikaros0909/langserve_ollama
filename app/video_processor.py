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


def transcribe_audio(audio_path: str, model_size: str = "base") -> Dict:
    """Whisper로 음성을 텍스트로 변환."""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="ko")
    return {
        "text": result["text"],
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result["segments"]
        ],
        "language": result.get("language", "ko"),
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


def process_video(video_path: str, whisper_model: str = "base", max_frames: int = 5) -> Dict:
    """
    동영상 처리 통합 함수.
    반환: {"transcript": {...}, "frames": [...]}
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = {}

        # 1) 음성 추출 + STT
        audio_path = os.path.join(tmp_dir, "audio.wav")
        if extract_audio(video_path, audio_path):
            result["transcript"] = transcribe_audio(audio_path, whisper_model)
        else:
            result["transcript"] = {"text": "(음성을 추출할 수 없습니다)", "segments": [], "language": ""}

        # 2) 주요 프레임 추출
        frame_paths = extract_keyframes(video_path, tmp_dir, max_frames)
        result["frames"] = frames_to_base64(frame_paths)

        return result
