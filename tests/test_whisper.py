"""
Whisper 직접 테스트 — 서버를 거치지 않고 video_processor를 직접 호출.
사용법: python tests/test_whisper.py <영상파일>
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from video_processor import extract_audio, get_audio_duration, transcribe_audio


def main():
    if len(sys.argv) < 2:
        print("사용법: python tests/test_whisper.py <영상 또는 음성 파일>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"파일 없음: {input_file}")
        sys.exit(1)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(input_file)[1].lower()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1) 음성 추출
        if ext in video_exts:
            audio_path = os.path.join(tmp_dir, "audio.wav")
            print(f"[1] 음성 추출 중: {input_file} → {audio_path}")
            ok = extract_audio(input_file, audio_path)
            if not ok:
                print("[1] 음성 추출 실패!")
                sys.exit(1)
        else:
            audio_path = input_file

        # 2) 음성 길이 확인
        duration = get_audio_duration(audio_path)
        file_size = os.path.getsize(audio_path)
        print(f"[2] 음성 파일: {audio_path}")
        print(f"    길이: {duration:.1f}초 ({duration/60:.1f}분)")
        print(f"    크기: {file_size / 1024 / 1024:.1f}MB")

        # 3) Whisper 변환
        print(f"[3] Whisper 변환 시작 (base 모델)")
        result = transcribe_audio(audio_path, model_size="base")

        # 4) 결과
        segments = result["segments"]
        text = result["text"]
        if segments:
            last_seg = segments[-1]
            print(f"\n[결과]")
            print(f"  총 세그먼트: {len(segments)}개")
            print(f"  총 문자수: {len(text)}자")
            print(f"  마지막 세그먼트 끝: {last_seg['end']:.1f}초 ({last_seg['end']/60:.1f}분)")
            print(f"  음성 길이 대비: {last_seg['end']/duration*100:.1f}%")
            print(f"\n  처음 3개:")
            for s in segments[:3]:
                print(f"    [{s['start']:.1f}~{s['end']:.1f}] {s['text']}")
            print(f"\n  마지막 3개:")
            for s in segments[-3:]:
                print(f"    [{s['start']:.1f}~{s['end']:.1f}] {s['text']}")
        else:
            print("[결과] 세그먼트 없음!")
            print(f"  텍스트: {text[:200]}")


if __name__ == "__main__":
    main()
