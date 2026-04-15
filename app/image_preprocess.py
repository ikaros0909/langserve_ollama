"""
손글씨 이미지 전처리 모듈
- 그레이스케일 변환
- 대비 강화 (CLAHE)
- 노이즈 제거
- 적응형 이진화
- 기울기 보정 (Deskew)
"""
import cv2
import numpy as np
from typing import Optional


def preprocess_handwriting(image_bytes: bytes) -> bytes:
    """손글씨 이미지를 OCR/멀티모달 LLM에 최적화된 형태로 전처리."""
    # bytes → numpy array
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    # 1) 그레이스케일
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) 대비 강화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3) 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4) 기울기 보정
    deskewed = _deskew(denoised)

    # 5) 적응형 이진화 (글씨를 더 선명하게)
    binary = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    # 6) 해상도 업스케일 (작은 이미지인 경우)
    h, w = binary.shape[:2]
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # numpy array → bytes (PNG)
    _, buf = cv2.imencode(".png", binary)
    return buf.tobytes()


def _deskew(img: np.ndarray) -> np.ndarray:
    """이미지 기울기 보정."""
    # 텍스트 영역의 각도 추정
    coords = np.column_stack(np.where(img < 128))
    if len(coords) < 100:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 미세 기울기만 보정 (±15도 이내)
    if abs(angle) > 15 or abs(angle) < 0.5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
