"""
Yarışma çıktı şeması doğrulama katmanı.

Her kareden üretilen JSON sözlüğünün TEKNOFEST 2026 Gökbörü standartlarına
uygunluğunu sıkı tip denetimiyle kontrol eder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

REQUIRED_FRAME_KEYS = (
    "frame_id",
    "timestamp",
    "govev_1_objects",
    "govev_2_position",
    "govev_3_match",
)

REQUIRED_OBJECT_KEYS = (
    "class_name",
    "bbox",
    "is_moving",
    "landing_status",
)

REQUIRED_POSITION_KEYS = ("x", "y", "z")

REQUIRED_MATCH_KEYS = (
    "top_left_x",
    "top_left_y",
    "width",
    "height",
    "score",
)

VALID_CLASS_NAMES = frozenset({"Tasit", "Insan", "UAP", "UAI"})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    for component in bbox:
        if not _is_int(component):
            return False
        if component < 0:
            return False
    _, _, width, height = bbox
    return width > 0 and height > 0


def _validate_govev_1_object(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    for key in REQUIRED_OBJECT_KEYS:
        if key not in obj:
            return False

    class_name = obj["class_name"]
    if not isinstance(class_name, str) or not class_name.strip():
        return False

    if not _validate_bbox(obj["bbox"]):
        return False

    is_moving = obj["is_moving"]
    if not _is_int(is_moving) or is_moving not in (0, 1):
        return False

    landing_status = obj["landing_status"]
    if not _is_int(landing_status) or landing_status not in (0, 1):
        return False

    return True


def _validate_govev_1_objects(objects: Any) -> bool:
    if not isinstance(objects, list):
        return False
    for obj in objects:
        if not _validate_govev_1_object(obj):
            return False
    return True


def _validate_govev_2_position(position: Any) -> bool:
    if not isinstance(position, dict):
        return False

    for key in REQUIRED_POSITION_KEYS:
        if key not in position:
            return False
        if not _is_number(position[key]):
            return False

    return True


def _validate_govev_3_match(match: Any) -> bool:
    if not isinstance(match, dict):
        return False

    for key in REQUIRED_MATCH_KEYS:
        if key not in match:
            return False

    score = match["score"]
    if not _is_number(score):
        return False
    if score < 0.0 or score > 1.0:
        return False

    for coord_key in ("top_left_x", "top_left_y", "width", "height"):
        value = match[coord_key]
        if not _is_int(value):
            return False
        if value < 0:
            return False

    width = match["width"]
    height = match["height"]
    if score > 0.0 and (width <= 0 or height <= 0):
        return False

    if "found" in match and not isinstance(match["found"], bool):
        return False

    return True


def validate_frame_output(frame_data: Any) -> bool:
    """
    Tek kare çıktı sözlüğünü doğrular.

    Args:
        frame_data: Kare işlem sonucu sözlüğü.

    Returns:
        Şema geçerliyse True.
    """
    try:
        if not isinstance(frame_data, dict):
            return False

        for key in REQUIRED_FRAME_KEYS:
            if key not in frame_data:
                return False

        frame_id = frame_data["frame_id"]
        if not _is_int(frame_id) or frame_id < 0:
            return False

        timestamp = frame_data["timestamp"]
        if not _is_number(timestamp) or timestamp < 0.0:
            return False

        if not _validate_govev_1_objects(frame_data["govev_1_objects"]):
            return False

        if not _validate_govev_2_position(frame_data["govev_2_position"]):
            return False

        if not _validate_govev_3_match(frame_data["govev_3_match"]):
            return False

        return True

    except Exception:
        return False


def validate_session_output(session_data: Any) -> Tuple[bool, Optional[str]]:
    """
    Tam oturum JSON çıktısını doğrular.

    Args:
        session_data: session_metadata + frames içeren oturum sözlüğü.

    Returns:
        (geçerli_mi, hata_mesajı) tuple.
    """
    if not isinstance(session_data, dict):
        return False, "Oturum verisi sözlük olmalıdır."

    if "frames" not in session_data:
        return False, "'frames' anahtarı eksik."

    frames = session_data["frames"]
    if not isinstance(frames, list):
        return False, "'frames' liste olmalıdır."

    for idx, frame in enumerate(frames):
        if not validate_frame_output(frame):
            return False, f"Kare {idx} şema doğrulamasından geçemedi."

    if "session_metadata" in session_data:
        metadata = session_data["session_metadata"]
        if not isinstance(metadata, dict):
            return False, "'session_metadata' sözlük olmalıdır."

    return True, None


def create_empty_frame_output(frame_id: int, timestamp: float) -> Dict[str, Any]:
    """
    Yarışma standartlarında boş kare çıktısı üretir.

    Args:
        frame_id: Kare indeksi.
        timestamp: Saniye cinsinden zaman damgası.

    Returns:
        Doğrulanabilir boş çıktı sözlüğü.
    """
    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "govev_1_objects": [],
        "govev_2_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "govev_3_match": {
            "top_left_x": 0,
            "top_left_y": 0,
            "width": 0,
            "height": 0,
            "score": 0.0,
            "found": False,
        },
    }


def get_validation_errors(frame_data: Any) -> List[str]:
    """
    Kare çıktısındaki doğrulama hatalarını listeler (hata ayıklama için).

    Args:
        frame_data: Doğrulanacak kare sözlüğü.

    Returns:
        Hata mesajları listesi; boş liste geçerli demektir.
    """
    errors: List[str] = []

    if not isinstance(frame_data, dict):
        return ["Kare verisi sözlük olmalıdır."]

    for key in REQUIRED_FRAME_KEYS:
        if key not in frame_data:
            errors.append(f"Eksik anahtar: '{key}'")

    if errors:
        return errors

    if not _is_int(frame_data["frame_id"]) or frame_data["frame_id"] < 0:
        errors.append("'frame_id' negatif olmayan tam sayı olmalıdır.")

    if not _is_number(frame_data["timestamp"]) or frame_data["timestamp"] < 0.0:
        errors.append("'timestamp' negatif olmayan sayı olmalıdır.")

    objects = frame_data["govev_1_objects"]
    if not isinstance(objects, list):
        errors.append("'govev_1_objects' liste olmalıdır.")
    else:
        for idx, obj in enumerate(objects):
            if not _validate_govev_1_object(obj):
                errors.append(f"govev_1_objects[{idx}] geçersiz.")

    if not _validate_govev_2_position(frame_data["govev_2_position"]):
        errors.append("'govev_2_position' geçersiz.")

    if not _validate_govev_3_match(frame_data["govev_3_match"]):
        errors.append("'govev_3_match' geçersiz.")

    return errors
