"""
Video işleme öncesi gerekli yardımcı dosyaları üretir:
  - data/telemetri.csv   (synthetic telemetri + health geçişi)
  - data/referans_obje.png (ilk videodan araç kırpımı şablon)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("PrepareAssets")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TELEMETRY_PATH = DATA_DIR / "telemetri.csv"
TEMPLATE_PATH = DATA_DIR / "referans_obje.png"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

VIDEO_DIR_CANDIDATES = [
    Path(r"C:\Users\Metehan\Desktop\trafik videolari"),
    Path(r"C:\Users\Metehan\Desktop\trafik videoları"),
]


def resolve_video_dir() -> Path:
    for candidate in VIDEO_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    desktop = Path(r"C:\Users\Metehan\Desktop")
    for item in desktop.iterdir():
        if item.is_dir() and "trafik" in item.name.lower():
            return item
    raise FileNotFoundError("Trafik video klasörü bulunamadı.")


def find_first_video(video_dir: Path) -> Path:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)
    if not videos:
        raise FileNotFoundError(f"Video bulunamadı: {video_dir}")
    return videos[0]


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 450
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 450)


def generate_telemetry(total_frames: int, health_cutoff: int = 450) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    x, y, z = 41.015137, 28.979530, 45.0

    for frame_id in range(total_frames):
        health = 1 if frame_id < health_cutoff else 0
        if health == 1:
            x += 0.00002 * np.sin(frame_id / 30.0)
            y += 0.00002 * np.cos(frame_id / 30.0)
            z = 45.0 + 0.01 * np.sin(frame_id / 50.0)
        rows.append(
            {
                "frame_id": frame_id,
                "x": round(x, 6),
                "y": round(y, 6),
                "z": round(z, 3),
                "health": health,
            }
        )

    with TELEMETRY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "x", "y", "z", "health"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Telemetri oluşturuldu: %s (%d satır)", TELEMETRY_PATH, len(rows))


def extract_reference_template(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("İlk kare okunamadı.")

    crop = None

    if MODEL_PATH.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(MODEL_PATH))
            results = model.predict(frame, conf=0.35, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in (0, 1, 2, 3, 5, 7):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        h, w = frame.shape[:2]
                        pad = 8
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                        if x2 > x1 + 20 and y2 > y1 + 20:
                            crop = frame[y1:y2, x1:x2].copy()
                            break
        except Exception as exc:
            logger.warning("Model ile şablon çıkarımı başarısız: %s", exc)

    if crop is None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = min(w, h) // 6
        crop = frame[cy - size : cy + size, cx - size : cx + size].copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEMPLATE_PATH), crop)
    logger.info("Referans şablon kaydedildi: %s (%dx%d)", TEMPLATE_PATH, crop.shape[1], crop.shape[0])


def main() -> None:
    video_dir = resolve_video_dir()
    first_video = find_first_video(video_dir)
    logger.info("Video klasörü: %s", video_dir)
    logger.info("Referans video: %s", first_video.name)

    frame_count = get_video_frame_count(first_video)
    generate_telemetry(total_frames=max(frame_count, 600), health_cutoff=min(450, frame_count // 2))
    extract_reference_template(first_video)
    logger.info("Yardımcı dosyalar hazır.")


if __name__ == "__main__":
    main()
