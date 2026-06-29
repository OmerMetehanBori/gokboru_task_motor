"""
Tüm trafik videolarını Gökbörü pipeline ile işler.
Her video için telemetri + JSON çıktısı üretir.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ProcessVideos")

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_SRC = PROJECT_ROOT / "runs" / "detect" / "gokboru_trafik_cpu" / "weights" / "best.pt"
MODEL_DST = PROJECT_ROOT / "models" / "best.pt"
TEMPLATE_PATH = DATA_DIR / "referans_obje.png"


def resolve_video_dir() -> Path:
    candidates = [
        Path(r"C:\Users\Metehan\Desktop\trafik videolari"),
        Path(r"C:\Users\Metehan\Desktop\trafik videoları"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    desktop = Path(r"C:\Users\Metehan\Desktop")
    for item in desktop.iterdir():
        if item.is_dir() and "trafik" in item.name.lower():
            return item
    raise FileNotFoundError("Video klasörü bulunamadı.")


def list_videos(video_dir: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)


def ensure_model() -> None:
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DST.exists():
        logger.info("Model mevcut: %s", MODEL_DST)
        return
    if MODEL_SRC.exists():
        shutil.copy2(MODEL_SRC, MODEL_DST)
        logger.info("Model kopyalandı: %s", MODEL_DST)
        return
    raise FileNotFoundError("Eğitilmiş model bulunamadı. Önce train.py çalıştırın.")


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 600
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 1)


def write_telemetry(video_path: Path, output_csv: Path, health_cutoff: int = 450) -> None:
    import numpy as np

    total_frames = get_frame_count(video_path)
    health_cutoff = min(health_cutoff, max(total_frames // 2, 30))

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "x", "y", "z", "health"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Telemetri: %s (%d kare, health=0 @ kare %d)",
        output_csv.name,
        total_frames,
        health_cutoff,
    )


def ensure_template(video_dir: Path) -> None:
    if TEMPLATE_PATH.exists():
        return
    subprocess.run([PYTHON, "prepare_assets.py"], cwd=str(PROJECT_ROOT), check=True)


def safe_stem(name: str, max_len: int = 40) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe[:max_len]


def process_video(video_path: Path, index: int) -> int:
    telemetry_csv = DATA_DIR / f"telemetri_{index:02d}.csv"
    output_json = OUTPUT_DIR / f"gokboru_session_{index:02d}_{safe_stem(video_path.stem)}.json"

    write_telemetry(video_path, telemetry_csv)

    cmd = [
        PYTHON,
        "main.py",
        "--video",
        str(video_path),
        "--telemetry",
        str(telemetry_csv),
        "--template",
        str(TEMPLATE_PATH),
        "--model",
        str(MODEL_DST),
        "--output",
        str(output_json),
    ]

    logger.info("=" * 60)
    logger.info("[%02d] %s", index, video_path.name)
    logger.info("Çıktı: %s", output_json.name)
    logger.info("=" * 60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> int:
    video_dir = resolve_video_dir()
    videos = list_videos(video_dir)
    if not videos:
        logger.error("Video bulunamadı.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_model()
    ensure_template(video_dir)

    logger.info("%d video işlenecek.", len(videos))

    failed = []
    for idx, video in enumerate(videos, start=1):
        code = process_video(video, idx)
        if code != 0:
            failed.append(video.name)
            logger.error("Başarısız: %s (kod=%d)", video.name, code)
        else:
            logger.info("Tamamlandı: %s", video.name)

    logger.info("=" * 60)
    if failed:
        logger.warning("Başarısız videolar: %s", ", ".join(failed))
        return 1

    logger.info("Tüm videolar başarıyla işlendi. Çıktılar: %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
