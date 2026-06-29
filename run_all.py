"""
Gökbörü tam pipeline: dataset → eğitim → asset hazırlığı → video işleme.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GokboruPipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

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


def run_step(label: str, args: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("ADIM: %s", label)
    logger.info("Komut: %s %s", PYTHON, " ".join(args))
    logger.info("=" * 60)
    result = subprocess.run(
        [PYTHON, *args],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"'{label}' adımı başarısız (kod={result.returncode})")


def dataset_ready() -> bool:
    train_dir = PROJECT_ROOT / "dataset" / "images" / "train"
    if not train_dir.exists():
        return False
    return len(list(train_dir.glob("*.jpg"))) > 0


def model_ready() -> bool:
    return (PROJECT_ROOT / "models" / "best.pt").exists()


def list_videos(video_dir: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)


def main() -> int:
    video_dir = resolve_video_dir()
    videos = list_videos(video_dir)
    if not videos:
        logger.error("Video bulunamadı: %s", video_dir)
        return 1

    logger.info("%d video bulundu: %s", len(videos), video_dir)

    if not dataset_ready():
        logger.info("Dataset boş — build_dataset.py çalıştırılıyor...")
        run_step("Veri Seti Oluşturma", ["build_dataset.py"])
    else:
        logger.info("Dataset mevcut, build_dataset atlanıyor.")

    if not model_ready():
        logger.info("Model yok — train.py çalıştırılıyor...")
        run_step(
            "Model Eğitimi",
            [
                "train.py",
                "--data",
                "data.yaml",
                "--epochs",
                "100",
                "--batch",
                "8",
                "--imgsz",
                "1280",
                "--device",
                "0",
                "--workers",
                "2",
            ],
        )
    else:
        logger.info("Model mevcut (models/best.pt), eğitim atlanıyor.")

    run_step("Yardımcı Dosya Hazırlığı", ["prepare_assets.py"])

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, video in enumerate(videos, start=1):
        out_file = output_dir / f"gokboru_session_{idx:02d}_{video.stem[:30]}.json"
        run_step(
            f"Video İşleme [{idx}/{len(videos)}] {video.name}",
            [
                "main.py",
                "--video",
                str(video),
                "--telemetry",
                str(PROJECT_ROOT / "data" / "telemetri.csv"),
                "--template",
                str(PROJECT_ROOT / "data" / "referans_obje.png"),
                "--model",
                str(PROJECT_ROOT / "models" / "best.pt"),
                "--output",
                str(out_file),
            ],
        )

    logger.info("=" * 60)
    logger.info("TÜM İŞLEMLER TAMAMLANDI")
    logger.info("Çıktılar: %s", output_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
