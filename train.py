"""
Gökbörü YOLOv8 Model Eğitimi — Drone (İHA) Görüntüleri.

Kendi toplanan havadan etiketli fotoğraflarla özel nesne tespit modeli eğitir.
Küçük nesneler (araç, insan) için yüksek çözünürlük (1280 px) kullanılır.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GokboruTrainer")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = "data.yaml"
DEFAULT_BASE_MODEL = "yolo26n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 8
DEFAULT_IMGSZ = 1280
DEFAULT_DEVICE = 0
DEFAULT_WORKERS = 2
DEFAULT_RUN_NAME = "gokboru_train"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_WEIGHTS_TARGET = MODELS_DIR / "best.pt"


def resolve_data_yaml(data_yaml: str) -> Path:
    """data.yaml yolunu proje köküne göre çözümler."""
    path = Path(data_yaml)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_base_model(base_model: str) -> str:
    """Temel YOLO ağırlık dosyasının yolunu çözümler."""
    path = Path(base_model)
    if path.is_absolute() and path.exists():
        return str(path)

    local_candidate = PROJECT_ROOT / base_model
    if local_candidate.exists():
        return str(local_candidate.resolve())

    return base_model


def copy_best_weights(save_dir: Path) -> bool:
    """Eğitim sonrası best.pt dosyasını models/ klasörüne kopyalar."""
    source = save_dir / "weights" / "best.pt"
    if not source.exists():
        logger.warning("best.pt bulunamadı: %s", source)
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, BEST_WEIGHTS_TARGET)
    logger.info("En iyi ağırlık kopyalandı: %s", BEST_WEIGHTS_TARGET)
    return True


def train_model(
    data_yaml: str = DEFAULT_DATA_YAML,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    imgsz: int = DEFAULT_IMGSZ,
    device: Union[int, str] = DEFAULT_DEVICE,
    workers: int = DEFAULT_WORKERS,
    project_dir: Optional[str] = None,
    run_name: str = DEFAULT_RUN_NAME,
) -> int:
    """
    YOLOv8 modelini drone veri seti üzerinde eğitir.

    Args:
        data_yaml: Veri seti yapılandırma dosyası (varsayılan: data.yaml).
        base_model: Başlangıç ağırlık dosyası.
        epochs: Eğitim epoch sayısı (varsayılan: 100).
        batch: Batch boyutu (varsayılan: 8).
        imgsz: Giriş görüntü boyutu (varsayılan: 1280).
        device: GPU indeksi veya 'cpu' (varsayılan: 0).
        workers: DataLoader worker sayısı (varsayılan: 2).
        project_dir: Ultralytics çıktı klasörü.
        run_name: Eğitim koşusu adı.

    Returns:
        Başarılıysa 0, aksi halde 1.
    """
    data_path = resolve_data_yaml(data_yaml)
    if not data_path.exists():
        logger.error("data.yaml bulunamadı: %s", data_path)
        logger.error(
            "Veri seti yapılandırması oluşturulmalıdır. Örnek sınıflar: "
            "0=Tasit, 1=Insan"
        )
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics paketi yüklü değil. Kurulum: pip install ultralytics"
        )
        return 1

    base_weights = resolve_base_model(base_model)
    output_project = project_dir or str(PROJECT_ROOT / "runs" / "detect")

    logger.info("=" * 60)
    logger.info("Gökbörü YOLOv8 Drone Model Eğitimi Başlatılıyor")
    logger.info("=" * 60)
    logger.info("Veri seti      : %s", data_path)
    logger.info("Temel model    : %s", base_weights)
    logger.info("Epoch          : %d", epochs)
    logger.info("Batch          : %d", batch)
    logger.info("Görüntü boyutu : %d px", imgsz)
    logger.info("Cihaz          : %s", device)
    logger.info("Workers        : %d", workers)
    logger.info("Çıktı klasörü  : %s/%s", output_project, run_name)
    logger.info("=" * 60)

    try:
        model = YOLO(base_weights)
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=output_project,
            name=run_name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            patience=20,
            save=True,
            plots=True,
        )

        save_dir = Path(results.save_dir)
        if copy_best_weights(save_dir):
            logger.info("Eğitim tamamlandı. Model hazır: %s", BEST_WEIGHTS_TARGET)
        else:
            logger.warning(
                "Eğitim bitti ancak best.pt kopyalanamadı. "
                "Manuel kontrol: %s/weights/best.pt",
                save_dir,
            )

        return 0

    except Exception as exc:
        logger.error("Eğitim sırasında hata oluştu: %s", exc, exc_info=True)
        return 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gökbörü YOLOv8 Drone Model Eğitimi — TEKNOFEST 2026"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_YAML,
        help=f"Veri seti YAML dosyası (varsayılan: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Başlangıç YOLO ağırlığı (varsayılan: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Epoch sayısı (varsayılan: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch boyutu (varsayılan: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Giriş görüntü boyutu (varsayılan: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--device",
        default=str(DEFAULT_DEVICE),
        help=f"Eğitim cihazı — GPU indeksi veya 'cpu' (varsayılan: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"DataLoader worker sayısı (varsayılan: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics çıktı proje klasörü",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_RUN_NAME,
        help=f"Eğitim koşusu adı (varsayılan: {DEFAULT_RUN_NAME})",
    )
    return parser


def parse_device(raw_device: str) -> Union[int, str]:
    """Komut satırı cihaz argümanını çözümler."""
    if raw_device.lower() == "cpu":
        return "cpu"
    try:
        return int(raw_device)
    except ValueError:
        return raw_device


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    return train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=parse_device(str(args.device)),
        workers=args.workers,
        project_dir=args.project,
        run_name=args.name,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
