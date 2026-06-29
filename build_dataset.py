"""
Gökbörü Otomatik Veri Seti Oluşturucu
======================================
1. Videoları tarar ve belirlenen FPS ile kare çıkarır
2. YOLOv8n (COCO) ile otomatik ön-etiketleme yapar
3. COCO sınıflarını Gökbörü sınıflarına eşler
4. YOLO normalize bbox formatında .txt etiketleri üretir
5. %80 train / %20 val olarak böler
6. gokboru_task_motor/dataset/ altına yerleştirir
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GokboruDatasetBuilder")

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_DIR    = Path(r"C:\Users\Metehan\Desktop\trafik videolari")
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL   = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL   = DATASET_DIR / "labels" / "val"

# ── Parametreler ───────────────────────────────────────────────────────────────
EXTRACT_FPS    = 0.5    # Saniyede kaç kare çıkarılacak (1 kare / 2 sn)
CONF_THRESHOLD = 0.45   # Minimum güven eşiği
TRAIN_RATIO    = 0.80   # Eğitim seti oranı
RANDOM_SEED    = 42
IMAGE_QUALITY  = 90     # JPEG kalitesi

# ── COCO → Gökbörü sınıf eşlemesi ─────────────────────────────────────────────
# Gökbörü: 0=Tasit, 1=Insan, 2=UAP, 3=UAI
COCO_TO_GOKBORU = {
    0: 1,   # person  → Insan
    2: 0,   # car     → Tasit
    3: 0,   # motorcycle → Tasit
    5: 0,   # bus     → Tasit
    7: 0,   # truck   → Tasit
}
# UAP (2) ve UAI (3) bu videolarda yok → atlanır


def setup_directories() -> None:
    """Tüm dataset dizin yapısını oluşturur."""
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Dizin yapısı hazır: %s", DATASET_DIR)


def load_yolo_model():
    """YOLOv8n COCO modelini yükler."""
    try:
        from ultralytics import YOLO
        logger.info("YOLOv8n (COCO) yükleniyor...")
        model = YOLO("yolo26n.pt")
        logger.info("Model hazır.")
        return model
    except ImportError:
        logger.error("ultralytics yüklü değil! Kur: pip install ultralytics")
        raise


def run_inference(model, frame: np.ndarray, w: int, h: int) -> list[str]:
    """Tek kare üzerinde YOLO çıkarımı; YOLO normalize bbox satırları döner."""
    lines: list[str] = []
    try:
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        if not results:
            return lines

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return lines

        for box in boxes:
            coco_cls = int(box.cls[0])
            if coco_cls not in COCO_TO_GOKBORU:
                continue

            gokboru_cls = COCO_TO_GOKBORU[coco_cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = max(0.0, min(1.0, ((x1 + x2) / 2) / w))
            cy = max(0.0, min(1.0, ((y1 + y2) / 2) / h))
            bw = max(0.001, min(1.0, (x2 - x1) / w))
            bh = max(0.001, min(1.0, (y2 - y1) / h))

            lines.append(f"{gokboru_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    except Exception as exc:
        logger.debug("Çıkarım hatası: %s", exc)

    return lines


def extract_and_label_video(
    video_path: Path,
    model,
    video_idx: int,
    temp_dir: Path,
) -> list[tuple[Path, Path]]:
    """Tek videodan kare çıkarır, etiketler. Döner: [(img_path, lbl_path)]"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Video açılamadı: %s", video_path.name)
        return []

    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_fps > 120:
        source_fps = 25.0

    frame_interval = max(1, int(round(source_fps / EXTRACT_FPS)))
    duration_sec   = total_frames / source_fps

    logger.info(
        "[%02d] %-50s | %.1f sn | her %d. kare",
        video_idx, video_path.name[:50], duration_sec, frame_interval,
    )

    pairs: list[tuple[Path, Path]] = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_num % frame_interval == 0:
            stem     = f"v{video_idx:02d}_f{frame_num:06d}"
            img_path = temp_dir / f"{stem}.jpg"
            lbl_path = temp_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])

            h, w = frame.shape[:2]
            label_lines = run_inference(model, frame, w, h)
            lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

            pairs.append((img_path, lbl_path))

        frame_num += 1

    cap.release()
    labeled = sum(1 for _, lp in pairs if lp.read_text().strip())
    logger.info("[%02d] %d kare → %d etiketli", video_idx, len(pairs), labeled)
    return pairs


def split_and_organize(all_pairs: list[tuple[Path, Path]]) -> dict:
    """Train/val böler ve dosyaları hedef dizinlere kopyalar."""
    random.seed(RANDOM_SEED)
    shuffled = list(all_pairs)
    random.shuffle(shuffled)

    split_idx   = int(len(shuffled) * TRAIN_RATIO)
    train_pairs = shuffled[:split_idx]
    val_pairs   = shuffled[split_idx:]

    stats = {"train": 0, "val": 0, "train_with_labels": 0, "val_with_labels": 0}

    for pairs, img_dir, lbl_dir, key in [
        (train_pairs, IMAGES_TRAIN, LABELS_TRAIN, "train"),
        (val_pairs,   IMAGES_VAL,   LABELS_VAL,   "val"),
    ]:
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            shutil.copy2(lbl_src, lbl_dir / lbl_src.name)
            stats[key] += 1
            if lbl_src.read_text().strip():
                stats[f"{key}_with_labels"] += 1

    return stats


def class_distribution(all_pairs: list[tuple[Path, Path]]) -> dict[int, int]:
    counts: dict[int, int] = {0: 0, 1: 0}
    for _, lbl in all_pairs:
        if not lbl.exists():
            continue
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def print_report(stats: dict, dist: dict[int, int], total: int) -> None:
    logger.info("=" * 55)
    logger.info("  VERİ SETİ ÖZET RAPORU")
    logger.info("=" * 55)
    logger.info("  Toplam kare    : %d", total)
    logger.info("  Train          : %d  (%d etiketli)", stats["train"], stats["train_with_labels"])
    logger.info("  Val            : %d  (%d etiketli)", stats["val"],   stats["val_with_labels"])
    logger.info("  Sınıf dağılımı :")
    logger.info("    Tasit (0)    : %d", dist.get(0, 0))
    logger.info("    Insan (1)    : %d", dist.get(1, 0))
    logger.info("=" * 55)


def save_report(stats: dict, dist: dict[int, int], total: int) -> None:
    report = {
        "total_frames": total,
        "train_frames": stats["train"],
        "val_frames":   stats["val"],
        "train_labeled": stats["train_with_labels"],
        "val_labeled":   stats["val_with_labels"],
        "class_distribution": {"Tasit": dist.get(0, 0), "Insan": dist.get(1, 0)},
    }
    out = DATASET_DIR / "dataset_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rapor: %s", out)


def find_videos(directory: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    return sorted(p for p in directory.iterdir() if p.suffix in exts)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  GÖKBÖRÜ OTOMATİK VERİ SETİ OLUŞTURUCU")
    logger.info("=" * 55)

    # Video dizinini bul (Türkçe karakter toleransı)
    video_dir = VIDEO_DIR
    if not video_dir.exists():
        # Windows'ta Türkçe karakter sorununa karşı alternatif yol dene
        alt = Path(r"C:\Users\Metehan\Desktop") / "trafik videoları"
        if alt.exists():
            video_dir = alt
        else:
            logger.error("Video dizini bulunamadı: %s", VIDEO_DIR)
            logger.error("Lütfen VIDEO_DIR değişkenini kontrol edin.")
            return

    videos = find_videos(video_dir)
    if not videos:
        logger.error("Video bulunamadı: %s", video_dir)
        return

    logger.info("%d video bulundu.", len(videos))

    setup_directories()
    model = load_yolo_model()

    temp_dir = DATASET_DIR / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[tuple[Path, Path]] = []
    for idx, vpath in enumerate(videos, start=1):
        pairs = extract_and_label_video(vpath, model, idx, temp_dir)
        all_pairs.extend(pairs)

    if not all_pairs:
        logger.error("Hiç kare çıkarılamadı!")
        return

    logger.info("Toplam %d kare. Bölünüyor...", len(all_pairs))
    stats = split_and_organize(all_pairs)
    dist  = class_distribution(all_pairs)

    print_report(stats, dist, len(all_pairs))
    save_report(stats, dist, len(all_pairs))

    # Geçici klasörü temizle
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Geçici dosyalar temizlendi.")
    logger.info("Hazır! Şimdi çalıştırabilirsiniz: python train.py --epochs 100 --batch 8")


if __name__ == "__main__":
    main()
