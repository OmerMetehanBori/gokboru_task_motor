"""
VisDrone → Gökbörü Dataset Hazırlama Scripti

İşlem sırası:
  1. Mevcut dataset/ klasörünü temizler.
  2. VisDrone2019-DET train+val zip dosyalarını indirir.
  3. VisDrone annotation formatını YOLO formatına çevirir.
  4. Sınıfları Tasit (0) ve Insan (1) olarak yeniden etiketler.
  5. data.yaml dosyasını nc=2 ile günceller.

VisDrone sınıf eşleştirmesi:
  1=pedestrian  → 1 (Insan)
  2=people      → 1 (Insan)
  3=bicycle     → 0 (Tasit)
  4=car         → 0 (Tasit)
  5=van         → 0 (Tasit)
  6=truck       → 0 (Tasit)
  7=tricycle    → 0 (Tasit)
  8=awning-tri  → 0 (Tasit)
  9=bus         → 0 (Tasit)
  10=motor      → 0 (Tasit)
  0=ignored, 11=others → atlanır
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VisDronePrep")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TEMP_DIR = PROJECT_ROOT / "_visdrone_tmp"
DATA_YAML = PROJECT_ROOT / "data.yaml"

VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}

# VisDrone kategori ID → Gökbörü sınıf ID
# 0=ignored, 11=others → None (atlanır)
VIS_TO_GOKBORU: dict[int, int | None] = {
    0:  None,  # ignored regions
    1:  1,     # pedestrian → Insan
    2:  1,     # people     → Insan
    3:  0,     # bicycle    → Tasit
    4:  0,     # car        → Tasit
    5:  0,     # van        → Tasit
    6:  0,     # truck      → Tasit
    7:  0,     # tricycle   → Tasit
    8:  0,     # awning-tricycle → Tasit
    9:  0,     # bus        → Tasit
    10: 0,     # motor      → Tasit
    11: None,  # others
}


# ---------------------------------------------------------------------------
# Adım 1: Mevcut dataset temizleme
# ---------------------------------------------------------------------------
def clean_dataset() -> None:
    log.info("Mevcut dataset temizleniyor: %s", DATASET_DIR)
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)
    log.info("Dataset klasörü sıfırlandı.")


# ---------------------------------------------------------------------------
# Adım 2: VisDrone zip indirme
# ---------------------------------------------------------------------------
def _curl_download(url: str, dest: Path) -> None:
    """curl.exe ile hızlı indirme — ilerlemeyi doğrudan terminale yazar."""
    cmd = [
        "curl.exe",
        "-L",
        "--retry", "5",
        "--retry-delay", "3",
        "--retry-connrefused",
        "-o", str(dest),
        "--progress-bar",
        "--max-time", "7200",  # maks 2 saat
        url,
    ]
    log.info("curl ile indiriliyor → %s", dest.name)
    # stdout/stderr'ı inherit et ki ilerleme terminalde görünsün
    result = subprocess.run(cmd, check=False, stdout=None, stderr=None)
    if result.returncode not in (0, 18, 23):  # 18=kısmi, 23=yazma hatası
        raise RuntimeError(f"curl başarısız: returncode={result.returncode}")


def download_visdrone() -> dict[str, Path]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}

    for split, url in VISDRONE_URLS.items():
        zip_path = TEMP_DIR / f"VisDrone2019-DET-{split}.zip"
        extract_dir = TEMP_DIR / f"VisDrone2019-DET-{split}"

        if extract_dir.exists():
            log.info("[%s] Zaten indirilmiş, atlanıyor.", split)
        else:
            _curl_download(url, zip_path)

            log.info("[%s] Çıkartılıyor...", split)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(TEMP_DIR)
            zip_path.unlink(missing_ok=True)
            log.info("[%s] Tamamlandı.", split)

        vd_root = TEMP_DIR / f"VisDrone2019-DET-{split}"
        extracted[split] = vd_root

    return extracted


# ---------------------------------------------------------------------------
# Adım 3 & 4: VisDrone annotation → YOLO + sınıf dönüşümü
# ---------------------------------------------------------------------------
def visdrone_to_yolo(vd_root: Path, split: str) -> tuple[int, int]:
    """
    VisDrone annotation dosyalarını YOLO formatına çevirir.

    VisDrone format: bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
    YOLO format: class_id cx cy w h (normalize edilmiş)

    Returns:
        (dönüştürülen görüntü sayısı, toplam etiket satırı)
    """
    img_src_dir = vd_root / "images"
    ann_src_dir = vd_root / "annotations"

    img_dst_dir = DATASET_DIR / "images" / split
    lbl_dst_dir = DATASET_DIR / "labels" / split

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_src_dir.glob("*.jpg")) + sorted(img_src_dir.glob("*.png"))
    converted = 0
    total_labels = 0

    for img_path in img_files:
        ann_path = ann_src_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        # Görüntü boyutlarını oku
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Görüntü okunamadı: %s", img_path)
            continue
        img_h, img_w = img.shape[:2]

        yolo_lines: list[str] = []

        with ann_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                bbox_left   = int(parts[0])
                bbox_top    = int(parts[1])
                bbox_width  = int(parts[2])
                bbox_height = int(parts[3])
                # parts[4] = score (genellikle 1)
                category    = int(parts[5])

                gokboru_cls = VIS_TO_GOKBORU.get(category)
                if gokboru_cls is None:
                    continue

                # Sıfır/negatif boyutları atla
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # YOLO normalize koordinatlar
                cx = (bbox_left + bbox_width  / 2) / img_w
                cy = (bbox_top  + bbox_height / 2) / img_h
                nw = bbox_width  / img_w
                nh = bbox_height / img_h

                # Sınır kontrolü
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(f"{gokboru_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Görüntüyü kopyala
        shutil.copy2(img_path, img_dst_dir / img_path.name)

        # Etiket dosyasını yaz (boş olsa bile)
        lbl_path = lbl_dst_dir / (img_path.stem + ".txt")
        lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        total_labels += len(yolo_lines)
        converted += 1

    log.info("[%s] %d görüntü dönüştürüldü, %d etiket yazıldı.", split, converted, total_labels)
    return converted, total_labels


# ---------------------------------------------------------------------------
# Adım 5: data.yaml güncelleme
# ---------------------------------------------------------------------------
def update_data_yaml() -> None:
    content = """\
# Gökbörü Drone Veri Seti — VisDrone tabanlı (Tasit + Insan)
path: ./dataset
train: images/train
val:   images/val

nc: 2
names:
  0: Tasit
  1: Insan
"""
    DATA_YAML.write_text(content, encoding="utf-8")
    log.info("data.yaml güncellendi: nc=2, sınıflar=[Tasit, Insan]")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("Gökbörü VisDrone Dataset Hazırlama Başlıyor")
    log.info("=" * 60)

    clean_dataset()
    extracted = download_visdrone()

    stats: dict[str, tuple[int, int]] = {}
    for split, vd_root in extracted.items():
        log.info("\n[%s] Dönüştürme başlıyor...", split)
        stats[split] = visdrone_to_yolo(vd_root, split)

    update_data_yaml()

    # Özet rapor
    log.info("\n" + "=" * 60)
    log.info("ÖZET")
    log.info("=" * 60)
    total_imgs = 0
    total_lbls = 0
    for split, (imgs, lbls) in stats.items():
        log.info("  %-6s → %5d görüntü, %6d etiket", split, imgs, lbls)
        total_imgs += imgs
        total_lbls += lbls
    log.info("  TOPLAM → %5d görüntü, %6d etiket", total_imgs, total_lbls)
    log.info("=" * 60)
    log.info("Dataset hazır! Eğitim için: python train.py --device cpu")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
