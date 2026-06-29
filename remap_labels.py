"""
Gökbörü Evrensel Etiket Dönüştürücü

Herhangi bir YOLO formatındaki dataseti otomatik olarak
Gökbörü şemasına (Tasit=0, Insan=1) çevirir.

Desteklenen kaynak formatları:
  - Numeric ID'li YOLO labellar + data.yaml (class names listesi)
  - "people", "pedestrian", "car", "vehicle" gibi string isimlerle eşleşme
  - VisDrone, COCO, Roboflow export'ları

Kullanım:
  python remap_labels.py --dataset ./dataset --yaml data.yaml
  python remap_labels.py --dataset ./baska_dataset --source-classes "car,truck,bus,person,people"
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LabelRemap")

# ---------------------------------------------------------------------------
# Evrensel sınıf ismi → Gökbörü ID eşleşme tablosu
# ---------------------------------------------------------------------------
# Tasit = 0
TASIT_NAMES = {
    "tasit", "vehicle", "car", "auto", "automobile",
    "truck", "lorry", "van", "pickup",
    "bus", "minibus", "coach",
    "bicycle", "bike", "cycle",
    "motorbike", "motorcycle", "motor", "moto",
    "tricycle", "awning-tricycle", "awningtricycle",
    "scooter", "vespa",
    "tractor", "heavy_machinery", "machinery",
    "taxi", "cab",
    "suv", "jeep", "minivan",
    # Türkçe
    "araç", "araba", "kamyon", "otobüs", "bisiklet", "motorsiklet",
}

# Insan = 1
INSAN_NAMES = {
    "insan", "person", "people", "pedestrian", "human",
    "man", "woman", "child", "adult",
    "walker", "cyclist", "rider",
    # Türkçe
    "yaya", "insan", "çocuk",
}


def normalize(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def name_to_gokboru(name: str) -> int | None:
    n = normalize(name)
    for tasit_name in TASIT_NAMES:
        if n == normalize(tasit_name) or normalize(tasit_name) in n or n in normalize(tasit_name):
            return 0
    for insan_name in INSAN_NAMES:
        if n == normalize(insan_name) or normalize(insan_name) in n or n in normalize(insan_name):
            return 1
    return None


def build_id_map(class_names: list[str]) -> dict[int, int | None]:
    """Kaynak class listesini Gökbörü ID'lerine eşleştirir."""
    mapping: dict[int, int | None] = {}
    for idx, name in enumerate(class_names):
        gokboru_id = name_to_gokboru(name)
        mapping[idx] = gokboru_id
        if gokboru_id is not None:
            gname = "Tasit" if gokboru_id == 0 else "Insan"
            log.info("  [%2d] %-25s → %s (%d)", idx, name, gname, gokboru_id)
        else:
            log.info("  [%2d] %-25s → ATLANACAK", idx, name)
    return mapping


def load_classes_from_yaml(yaml_path: Path) -> list[str]:
    """data.yaml dosyasından class isimlerini okur (PyYAML gerektirmez)."""
    names: list[str] = []
    in_names = False
    with yaml_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if re.match(r"^names\s*:", stripped):
                in_names = True
                inline = re.sub(r"^names\s*:\s*", "", stripped)
                if inline.startswith("["):
                    items = re.findall(r"[\w\-\. ]+", inline)
                    names = [i.strip() for i in items if i.strip()]
                    in_names = False
                continue
            if in_names:
                m = re.match(r"^\s+\d+\s*:\s*(.+)$", line)
                if m:
                    names.append(m.group(1).strip().strip("'\""))
                elif re.match(r"^\s+-\s+(.+)$", line):
                    m2 = re.match(r"^\s+-\s+(.+)$", line)
                    names.append(m2.group(1).strip().strip("'\""))  # type: ignore[union-attr]
                elif stripped and not stripped.startswith("#") and not re.match(r"^\s", line):
                    in_names = False
    return names


def remap_label_file(
    src: Path,
    dst: Path,
    id_map: dict[int, int | None],
    img_w: int = 0,
    img_h: int = 0,
) -> tuple[int, int]:
    """
    Tek bir YOLO label dosyasını dönüştürür.
    Returns: (yazılan satır sayısı, atlanan satır sayısı)
    """
    written = 0
    skipped = 0
    out_lines: list[str] = []

    with src.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            raw_cls = parts[0]

            # String class name desteği (nadir ama bazı Roboflow export'larında olur)
            if not raw_cls.lstrip("-").isdigit():
                gokboru_id = name_to_gokboru(raw_cls)
            else:
                src_id = int(raw_cls)
                gokboru_id = id_map.get(src_id)

            if gokboru_id is None:
                skipped += 1
                continue

            if len(parts) >= 5:
                out_lines.append(f"{gokboru_id} {' '.join(parts[1:5])}")
                written += 1
            else:
                skipped += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return written, skipped


def remap_dataset(
    dataset_dir: Path,
    id_map: dict[int, int | None],
    in_place: bool = True,
) -> None:
    label_files = list(dataset_dir.rglob("labels/**/*.txt"))
    if not label_files:
        label_files = list(dataset_dir.rglob("**/*.txt"))
        label_files = [f for f in label_files if "images" not in str(f)]

    total_written = 0
    total_skipped = 0
    converted_files = 0

    for lbl_path in label_files:
        if lbl_path.name in ("classes.txt", "data.yaml"):
            continue

        if in_place:
            dst = lbl_path
            tmp = lbl_path.with_suffix(".tmp")
            w, s = remap_label_file(lbl_path, tmp, id_map)
            shutil.move(str(tmp), str(dst))
        else:
            dst = lbl_path
            w, s = remap_label_file(lbl_path, dst, id_map)

        total_written += w
        total_skipped += s
        converted_files += 1

    log.info("─" * 50)
    log.info("Dönüştürülen dosya : %d", converted_files)
    log.info("Yazılan etiket     : %d", total_written)
    log.info("Atlanan etiket     : %d  (bilinmeyen sınıf)", total_skipped)

    # Sınıf dağılımı kontrolü
    cls_count = {0: 0, 1: 0}
    for lbl_path in label_files:
        if not lbl_path.exists():
            continue
        with lbl_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    c = int(parts[0])
                    if c in cls_count:
                        cls_count[c] += 1
    log.info("Tasit  (0) etiketi : %d", cls_count[0])
    log.info("Insan  (1) etiketi : %d", cls_count[1])


def update_yaml(yaml_path: Path) -> None:
    content = """\
# Gökbörü Drone Veri Seti — Tasit + Insan
path: ./dataset
train: images/train
val:   images/val

nc: 2
names:
  0: Tasit
  1: Insan
"""
    yaml_path.write_text(content, encoding="utf-8")
    log.info("data.yaml güncellendi: nc=2 [Tasit, Insan]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gökbörü Evrensel Etiket Dönüştürücü"
    )
    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset kök dizini (varsayılan: ./dataset)",
    )
    parser.add_argument(
        "--yaml",
        default="./data.yaml",
        help="Kaynak data.yaml dosyası (sınıf isimleri için)",
    )
    parser.add_argument(
        "--source-classes",
        default=None,
        help="Virgülle ayrılmış kaynak sınıf listesi, örn: 'car,truck,person,people'",
    )
    parser.add_argument(
        "--update-yaml",
        action="store_true",
        default=True,
        help="data.yaml'ı nc=2 ile güncelle",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    yaml_path = Path(args.yaml).resolve()

    log.info("=" * 55)
    log.info("Gökbörü Etiket Dönüştürücü")
    log.info("=" * 55)
    log.info("Dataset: %s", dataset_dir)

    # Sınıf listesini belirle
    class_names: list[str] = []

    if args.source_classes:
        class_names = [c.strip() for c in args.source_classes.split(",")]
        log.info("Kullanıcı tanımlı sınıflar: %s", class_names)
    elif yaml_path.exists():
        class_names = load_classes_from_yaml(yaml_path)
        log.info("data.yaml'dan okunan sınıflar: %s", class_names)
    else:
        log.warning(
            "data.yaml bulunamadı ve --source-classes verilmedi. "
            "Sadece string isim eşleşmesi yapılacak (numeric ID'ler atlanır)."
        )

    log.info("\nSınıf eşleşme tablosu:")
    id_map = build_id_map(class_names) if class_names else {}

    log.info("\nLabel dosyaları dönüştürülüyor...")
    remap_dataset(dataset_dir, id_map, in_place=True)

    if args.update_yaml:
        update_yaml(yaml_path)

    log.info("=" * 55)
    log.info("Tamamlandı! Eğitim için: python train.py --device cpu")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
