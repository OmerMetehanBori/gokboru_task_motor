"""
Gökbörü Çevrim Dışı Görev Motoru — Ana Pipeline ve Video Akış Yöneticisi.

TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması için offline görev işleme
motoru. Video karelerini sırayla işler; YOLOv8+takip detektörü, görsel
odometri ve şablon eşleştiriciyi tetikleyerek gokboru_session_output.json üretir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from modules.detector import GokboruDetector
from modules.matcher import GokboruMatcher
from modules.odometry import GokboruOdometry
from utils.json_validator import (
    create_empty_frame_output,
    get_validation_errors,
    validate_frame_output,
    validate_session_output,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GokboruTaskEngine")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "telemetri.csv"
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "data" / "referans_obje.png"
DEFAULT_OUTPUT_FILE = "gokboru_session_output.json"
DEFAULT_FPS = 7.5


class GokboruTaskEngine:
    """
    Tüm görev modüllerini birleştiren ana çevrim dışı işlem motoru.

    Pipeline sırası:
        1. GokboruDetector  — YOLOv8 + merkez noktası takip
        2. GokboruOdometry  — telemetri + optik akış odometri
        3. GokboruMatcher   — çoklu ölçekli şablon eşleme
        4. json_validator   — çıktı şema doğrulama
    """

    def __init__(
        self,
        video_path: str,
        telemetry_csv_path: Optional[str] = None,
        template_path: Optional[str] = None,
        model_path: Optional[str] = None,
        output_path: str = DEFAULT_OUTPUT_FILE,
        target_fps: float = DEFAULT_FPS,
        max_frames: Optional[int] = None,
        display_preview: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.display_preview = display_preview

        resolved_model = model_path or str(DEFAULT_MODEL_PATH)
        resolved_telemetry = telemetry_csv_path or str(DEFAULT_TELEMETRY_PATH)
        resolved_template = template_path or str(DEFAULT_TEMPLATE_PATH)

        self.detector = GokboruDetector(model_path=resolved_model)
        self.odometry = GokboruOdometry(telemetry_csv_path=resolved_telemetry)
        self.matcher = GokboruMatcher(template_path=resolved_template)

        self.session_results: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "processed_frames": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "failed_modules": 0,
        }

    def _validate_inputs(self) -> bool:
        if not self.video_path.exists():
            logger.error("Video dosyası bulunamadı: %s", self.video_path)
            return False

        if not self.detector.is_model_loaded:
            logger.warning(
                "YOLO modeli yüklenemedi (%s). "
                "Önce train.py ile model eğitin.",
                self.detector.model_path,
            )

        if not self.odometry.telemetry_loaded:
            logger.warning(
                "Telemetri dosyası yüklenemedi. "
                "Görsel odometri modu tüm karelerde aktif olabilir."
            )

        if not self.matcher.template_loaded:
            logger.warning(
                "Referans şablon yüklenemedi. "
                "Görev 3 eşleştirme skorları sıfır dönecektir."
            )

        return True

    def _compute_timestamp(self, frame_id: int, capture_fps: float) -> float:
        if capture_fps > 0:
            return round(frame_id / capture_fps, 6)
        return round(frame_id / self.target_fps, 6)

    def _process_single_frame(
        self,
        frame,
        frame_id: int,
        timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        frame_output = create_empty_frame_output(frame_id, timestamp)

        try:
            frame_output["govev_1_objects"] = self.detector.process_frame(frame)
        except Exception as exc:
            logger.warning("Detector hatası (kare %d): %s", frame_id, exc)
            self.stats["failed_modules"] += 1

        try:
            frame_output["govev_2_position"] = self.odometry.process_frame(
                frame, frame_id
            )
        except Exception as exc:
            logger.warning("Odometry hatası (kare %d): %s", frame_id, exc)
            self.stats["failed_modules"] += 1

        try:
            match_result = self.matcher.process_frame(frame)
            frame_output["govev_3_match"] = {
                "top_left_x": int(match_result["top_left_x"]),
                "top_left_y": int(match_result["top_left_y"]),
                "width": int(match_result["width"]),
                "height": int(match_result["height"]),
                "score": float(match_result["score"]),
                "found": bool(match_result.get("found", False)),
            }
        except Exception as exc:
            logger.warning("Matcher hatası (kare %d): %s", frame_id, exc)
            self.stats["failed_modules"] += 1

        if validate_frame_output(frame_output):
            return frame_output

        validation_errors = get_validation_errors(frame_output)
        logger.warning(
            "Kare %d şema doğrulamasından geçemedi: %s",
            frame_id,
            "; ".join(validation_errors),
        )
        self.stats["invalid_frames"] += 1
        return None

    def _build_session_payload(self) -> Dict[str, Any]:
        return {
            "session_metadata": {
                "video_source": str(self.video_path.resolve()),
                "output_file": str(self.output_path.resolve()),
                "target_fps": self.target_fps,
                "total_frames": len(self.session_results),
                "model_path": self.detector.model_path,
                "model_loaded": self.detector.is_model_loaded,
                "telemetry_loaded": self.odometry.telemetry_loaded,
                "visual_odometry_used": self.odometry.is_visual_odometry_active,
                "template_loaded": self.matcher.template_loaded,
                "active_tracks_at_end": self.detector.active_track_count,
                "statistics": self.stats,
            },
            "frames": self.session_results,
        }

    def _write_session_output(self) -> bool:
        session_payload = self._build_session_payload()

        is_valid, error_msg = validate_session_output(session_payload)
        if not is_valid:
            logger.error("Oturum çıktısı doğrulama hatası: %s", error_msg)

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("w", encoding="utf-8") as handle:
                json.dump(session_payload, handle, ensure_ascii=False, indent=4)
            logger.info("Oturum çıktısı yazıldı: %s", self.output_path.resolve())
            return True
        except Exception as exc:
            logger.error("JSON yazma hatası: %s", exc)
            return False

    def _draw_preview_overlay(
        self,
        frame,
        frame_id: int,
        frame_output: Optional[Dict[str, Any]],
    ):
        preview = frame.copy()

        cv2.putText(
            preview,
            f"Frame: {frame_id}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if frame_output is None:
            return preview

        for obj in frame_output.get("govev_1_objects", []):
            x, y, w, h = obj["bbox"]
            color = (0, 255, 255) if obj.get("is_moving") == 1 else (255, 128, 0)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            label = f"{obj['class_name']}"
            if obj["class_name"] == "Tasit":
                label += f" M={obj['is_moving']}"
            cv2.putText(
                preview,
                label,
                (x, max(y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        match = frame_output.get("govev_3_match", {})
        if match.get("found"):
            mx = match["top_left_x"]
            my = match["top_left_y"]
            mw = match["width"]
            mh = match["height"]
            cv2.rectangle(preview, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        pos = frame_output.get("govev_2_position", {})
        pos_text = f"Pos: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})"
        cv2.putText(
            preview,
            pos_text,
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 0),
            1,
        )

        return preview

    def run(self) -> int:
        """
        Video işleme döngüsünü başlatır.

        Returns:
            İşlem başarılıysa 0, aksi halde 1.
        """
        if not self._validate_inputs():
            return 1

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            logger.error("Video açılamadı: %s", self.video_path)
            return 1

        capture_fps = capture.get(cv2.CAP_PROP_FPS)
        if capture_fps <= 0 or capture_fps > 120:
            capture_fps = self.target_fps
            logger.info(
                "Video FPS okunamadı, hedef FPS kullanılıyor: %.2f", capture_fps
            )

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        frame_id = 0

        logger.info("=" * 60)
        logger.info("Gökbörü Görev Motoru Başlatıldı")
        logger.info("=" * 60)
        logger.info("Video      : %s", self.video_path)
        logger.info("Model      : %s (loaded=%s)", self.detector.model_path, self.detector.is_model_loaded)
        logger.info("Telemetri  : loaded=%s", self.odometry.telemetry_loaded)
        logger.info("Şablon     : loaded=%s", self.matcher.template_loaded)
        logger.info("Hedef FPS  : %.2f", self.target_fps)
        logger.info("Çıktı      : %s", self.output_path)
        logger.info("=" * 60)

        try:
            while True:
                if self.max_frames is not None and frame_id >= self.max_frames:
                    logger.info("Maksimum kare limitine ulaşıldı: %d", self.max_frames)
                    break

                loop_start = time.perf_counter()
                success, frame = capture.read()

                if not success or frame is None:
                    logger.info("Video akışı sona erdi (toplam kare: %d).", frame_id)
                    break

                timestamp = self._compute_timestamp(frame_id, capture_fps)
                frame_result = self._process_single_frame(frame, frame_id, timestamp)

                self.stats["processed_frames"] += 1
                if frame_result is not None:
                    self.session_results.append(frame_result)
                    self.stats["valid_frames"] += 1

                if self.display_preview:
                    preview = self._draw_preview_overlay(frame, frame_id, frame_result)
                    cv2.imshow("Gokboru Task Engine", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Kullanıcı tarafından döngü sonlandırıldı.")
                        break

                frame_id += 1

                if frame_interval > 0:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if frame_id > 0 and frame_id % 100 == 0:
                    logger.info(
                        "İşlenen: %d | Geçerli: %d | Geçersiz: %d | Modül hata: %d",
                        frame_id,
                        self.stats["valid_frames"],
                        self.stats["invalid_frames"],
                        self.stats["failed_modules"],
                    )

        except KeyboardInterrupt:
            logger.info("Klavye kesintisi alındı, oturum kaydediliyor...")
        except Exception as exc:
            logger.error("Beklenmeyen işlem hatası: %s", exc, exc_info=True)
            return 1
        finally:
            capture.release()
            if self.display_preview:
                cv2.destroyAllWindows()

        if not self._write_session_output():
            return 1

        logger.info(
            "İşlem tamamlandı. %d/%d kare geçerli çıktı üretti.",
            self.stats["valid_frames"],
            self.stats["processed_frames"],
        )
        return 0

    def reset(self) -> None:
        """Tüm modül durumlarını ve oturum verisini sıfırlar."""
        self.detector.reset()
        self.odometry.reset()
        self.matcher.reset()
        self.session_results.clear()
        self.stats = {
            "processed_frames": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "failed_modules": 0,
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gökbörü Çevrim Dışı Görev Motoru — TEKNOFEST 2026"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="İşlenecek video dosyası yolu",
    )
    parser.add_argument(
        "--telemetry",
        default=str(DEFAULT_TELEMETRY_PATH),
        help=f"Telemetri CSV dosyası (varsayılan: {DEFAULT_TELEMETRY_PATH})",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help=f"Referans şablon görseli (varsayılan: {DEFAULT_TEMPLATE_PATH})",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"YOLO model ağırlığı (varsayılan: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Çıktı JSON dosyası (varsayılan: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Hedef işleme FPS (varsayılan: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="İşlenecek maksimum kare sayısı",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="İşleme sırasında önizleme penceresi göster",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ayrıntılı log çıktısı",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    engine = GokboruTaskEngine(
        video_path=args.video,
        telemetry_csv_path=args.telemetry,
        template_path=args.template,
        model_path=args.model,
        output_path=args.output,
        target_fps=args.fps,
        max_frames=args.max_frames,
        display_preview=args.preview,
    )

    return engine.run()


if __name__ == "__main__":
    raise SystemExit(main())
