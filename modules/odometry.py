"""
Görev 2: Görsel Odometri (Optik Akış) ve Telemetri Entegrasyonu.

data/telemetri.csv dosyasından konum okur; health=1 iken GPS/konum verisi
doğrudan kullanılır ve sistem kalibre edilir. health=0 olduğunda sunucu
beslemesi kesildiği için alt-görüş kamerası optik akışı ile konum kümülatif
olarak tahmin edilir.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "telemetri.csv"
)
ALTITUDE_SCALE_FACTOR = 0.0015
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
MAX_FEATURE_CORNERS = 200
MIN_FEATURE_DISTANCE = 12.0
MIN_VALID_TRACKS = 8


class GokboruOdometry:
    """
    Telemetri + görsel odometri birleşik konum tahmin modülü.

    health=1: CSV'den (x, y, z) doğrudan alınır.
    health=0: Son bilinen konum referans alınarak optik akış ile kümülatif güncellenir.
    """

    def __init__(
        self,
        telemetry_csv_path: Optional[str] = None,
        altitude_scale_factor: float = ALTITUDE_SCALE_FACTOR,
    ) -> None:
        self.altitude_scale_factor = altitude_scale_factor

        self._telemetry_records: List[Dict[str, float]] = []
        self._telemetry_by_frame: Dict[int, Dict[str, float]] = {}

        self._position_x = 0.0
        self._position_y = 0.0
        self._position_z = 0.0

        self._health_status = 1
        self._visual_odometry_active = False
        self._calibrated = False

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None

        self._last_telemetry_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        csv_path = telemetry_csv_path or str(DEFAULT_TELEMETRY_PATH)
        self.load_telemetry(csv_path)

    def load_telemetry(self, csv_path: str) -> bool:
        """
        data/telemetri.csv dosyasını yükler ve indeksler.

        Beklenen sütunlar (esnek eşleme):
            frame_id, x, y, z, health

        Args:
            csv_path: Telemetri CSV dosya yolu.

        Returns:
            Yükleme başarılıysa True.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("Telemetri dosyası bulunamadı: %s", path)
            self._telemetry_records = []
            self._telemetry_by_frame = {}
            return False

        records: List[Dict[str, float]] = []
        frame_index: Dict[int, Dict[str, float]] = {}

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    logger.warning("Telemetri CSV başlık satırı okunamadı.")
                    return False

                column_map = self._build_column_map(reader.fieldnames)

                for row_idx, row in enumerate(reader):
                    try:
                        frame_id = int(
                            self._parse_float(
                                row,
                                column_map.get("frame_id"),
                                default=float(row_idx),
                            )
                        )
                        record = {
                            "frame_id": float(frame_id),
                            "x": self._parse_float(row, column_map.get("x"), 0.0),
                            "y": self._parse_float(row, column_map.get("y"), 0.0),
                            "z": self._parse_float(row, column_map.get("z"), 10.0),
                            "health": self._parse_float(
                                row, column_map.get("health"), 1.0
                            ),
                        }
                        records.append(record)
                        frame_index[frame_id] = record
                    except (ValueError, TypeError) as row_exc:
                        logger.debug(
                            "Telemetri satırı atlandı (satır %d): %s", row_idx, row_exc
                        )

            self._telemetry_records = records
            self._telemetry_by_frame = frame_index
            logger.info("%d telemetri kaydı yüklendi: %s", len(records), path)
            return len(records) > 0

        except Exception as exc:
            logger.error("Telemetri okuma hatası: %s", exc)
            self._telemetry_records = []
            self._telemetry_by_frame = {}
            return False

    def _build_column_map(self, fieldnames: List[str]) -> Dict[str, str]:
        normalized = {name.strip().lower(): name for name in fieldnames}

        def pick(candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                if candidate in normalized:
                    return normalized[candidate]
            return None

        mapping = {
            "frame_id": pick(
                ["frame_id", "frame", "kare", "kare_id", "index", "id"]
            ),
            "x": pick(["x", "pos_x", "position_x", "longitude", "lon"]),
            "y": pick(["y", "pos_y", "position_y", "latitude", "lat"]),
            "z": pick(["z", "pos_z", "altitude", "height", "yukseklik", "yükseklik"]),
            "health": pick(
                ["health", "saglik", "sağlık", "health_status", "saglik_bilgisi"]
            ),
        }

        return {key: value for key, value in mapping.items() if value is not None}

    def _parse_float(
        self,
        row: Dict[str, str],
        column_name: Optional[str],
        default: float = 0.0,
    ) -> float:
        if column_name is None or column_name not in row:
            return float(default)

        raw = row[column_name]
        if raw is None or str(raw).strip() == "":
            return float(default)

        return float(str(raw).replace(",", ".").strip())

    def _get_telemetry_record(self, frame_id: int) -> Optional[Dict[str, float]]:
        if frame_id in self._telemetry_by_frame:
            return self._telemetry_by_frame[frame_id]

        if 0 <= frame_id < len(self._telemetry_records):
            return self._telemetry_records[frame_id]

        if self._telemetry_records:
            return self._telemetry_records[-1]

        return None

    def _parse_health_value(self, record: Optional[Dict[str, float]]) -> int:
        if record is None:
            return 0
        health_raw = record.get("health", 0.0)
        return 1 if float(health_raw) >= 0.5 else 0

    def _apply_telemetry_position(self, record: Dict[str, float]) -> None:
        self._position_x = float(record["x"])
        self._position_y = float(record["y"])
        self._position_z = float(record["z"])
        self._last_telemetry_position = (
            self._position_x,
            self._position_y,
            self._position_z,
        )
        self._calibrated = True

    def _detect_feature_points(self, gray: np.ndarray) -> Optional[np.ndarray]:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_FEATURE_CORNERS,
            qualityLevel=0.01,
            minDistance=MIN_FEATURE_DISTANCE,
            blockSize=7,
            useHarrisDetector=False,
        )
        if points is None or len(points) < MIN_VALID_TRACKS:
            return None
        return points

    def _compute_pixel_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[float, float]:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA,
        )

        if next_points is None or status is None:
            return (0.0, 0.0)

        valid_mask = status.reshape(-1) == 1
        if int(np.count_nonzero(valid_mask)) < MIN_VALID_TRACKS:
            return (0.0, 0.0)

        displacement = next_points[valid_mask] - prev_points[valid_mask]
        dx = float(np.median(displacement[:, 0, 0]))
        dy = float(np.median(displacement[:, 0, 1]))
        return (dx, dy)

    def _apply_visual_odometry(self, frame: np.ndarray) -> None:
        """Alt-görüş kamerası piksel kaymasını optik akış ile konuma çevirir."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._prev_points = self._detect_feature_points(gray)
            return

        if self._prev_points is None or len(self._prev_points) < MIN_VALID_TRACKS:
            self._prev_points = self._detect_feature_points(self._prev_gray)

        dx_px, dy_px = (0.0, 0.0)
        if self._prev_points is not None:
            dx_px, dy_px = self._compute_pixel_flow(
                self._prev_gray, gray, self._prev_points
            )

        altitude = max(self._position_z, 1.0)
        scale = self.altitude_scale_factor * altitude

        self._position_x += -dx_px * scale
        self._position_y += -dy_px * scale

        self._prev_gray = gray.copy()
        self._prev_points = self._detect_feature_points(gray)
        if self._prev_points is None:
            self._prev_points = self._detect_feature_points(self._prev_gray)

    def _transition_to_visual_odometry(self, frame_id: int) -> None:
        """health=0 geçişinde son bilinen telemetri konumunu referans alır."""
        if not self._visual_odometry_active:
            self._visual_odometry_active = True
            record = self._get_telemetry_record(frame_id)
            if record is not None:
                self._apply_telemetry_position(record)
            elif self._calibrated:
                self._position_x, self._position_y, self._position_z = (
                    self._last_telemetry_position
                )
            logger.info(
                "Görsel odometri devreye girdi (kare %d). Referans: (%.4f, %.4f, %.4f)",
                frame_id,
                self._position_x,
                self._position_y,
                self._position_z,
            )

        self._prev_gray = None
        self._prev_points = None

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, float]:
        """
        Kare için konum tahmini üretir.

        Args:
            frame: BGR formatında alt-görüş kamera karesi.
            frame_id: Kare indeksi.

        Returns:
            govev_2_position sözlüğü {"x": float, "y": float, "z": float}.
        """
        record = self._get_telemetry_record(frame_id)
        self._health_status = self._parse_health_value(record)

        if self._health_status == 1:
            self._visual_odometry_active = False
            self._prev_gray = None
            self._prev_points = None

            if record is not None:
                self._apply_telemetry_position(record)
        else:
            self._transition_to_visual_odometry(frame_id)
            if frame is not None and frame.size > 0:
                try:
                    self._apply_visual_odometry(frame)
                except Exception as exc:
                    logger.warning(
                        "Optik akış hatası (kare %d): %s", frame_id, exc
                    )

        return {
            "x": round(float(self._position_x), 6),
            "y": round(float(self._position_y), 6),
            "z": round(float(self._position_z), 6),
        }

    @property
    def health_status(self) -> int:
        return self._health_status

    @property
    def is_visual_odometry_active(self) -> bool:
        return self._visual_odometry_active

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def telemetry_loaded(self) -> bool:
        return len(self._telemetry_records) > 0

    def reset(self) -> None:
        """Odometri durumunu sıfırlar."""
        self._position_x = 0.0
        self._position_y = 0.0
        self._position_z = 0.0
        self._health_status = 1
        self._visual_odometry_active = False
        self._calibrated = False
        self._prev_gray = None
        self._prev_points = None
        self._last_telemetry_position = (0.0, 0.0, 0.0)
