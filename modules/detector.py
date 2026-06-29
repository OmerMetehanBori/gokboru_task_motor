"""
Görev 1: YOLOv8 Nesne Tespiti + Merkez Noktası Takip (Centroid Tracking).

Eğitilmiş drone modeli (models/best.pt) ile araç, insan ve iniş alanı tespiti yapar.
Araçların gerçek hareketini rüzgar sallantısından ayırmak için 3.5 px tolerans uygular.
İniş alanı doluluk ve kadraj kontrolü geometrik IoU ile yapılır.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASS_ID_MAP = {
    0: "Tasit",
    1: "Insan",
}

VEHICLE_CLASS = "Tasit"
PERSON_CLASS = "Insan"
LANDING_CLASSES: frozenset[str] = frozenset()
OBSTACLE_CLASSES = frozenset({"Tasit", "Insan"})

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
CENTROID_HISTORY_LEN = 5
SWAY_TOLERANCE_PX = 3.5
STABLE_MOVING_MIN_TRANSITIONS = 3
IOU_TRACK_MATCH_THRESHOLD = 0.30
IOU_OBSTACLE_THRESHOLD = 0.05
FRAME_EDGE_TOLERANCE_PX = 2


@dataclass
class TrackedObject:
    """Merkez noktası takibi için dahili nesne durumu."""

    track_id: int
    class_name: str
    bbox: List[int]
    centroid: Tuple[float, float]
    centroid_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=CENTROID_HISTORY_LEN)
    )
    is_moving: int = 0


def _bbox_xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> List[int]:
    return [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]


def _centroid_from_bbox(bbox: List[int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _iou_xywh(box_a: List[int], box_b: List[int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _bbox_touches_frame_edge(
    bbox: List[int], frame_w: int, frame_h: int, tolerance: int = FRAME_EDGE_TOLERANCE_PX
) -> bool:
    x, y, w, h = bbox
    return (
        x <= tolerance
        or y <= tolerance
        or (x + w) >= (frame_w - tolerance)
        or (y + h) >= (frame_h - tolerance)
    )


def _normalize_class_name(raw_name: str, class_id: int = -1) -> str:
    if class_id in CLASS_ID_MAP:
        return CLASS_ID_MAP[class_id]

    name = raw_name.strip()
    upper = name.upper()
    lower = name.lower()

    if upper in {"TASIT", "ARAC", "VEHICLE", "CAR", "TRUCK", "BUS", "VAN"}:
        return VEHICLE_CLASS
    if upper in {"INSAN", "PERSON", "HUMAN"}:
        return PERSON_CLASS
    if upper == "UAP":
        return "UAP"
    if upper == "UAI":
        return "UAI"

    if lower in {"tasit", "arac", "vehicle", "car", "truck", "bus", "van"}:
        return VEHICLE_CLASS
    if lower in {"insan", "person", "human"}:
        return PERSON_CLASS

    return name if name else "unknown"


class GokboruDetector:
    """
    YOLOv8 tabanlı nesne tespiti ve merkez noktası takip modülü.

    Araçlara benzersiz takip ID'si atar, son 5 karenin merkez koordinatlarını
    saklar ve rüzgar sallantısını 3.5 px toleransla filtreleyerek is_moving üretir.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        sway_tolerance: float = SWAY_TOLERANCE_PX,
        iou_track_threshold: float = IOU_TRACK_MATCH_THRESHOLD,
        iou_obstacle_threshold: float = IOU_OBSTACLE_THRESHOLD,
        inference_imgsz: int = 1280,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.sway_tolerance = sway_tolerance
        self.iou_track_threshold = iou_track_threshold
        self.iou_obstacle_threshold = iou_obstacle_threshold
        self.inference_imgsz = inference_imgsz

        self._yolo_model: Any = None
        self._model_loaded = False
        self._model_path: Optional[Path] = None

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 1

        self._load_model(model_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            candidate = Path(model_path)
            if candidate.exists():
                return candidate.resolve()

        if DEFAULT_MODEL_PATH.exists():
            return DEFAULT_MODEL_PATH.resolve()

        return DEFAULT_MODEL_PATH

    def _load_model(self, model_path: Optional[str]) -> None:
        resolved = self._resolve_model_path(model_path)
        self._model_path = resolved

        if not resolved.exists():
            logger.error(
                "YOLO model dosyası bulunamadı: %s — Önce train.py ile eğitim yapın.",
                resolved,
            )
            self._model_loaded = False
            return

        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(str(resolved))
            self._model_loaded = True
            logger.info("Gökbörü drone modeli yüklendi: %s", resolved)
        except ImportError:
            logger.error(
                "ultralytics paketi yüklü değil. Kurulum: pip install ultralytics"
            )
            self._model_loaded = False
        except Exception as exc:
            logger.error("Model yükleme hatası: %s", exc)
            self._model_loaded = False

    def _run_yolo_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if not self._model_loaded or self._yolo_model is None:
            return detections

        try:
            results = self._yolo_model.predict(
                frame,
                imgsz=self.inference_imgsz,
                conf=self.confidence_threshold,
                verbose=False,
            )

            if not results:
                return detections

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return detections

            names = result.names if hasattr(result, "names") and result.names else {}

            for box in boxes:
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = _bbox_xyxy_to_xywh(int(x1), int(y1), int(x2), int(y2))

                class_id = int(box.cls[0]) if box.cls is not None else -1
                raw_name = str(names.get(class_id, "")) if names else ""
                class_name = _normalize_class_name(raw_name, class_id)

                detections.append(
                    {
                        "class_name": class_name,
                        "bbox": bbox,
                        "confidence": confidence,
                    }
                )

        except Exception as exc:
            logger.warning("YOLO çıkarım hatası: %s", exc)

        return detections

    def _match_detections_to_tracks(
        self, detections: List[Dict[str, Any]]
    ) -> None:
        """IoU tabanlı merkez noktası takip: tespitleri mevcut izlere eşleştirir."""
        unmatched = list(detections)
        updated_tracks: Dict[int, TrackedObject] = {}
        matched_track_ids: set[int] = set()

        for track_id, track in self._tracks.items():
            best_idx = -1
            best_iou = 0.0

            for idx, det in enumerate(unmatched):
                if det["class_name"] != track.class_name:
                    continue
                iou = _iou_xywh(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= self.iou_track_threshold:
                det = unmatched.pop(best_idx)
                centroid = _centroid_from_bbox(det["bbox"])
                history = deque(track.centroid_history, maxlen=CENTROID_HISTORY_LEN)
                history.append(centroid)

                updated_tracks[track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=det["class_name"],
                    bbox=det["bbox"],
                    centroid=centroid,
                    centroid_history=history,
                    is_moving=track.is_moving,
                )
                matched_track_ids.add(track_id)

        for det in unmatched:
            track_id = self._next_track_id
            self._next_track_id += 1
            centroid = _centroid_from_bbox(det["bbox"])
            history: Deque[Tuple[float, float]] = deque(maxlen=CENTROID_HISTORY_LEN)
            history.append(centroid)

            updated_tracks[track_id] = TrackedObject(
                track_id=track_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                centroid=centroid,
                centroid_history=history,
            )

        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids and track_id not in updated_tracks:
                continue

        self._tracks = updated_tracks

    def _compute_vehicle_motion(self, track: TrackedObject) -> int:
        """
        Son 5 karenin merkez geçmişine göre araç hareket durumunu hesaplar.

        Rüzgar kaynaklı sallantı SWAY_TOLERANCE_PX (3.5 px) altında kalırsa sabit sayılır.
        Toleransı aşan kararlı yer değiştirme varsa is_moving=1 döner.
        """
        if track.class_name != VEHICLE_CLASS:
            return 0

        history = list(track.centroid_history)
        if len(history) < CENTROID_HISTORY_LEN:
            return 0

        moving_transitions = 0
        for idx in range(1, len(history)):
            dx = history[idx][0] - history[idx - 1][0]
            dy = history[idx][1] - history[idx - 1][1]
            step_distance = float(np.hypot(dx, dy))
            if step_distance > self.sway_tolerance:
                moving_transitions += 1

        first = history[0]
        last = history[-1]
        net_dx = last[0] - first[0]
        net_dy = last[1] - first[1]
        net_displacement = float(np.hypot(net_dx, net_dy))

        is_stable_moving = (
            moving_transitions >= STABLE_MOVING_MIN_TRANSITIONS
            and net_displacement > self.sway_tolerance
        )

        return 1 if is_stable_moving else 0

    def _update_motion_states(self) -> None:
        for track in self._tracks.values():
            track.is_moving = self._compute_vehicle_motion(track)

    def _evaluate_landing_area(
        self,
        landing_bbox: List[int],
        all_tracks: List[TrackedObject],
        frame_w: int,
        frame_h: int,
    ) -> int:
        """
        İniş alanı durumunu değerlendirir.

        Returns:
            1 — Alan boş ve kadraja tam sığıyor (kırpılmamış).
            0 — Alan dolu, engelli veya kadraj dışına taşıyor.
        """
        if _bbox_touches_frame_edge(landing_bbox, frame_w, frame_h):
            return 0

        for track in all_tracks:
            if track.bbox == landing_bbox:
                continue

            if track.class_name in LANDING_CLASSES:
                overlap = _iou_xywh(landing_bbox, track.bbox)
                if overlap > self.iou_obstacle_threshold:
                    return 0
                continue

            if track.class_name in OBSTACLE_CLASSES:
                overlap = _iou_xywh(landing_bbox, track.bbox)
                if overlap > self.iou_obstacle_threshold:
                    return 0

        return 1

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tek video karesinde tespit, takip ve iniş alanı analizi yapar.

        Args:
            frame: BGR formatında giriş karesi.

        Returns:
            Yarışma formatında govev_1_objects listesi.
        """
        if frame is None or frame.size == 0:
            return []

        if not self._model_loaded:
            logger.warning("Model yüklü değil; boş tespit listesi döndürülüyor.")
            return []

        try:
            detections = self._run_yolo_inference(frame)
            self._match_detections_to_tracks(detections)
            self._update_motion_states()

            frame_h, frame_w = frame.shape[:2]
            track_list = list(self._tracks.values())
            output: List[Dict[str, Any]] = []

            for track in track_list:
                if track.class_name in LANDING_CLASSES:
                    landing_status = self._evaluate_landing_area(
                        track.bbox, track_list, frame_w, frame_h
                    )
                else:
                    landing_status = 0

                is_moving = track.is_moving if track.class_name == VEHICLE_CLASS else 0

                output.append(
                    {
                        "class_name": track.class_name,
                        "bbox": [int(v) for v in track.bbox],
                        "is_moving": int(is_moving),
                        "landing_status": int(landing_status),
                    }
                )

            return output

        except Exception as exc:
            logger.error("Detector işlem hatası: %s", exc, exc_info=True)
            return []

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_path(self) -> Optional[str]:
        return str(self._model_path) if self._model_path else None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Takip durumunu ve izleme hafızasını sıfırlar."""
        self._tracks.clear()
        self._next_track_id = 1
