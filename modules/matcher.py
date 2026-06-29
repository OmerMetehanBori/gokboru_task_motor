"""
Görev 3: Çoklu Ölçekli Şablon Eşleme (Multi-Scale Template Matching).

data/referans_obje.png şablonunu akan video karesinde %60–%120 ölçek aralığında
tarar; cv2.TM_CCOEFF_NORMED ile en yüksek benzerlik skorunu döndürür.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "referans_obje.png"
)
MIN_SCALE = 0.60
MAX_SCALE = 1.20
SCALE_STEP = 0.05
MATCH_METHOD = cv2.TM_CCOEFF_NORMED
DEFAULT_MATCH_THRESHOLD = 0.55
MIN_TEMPLATE_DIM = 8


class GokboruMatcher:
    """
    Referans nesne şablonunu çoklu ölçekte arayan eşleştirme modülü.
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        min_scale: float = MIN_SCALE,
        max_scale: float = MAX_SCALE,
        scale_step: float = SCALE_STEP,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step = scale_step
        self.match_threshold = match_threshold
        self.match_method = MATCH_METHOD

        self._template_gray: Optional[np.ndarray] = None
        self._template_width: int = 0
        self._template_height: int = 0

        resolved_path = template_path or str(DEFAULT_TEMPLATE_PATH)
        self.load_template(resolved_path)

    def load_template(self, template_path: str) -> bool:
        """
        Referans şablon görselini yükler.

        Args:
            template_path: Şablon dosya yolu (varsayılan: data/referans_obje.png).

        Returns:
            Yükleme başarılıysa True.
        """
        path = Path(template_path)
        if not path.exists():
            logger.warning("Şablon dosyası bulunamadı: %s", path)
            self._template_gray = None
            self._template_width = 0
            self._template_height = 0
            return False

        try:
            template_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template_bgr is None or template_bgr.size == 0:
                logger.warning("Şablon görüntüsü okunamadı: %s", path)
                self._template_gray = None
                return False

            self._template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            self._template_height, self._template_width = self._template_gray.shape[:2]
            logger.info(
                "Referans şablon yüklendi: %s (%dx%d)",
                path.name,
                self._template_width,
                self._template_height,
            )
            return True

        except Exception as exc:
            logger.error("Şablon yükleme hatası: %s", exc)
            self._template_gray = None
            self._template_width = 0
            self._template_height = 0
            return False

    def _generate_scales(self) -> List[float]:
        scales: List[float] = []
        current = self.min_scale
        while current <= self.max_scale + 1e-9:
            scales.append(round(current, 4))
            current += self.scale_step
        return scales if scales else [1.0]

    def _match_at_scale(
        self,
        frame_gray: np.ndarray,
        scale: float,
    ) -> Tuple[float, int, int, int, int]:
        """
        Belirli bir ölçekte şablon eşleştirmesi yapar.

        Returns:
            (score, top_left_x, top_left_y, width, height)
        """
        if self._template_gray is None:
            return (0.0, 0, 0, 0, 0)

        scaled_w = max(MIN_TEMPLATE_DIM, int(self._template_width * scale))
        scaled_h = max(MIN_TEMPLATE_DIM, int(self._template_height * scale))

        frame_h, frame_w = frame_gray.shape[:2]
        if scaled_w >= frame_w or scaled_h >= frame_h:
            return (0.0, 0, 0, 0, 0)

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled_template = cv2.resize(
            self._template_gray,
            (scaled_w, scaled_h),
            interpolation=interpolation,
        )

        try:
            result_map = cv2.matchTemplate(
                frame_gray, scaled_template, self.match_method
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result_map)
        except cv2.error as exc:
            logger.debug("Ölçek %.2f eşleştirme hatası: %s", scale, exc)
            return (0.0, 0, 0, 0, 0)

        top_left_x = int(max_loc[0])
        top_left_y = int(max_loc[1])
        return (float(max_val), top_left_x, top_left_y, scaled_w, scaled_h)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "top_left_x": 0,
            "top_left_y": 0,
            "width": 0,
            "height": 0,
            "score": 0.0,
            "found": False,
        }

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Video karesinde referans şablonu çoklu ölçekte arar.

        Args:
            frame: BGR formatında giriş karesi.

        Returns:
            govev_3_match sözlüğü.
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        if self._template_gray is None:
            return self._empty_result()

        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scales = self._generate_scales()

            best_score = 0.0
            best_x = 0
            best_y = 0
            best_w = 0
            best_h = 0

            for scale in scales:
                score, x, y, w, h = self._match_at_scale(frame_gray, scale)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_y = y
                    best_w = w
                    best_h = h

            found = best_score >= self.match_threshold

            return {
                "top_left_x": int(best_x),
                "top_left_y": int(best_y),
                "width": int(best_w),
                "height": int(best_h),
                "score": round(float(best_score), 6),
                "found": bool(found),
            }

        except Exception as exc:
            logger.error("Matcher işlem hatası: %s", exc, exc_info=True)
            return self._empty_result()

    @property
    def template_loaded(self) -> bool:
        return self._template_gray is not None

    @property
    def template_size(self) -> Tuple[int, int]:
        return (self._template_width, self._template_height)

    def reset(self) -> None:
        """Matcher durumunu sıfırlar (şablon korunur)."""
        pass
