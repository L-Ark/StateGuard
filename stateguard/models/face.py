"""Lightweight face cropper (MediaPipe → fallback Haar).

For the deployed app we want: small, fast, always-available. Haar on a
downsized frame keeps CPU usage low and runs everywhere; MediaPipe gives
tighter, more stable boxes when available.
"""
from typing import Optional, Tuple

import cv2
import numpy as np


class FaceCropper:
    def __init__(
        self,
        prefer_mediapipe: bool = True,
        detect_every_n: int = 5,
        smooth: float = 0.4,
        margin: float = 0.15,
    ) -> None:
        self.detect_every_n = int(detect_every_n)
        self.smooth = float(smooth)
        self.margin = float(margin)
        self._n = 0
        self._box: Optional[np.ndarray] = None  # [x1, y1, x2, y2]

        self._mp = None
        if prefer_mediapipe:
            try:
                import mediapipe as mp  # type: ignore
                self._mp = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception:
                self._mp = None
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def reset(self) -> None:
        self._n = 0
        self._box = None

    def _detect_mp(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        res = self._mp.process(rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w)); y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width) * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _detect_haar(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        scale = 320.0 / max(h, w)
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return np.array([fx / scale, fy / scale, (fx + fw) / scale, (fy + fh) / scale],
                        dtype=np.float32)

    def crop(self, frame_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Returns (face_rgb, box) or (None, None) if nothing yet."""
        if self._n % self.detect_every_n == 0:
            det = None
            if self._mp is not None:
                det = self._detect_mp(frame_rgb)
            if det is None:
                det = self._detect_haar(frame_rgb)
            if det is not None:
                if self._box is None:
                    self._box = det
                else:
                    self._box = self.smooth * det + (1.0 - self.smooth) * self._box
        self._n += 1

        if self._box is None:
            return None, None

        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = self._box
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - bw * self.margin))
        y1 = int(max(0, y1 - bh * self.margin))
        x2 = int(min(w, x2 + bw * self.margin))
        y2 = int(min(h, y2 + bh * self.margin))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None, None
        return frame_rgb[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
