"""Simple Mediapipe FaceMesh wrapper used by demos.

This module provides a small runtime-friendly wrapper around
`mediapipe.solutions.face_mesh` that returns normalized landmarks
and offers a draw helper for visualization.

Note: `mediapipe` is an optional dependency and should be installed
via `pip install mediapipe`. The package ships its models; no separate
model files are required.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None  # type: ignore


class FaceMeshRunner:
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise RuntimeError('mediapipe is not installed. Install with "pip install mediapipe"')
        self._mp = mp
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._fm = mp.solutions.face_mesh
        self._mesh = self._fm.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def process(self, frame_rgb: np.ndarray) -> Optional[List[dict]]:
        """Process an RGB uint8 frame and return list of landmarks for the
        first detected face. Each landmark is a dict with keys `x`,`y`,`z`
        in normalized coordinates (x,y in [0,1], z relative).
        Returns None if no face detected.
        """
        if frame_rgb is None:
            return None
        # Ensure uint8 RGB
        im = np.asarray(frame_rgb)
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError('frame_rgb must be HxWx3 RGB uint8')
        results = self._mesh.process(im)
        if not results or not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        out = []
        for p in lm.landmark:
            out.append({'x': float(p.x), 'y': float(p.y), 'z': float(p.z)})
        return out

    def draw_landmarks(self, image_bgr: np.ndarray, landmarks) -> None:
        """Draw the landmarks on a BGR image in-place (for display)."""
        if landmarks is None:
            return
        # Mediapipe expects the full results object; we reuse drawing helpers
        # by reconstructing a simple wrapper. Instead, call the library draw
        # directly if we still have the last result available. For simplicity
        # expect callers to keep the original mediapipe results when drawing.
        raise NotImplementedError('Use mediapipe drawing helpers with the raw results from FaceMesh')

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass

    def ensure_model_loaded(self) -> None:
        """Force the FaceMesh model to initialize by processing a blank image.

        This can help pre-load assets so the first real frame doesn't pay the
        model startup cost.
        """
        import numpy as _np

        dummy = _np.zeros((256, 256, 3), dtype=_np.uint8)
        try:
            self._mesh.process(dummy)
        except Exception:
            # ignore errors; purpose is warm-up
            pass
