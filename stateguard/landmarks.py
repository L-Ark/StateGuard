"""Simple Mediapipe FaceMesh wrapper used by demos.

Supports either legacy `mediapipe.solutions.face_mesh` or the newer
`mediapipe.tasks` FaceLandmarker. The tasks-based API requires a
local `.task` model file, which this module will download on demand.
"""
from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import urllib.request
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
        model_path: Optional[str] = None,
    ) -> None:
        if mp is None:
            raise RuntimeError('mediapipe is not installed. Install with "pip install mediapipe"')
        self._mp = mp
        self._use_solutions = hasattr(mp, 'solutions')
        self._last_landmarks = None
        self._last_results = None

        if self._use_solutions:
            self._drawing = mp.solutions.drawing_utils
            self._styles = mp.solutions.drawing_styles
            self._fm = mp.solutions.face_mesh
            self._mesh = self._fm.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence),
            )
        else:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            self._vision = vision
            model_file = self._resolve_model_path(model_path)
            self._ensure_model_file(model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_file)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)
        # default to repo weights folder
        return Path(__file__).resolve().parent / 'weights' / 'face_landmarker.task'

    def _ensure_model_file(self, model_path: Path) -> None:
        if model_path.exists():
            return
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task'
        )
        urllib.request.urlretrieve(url, str(model_path))

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
        if self._use_solutions:
            results = self._mesh.process(im)
            self._last_results = results
            if not results or not results.multi_face_landmarks:
                self._last_landmarks = None
                return None
            lm = results.multi_face_landmarks[0]
            out = []
            for p in lm.landmark:
                out.append({'x': float(p.x), 'y': float(p.y), 'z': float(p.z)})
            self._last_landmarks = out
            return out

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=im)
        result = self._landmarker.detect(mp_image)
        if not result or not result.face_landmarks:
            self._last_landmarks = None
            return None
        lm = result.face_landmarks[0]
        out = []
        for p in lm:
            out.append({'x': float(p.x), 'y': float(p.y), 'z': float(p.z)})
        self._last_landmarks = out
        return out

    def draw_landmarks(self, image_bgr: np.ndarray, landmarks) -> None:
        """Draw the landmarks on a BGR image in-place (for display)."""
        if self._use_solutions:
            # Draw using the stored mediapipe results if available.
            if not self._last_results:
                return
            results = self._last_results
            if not results.multi_face_landmarks:
                return
            for landmarks in results.multi_face_landmarks:
                self._drawing.draw_landmarks(
                    image_bgr,
                    landmarks,
                    self._fm.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._styles.get_default_face_mesh_tesselation_style(),
                )
                self._drawing.draw_landmarks(
                    image_bgr,
                    landmarks,
                    self._fm.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._styles.get_default_face_mesh_contours_style(),
                )
            return

        if self._last_landmarks is None:
            return
        try:
            import cv2
        except Exception:
            return
        h, w = image_bgr.shape[:2]
        for p in self._last_landmarks:
            x = int(p['x'] * w)
            y = int(p['y'] * h)
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 255), -1)

    def close(self) -> None:
        try:
            if self._use_solutions:
                self._mesh.close()
            else:
                self._landmarker.close()
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
            if self._use_solutions:
                self._mesh.process(dummy)
            else:
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=dummy)
                self._landmarker.detect(mp_image)
        except Exception:
            # ignore errors; purpose is warm-up
            pass
