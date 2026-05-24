"""VA (valence / arousal) batch runner — wraps va_mbf.onnx.

Usage:
    from stateguard.models.va_runner import VARunner
    va = VARunner('weights/va_mbf.onnx')
    out = va.predict(face_rgb_uint8_batch)   # (N, 224x224 or larger, RGB) -> (N, 2)
"""
import numpy as np
import onnxruntime as ort
import cv2


INPUT_NAME = 'frames'
IMG_SIZE = 112
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)


class VARunner:
    """Static-image VA prediction. Built for sparse, gated invocation
    (e.g. 4 keyframes per 15s window)."""

    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=['CPUExecutionProvider']
        )

    @staticmethod
    def preprocess(frames_rgb_uint8: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 RGB -> (N, 3, 112, 112) float32 normalized."""
        x = np.asarray(frames_rgb_uint8)
        if x.ndim == 3:
            x = x[None]
        N = x.shape[0]
        out = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for i in range(N):
            img = x[i]
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            out[i] = img.astype(np.float32) / 255.0
        # NHWC -> NCHW, normalize
        out = out.transpose(0, 3, 1, 2)
        out = (out - MEAN) / STD
        return out

    def predict(self, frames_rgb_uint8: np.ndarray) -> np.ndarray:
        """Returns (N, 2) array of [valence, arousal] in [-2, +2]."""
        x = self.preprocess(frames_rgb_uint8)
        out = self.sess.run(None, {INPUT_NAME: x})[0]
        return np.asarray(out, dtype=np.float32)
