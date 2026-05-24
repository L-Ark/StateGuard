"""Fatigue (binary drowsy/normal) runner — wraps fatigue.onnx.

Usage:
    from stateguard.models.fatigue_runner import FatigueRunner
    fat = FatigueRunner('weights/fatigue.onnx')
    p = fat.predict(face_rgb_uint8_batch)   # (N, H, W, 3) RGB uint8 -> (N,) P(fatigue)

Model I/O:
    input  'frames' (N, 3, 112, 112) float32, mean=std=0.5 normalized
    output 'logits' (N, 2) float32  [normal_logit, fatigue_logit]
           'prob'   (N,)   float32  P(fatigue) (softmax)

Trained on Kaggle drowsy_detection + Kaggle fatigue (binary).
Validation acc 0.864 (in-domain), 0.959 (cross-dataset drowsy/test).
"""
import numpy as np
import onnxruntime as ort
import cv2


INPUT_NAME = 'frames'
IMG_SIZE = 112
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)


class FatigueRunner:
    """Static-image fatigue prediction.

    Designed for sparse, gated invocation alongside VA on the same keyframes,
    so 'fatigue' adds <15 ms per 15 s VA window (mean of 4 frames).
    """

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
        out = out.transpose(0, 3, 1, 2)
        out = (out - MEAN) / STD
        return out

    def predict(self, frames_rgb_uint8: np.ndarray) -> np.ndarray:
        """Returns (N,) array of P(fatigue) in [0, 1]."""
        x = self.preprocess(frames_rgb_uint8)
        logits, prob = self.sess.run(None, {INPUT_NAME: x})
        return np.asarray(prob, dtype=np.float32)

    def predict_logits(self, frames_rgb_uint8: np.ndarray) -> np.ndarray:
        """Returns (N, 2) logits [normal, fatigue]."""
        x = self.preprocess(frames_rgb_uint8)
        logits, prob = self.sess.run(None, {INPUT_NAME: x})
        return np.asarray(logits, dtype=np.float32)
