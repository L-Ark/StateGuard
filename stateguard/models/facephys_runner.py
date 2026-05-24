"""FacePhys streaming rPPG runner — wraps step.onnx for per-frame inference.

Usage:
    from stateguard.models.facephys_runner import FacePhysRunner
    rppg = FacePhysRunner(model_path='weights/step.onnx', state_path='weights/state.pkl')
    bvp = rppg.step(face_36x36_rgb_uint8)   # one float per frame
"""
import os
import pickle
from typing import Optional

import numpy as np
import onnxruntime as ort


FACE_INPUT_NAME = 'arg_0.1'
DT_INPUT_NAME = 'onnx::Mul_37'
INPUT_RES = (36, 36)


class FacePhysRunner:
    """Per-frame streaming rPPG inference (InfinitePulse, ONNX).

    The model expects a single 36x36 RGB frame in [0, 1] (float32) and
    produces (a) one BVP scalar and (b) updated recurrent state. The state
    is carried over between calls.
    """

    def __init__(
        self,
        model_path: str,
        state_path: Optional[str] = None,
        num_threads: int = 1,
        fps: float = 30.0,
    ) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=['CPUExecutionProvider']
        )
        self._inputs = self.sess.get_inputs()
        self._input_names = [i.name for i in self._inputs]
        self._state_names = [
            n for n in self._input_names if n not in (FACE_INPUT_NAME, DT_INPUT_NAME)
        ]
        self.dt = np.array(1.0 / float(fps), dtype=np.float32)
        self.reset(state_path)

    def reset(self, state_path: Optional[str] = None) -> None:
        """Reset recurrent state. Optionally pre-warm from state.pkl."""
        state = {}
        for inp in self._inputs:
            if inp.name in (FACE_INPUT_NAME, DT_INPUT_NAME):
                continue
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            state[inp.name] = np.zeros(shape, dtype=np.float32)
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    pre = pickle.load(f)
                for k, v in pre.items():
                    if k in state and state[k].shape == v.shape:
                        state[k] = np.asarray(v, dtype=np.float32)
            except Exception:
                pass
        self.state = state

    @staticmethod
    def preprocess(face_rgb_uint8: np.ndarray) -> np.ndarray:
        """Resize-cropped 36x36 RGB uint8 -> (1,1,36,36,3) float32 in [0,1]."""
        x = np.asarray(face_rgb_uint8)
        if x.dtype != np.float32:
            x = x.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x[None, None]
        elif x.ndim == 4:
            x = x[None]
        return x

    def step(self, face_rgb: np.ndarray) -> float:
        """Run one frame; returns scalar BVP value, updates internal state."""
        feed = {FACE_INPUT_NAME: self.preprocess(face_rgb), DT_INPUT_NAME: self.dt}
        feed.update(self.state)
        out = self.sess.run(None, feed)
        # out[0] = BVP scalar, out[1:] = updated state in input order
        for idx, name in enumerate(self._state_names):
            self.state[name] = out[idx + 1]
        return float(np.asarray(out[0]).squeeze())
