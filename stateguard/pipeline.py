"""End-to-end streaming pipeline: webcam frame -> rPPG + gated VA + HRV.

Design:
  - Per-frame: face crop -> 36x36 -> FacePhys step -> BVP -> HRV buffer
  - Per VA window (default 15s): keep N keyframes (default 4, uniform) at
    full resolution; once window closes, batch-predict V/A.
  - Gated mode: VA is only invoked when HRV trigger fires (e.g. low arousal
    or HRV drop). Disabled by default for the demo.

This class is the single integration point a UI should consume.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, List

import numpy as np

from .models.facephys_runner import FacePhysRunner
from .models.va_runner import VARunner
from .models.fatigue_runner import FatigueRunner
from .models.face import FaceCropper
from .hrv import HRVStream
import cv2


@dataclass
class FrameResult:
    bvp: float
    hr: float
    rmssd: float
    sdnn: float
    quality: float
    valence: Optional[float] = None  # filled in only when a VA window closes
    arousal: Optional[float] = None
    fatigue: Optional[float] = None  # P(fatigue) in [0,1]; refreshed each VA window
    face_box: Optional[tuple] = None


@dataclass
class StateGuardConfig:
    facephys_path: str
    va_path: str
    state_path: Optional[str] = None
    fatigue_path: Optional[str] = None  # if None, fatigue head is disabled
    fps: float = 30.0           # FacePhys was trained at 30fps; the pipeline
                                # always processes frames at this rate. If the
                                # source camera/video runs faster, set
                                # `source_fps` and the pipeline will decimate.
    source_fps: Optional[float] = None
    va_window_sec: float = 15.0
    va_keyframes: int = 4
    hrv_window_sec: float = 30.0
    gate_va: bool = False
    num_threads: int = 1


class StateGuardPipeline:
    def __init__(self, cfg: StateGuardConfig) -> None:
        self.cfg = cfg
        self.face = FaceCropper()
        self.rppg = FacePhysRunner(
            cfg.facephys_path, state_path=cfg.state_path,
            num_threads=cfg.num_threads, fps=cfg.fps,
        )
        self.va = VARunner(cfg.va_path, num_threads=cfg.num_threads)
        self.fatigue = (
            FatigueRunner(cfg.fatigue_path, num_threads=cfg.num_threads)
            if cfg.fatigue_path else None
        )
        self.hrv = HRVStream(fps=cfg.fps, window_sec=cfg.hrv_window_sec)

        self._win_frames: List[np.ndarray] = []  # 224x224 RGB uint8 keyframe candidates
        self._win_idx_target = max(1, int(cfg.va_keyframes))
        self._win_total_target = max(1, int(round(cfg.fps * cfg.va_window_sec)))
        self._frames_in_window = 0
        self._last_va = (None, None)
        self._last_fatigue: Optional[float] = None
        # Throttle HRV recompute (Welch+peaks is the per-frame hot path)
        self._hrv_every = max(1, int(round(cfg.fps)))  # ~1Hz
        self._hrv_counter = 0
        self._hrv_cache = (float('nan'), float('nan'), float('nan'), 0.0)
        # Source-FPS decimation. FacePhys was trained at 30fps; if the source
        # is faster we drop frames so the model sees a uniform 30fps stream.
        # Using a fractional accumulator keeps the rate accurate even when the
        # ratio is non-integer (e.g. 25 -> 30 not supported, but 60 -> 30 is).
        self._src_fps = float(cfg.source_fps or cfg.fps)
        self._tgt_fps = float(cfg.fps)
        if self._src_fps < self._tgt_fps - 0.5:
            raise ValueError(
                f'source_fps {self._src_fps} < target fps {self._tgt_fps}: '
                f'cannot upsample. Lower cfg.fps or use a faster source.'
            )
        self._decim_acc = 0.0
        self._last_face_36 = None  # cache to repeat when frame is dropped (unused now)

    def reset(self) -> None:
        self.rppg.reset(self.cfg.state_path)
        self.face.reset()
        self.hrv = HRVStream(fps=self.cfg.fps, window_sec=self.cfg.hrv_window_sec)
        self._win_frames.clear()
        self._frames_in_window = 0
        self._last_va = (None, None)
        self._last_fatigue = None
        self._decim_acc = 0.0
        self._hrv_counter = 0
        self._hrv_cache = (float('nan'), float('nan'), float('nan'), 0.0)

    def _maybe_collect_keyframe(self, face_rgb: np.ndarray) -> None:
        """Uniformly pick `va_keyframes` frames per window without buffering everything."""
        # Pick frame indices: round(linspace(0, total-1, k))
        i = self._frames_in_window
        total = self._win_total_target
        k = self._win_idx_target
        # sample positions in [0, total-1]
        targets = np.round(np.linspace(0, total - 1, k)).astype(int)
        if i in targets and len(self._win_frames) < k:
            kf = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            self._win_frames.append(kf)

    def _close_window_if_due(self) -> Optional[np.ndarray]:
        if self._frames_in_window < self._win_total_target:
            return None
        self._frames_in_window = 0
        if not self._win_frames:
            return None
        # Optional gating: skip VA if HRV+quality looks fine
        run_va = True
        if self.cfg.gate_va:
            hr, rmssd, sdnn, q = self.hrv.estimate()
            # Trigger VA only when signal is reliable AND something looks off:
            # low HRV (RMSSD < 25 ms) or unusually low/high HR.
            if not (q > 0.5 and (rmssd < 25.0 or hr < 50 or hr > 110)):
                run_va = False
        out = None
        if run_va:
            batch = np.stack(self._win_frames, axis=0)
            preds = self.va.predict(batch)
            out = preds.mean(axis=0)
            self._last_va = (float(out[0]), float(out[1]))
            if self.fatigue is not None:
                # FatigueRunner does its own resize; pass 224x224 keyframes directly
                p = self.fatigue.predict(batch)  # (N,) P(fatigue)
                self._last_fatigue = float(p.mean())
        self._win_frames.clear()
        return out

    def step(self, frame_rgb: np.ndarray) -> FrameResult:
        """Process one frame; returns latest combined state.

        If `source_fps > fps`, frames are decimated so the model sees a
        steady stream at the target rate. Dropped frames return the last
        known result (same `_last_va`, NaN HR until window fills again).
        """
        # Source-FPS decimation: emit one model step per `src/tgt` source frames.
        self._decim_acc += self._tgt_fps / self._src_fps
        if self._decim_acc < 1.0 - 1e-6:
            # skip this frame at the model level
            hr, rmssd, sdnn, q = self._hrv_cache
            return FrameResult(
                bvp=float('nan'), hr=hr, rmssd=rmssd, sdnn=sdnn, quality=q,
                valence=self._last_va[0], arousal=self._last_va[1],
                fatigue=self._last_fatigue,
                face_box=None,
            )
        self._decim_acc -= 1.0

        face, box = self.face.crop(frame_rgb)
        if face is None:
            return FrameResult(bvp=float('nan'), hr=float('nan'), rmssd=float('nan'),
                               sdnn=float('nan'), quality=0.0,
                               valence=self._last_va[0], arousal=self._last_va[1],
                               fatigue=self._last_fatigue)

        # Per-frame rPPG
        face_36 = cv2.resize(face, (36, 36), interpolation=cv2.INTER_AREA)
        bvp = self.rppg.step(face_36)
        self.hrv.push(bvp)

        # Collect keyframes for VA
        self._maybe_collect_keyframe(face)
        self._frames_in_window += 1
        new_va = self._close_window_if_due()
        if new_va is not None:
            self._last_va = (float(new_va[0]), float(new_va[1]))

        # HRV (cheap; throttled to ~1Hz)
        self._hrv_counter += 1
        if self._hrv_counter >= self._hrv_every:
            self._hrv_cache = self.hrv.estimate()
            self._hrv_counter = 0
        hr, rmssd, sdnn, q = self._hrv_cache
        return FrameResult(
            bvp=bvp, hr=hr, rmssd=rmssd, sdnn=sdnn, quality=q,
            valence=self._last_va[0], arousal=self._last_va[1],
            fatigue=self._last_fatigue,
            face_box=box,
        )
