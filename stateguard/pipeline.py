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
import time

import numpy as np

from .models.facephys_runner import FacePhysRunner
from .models.va_runner import VARunner
from .models.fatigue_runner import FatigueRunner
from .models.face import FaceCropper
from .hrv import HRVStream
import cv2
try:
    from .landmarks import FaceMeshRunner, EyeMetricsTracker
except Exception:
    FaceMeshRunner = None
    EyeMetricsTracker = None


@dataclass
class FrameResult:
    bvp: float
    hr: float
    rmssd: float
    sdnn: float
    quality: float
    va_mode: str = 'vision'
    valence: Optional[float] = None  # filled in only when a VA window closes
    arousal: Optional[float] = None
    fatigue: Optional[float] = None  # P(fatigue) in [0,1]; refreshed each VA window
    face_box: Optional[tuple] = None
    landmarks: Optional[List[dict]] = None
    eye_ear: Optional[float] = None
    eye_closed: Optional[bool] = None
    perclos: Optional[float] = None
    blink_rate: Optional[float] = None
    mouth_open: Optional[float] = None
    yawn_prob: Optional[float] = None
    yawn_rate: Optional[float] = None
    fatigue_landmark: Optional[float] = None
    va_updated: bool = False


@dataclass
class StateGuardConfig:
    facephys_path: str
    va_path: str
    state_path: Optional[str] = None
    fatigue_path: Optional[str] = None  # if None, fatigue head is disabled
    # VA mode: 'vision' = image-only; 'multimodal' = fuse vision + rPPG;
    # 'auto' switches between the two using HRV quality.
    va_mode: str = 'auto'
    # Fusion weight for vision logits when va_mode='multimodal' (0..1)
    fusion_alpha: float = 0.5
    # In auto mode, q >= threshold selects multimodal; q <= threshold - hysteresis
    # selects vision. This avoids flapping when quality hovers near the cutoff.
    va_quality_threshold: float = 0.55
    va_quality_hysteresis: float = 0.08
    fps: float = 30.0           # FacePhys was trained at 30fps; the pipeline
                                # always processes frames at this rate. If the
                                # source camera/video runs faster, set
                                # `source_fps` and the pipeline will decimate.
    source_fps: Optional[float] = None
    va_window_sec: float = 15.0
    va_keyframes: int = 4
    hrv_window_sec: float = 30.0
    hrv_warmup_sec: float = 12.0
    gate_va: bool = False
    num_threads: int = 1
    # Eye metrics (PERCLOS / blink rate) from face landmarks
    enable_eye_metrics: bool = True
    perclos_window_sec: float = 60.0
    eye_closed_thresh: float = 0.20
    blink_min_sec: float = 0.05
    blink_max_sec: float = 0.60
    blink_window_sec: float = 60.0
    yawn_open_thresh: float = 0.30
    yawn_open_max: float = 0.55
    yawn_min_sec: float = 0.60
    yawn_max_sec: float = 3.00
    yawn_window_sec: float = 60.0


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
        self.hrv = HRVStream(
            fps=cfg.fps,
            window_sec=cfg.hrv_window_sec,
            warmup_sec=cfg.hrv_warmup_sec,
        )

        self._win_frames: List[np.ndarray] = []  # 224x224 RGB uint8 keyframe candidates
        self._win_idx_target = max(1, int(cfg.va_keyframes))
        self._win_total_target = max(1, int(round(cfg.fps * cfg.va_window_sec)))
        self._frames_in_window = 0
        self._last_va = (None, None)
        self._last_fatigue: Optional[float] = None
        self._va_mode_active = 'vision'
        self._fm_runner = None
        self._last_landmarks = None
        self._eye_tracker = None
        self._last_eye_ear = None
        self._last_eye_closed = None
        self._last_perclos = None
        self._last_blink_rate = None
        self._last_mouth_open = None
        self._last_yawn_prob = None
        self._last_yawn_rate = None
        self._last_fatigue_landmark = None
        if FaceMeshRunner is not None:
            try:
                self._fm_runner = FaceMeshRunner()
                self._fm_runner.ensure_model_loaded()
            except Exception:
                self._fm_runner = None
        if EyeMetricsTracker is not None and self._fm_runner is not None and self.cfg.enable_eye_metrics:
            self._eye_tracker = EyeMetricsTracker(
                perclos_window_sec=self.cfg.perclos_window_sec,
                eye_closed_thresh=self.cfg.eye_closed_thresh,
                blink_min_sec=self.cfg.blink_min_sec,
                blink_max_sec=self.cfg.blink_max_sec,
                blink_window_sec=self.cfg.blink_window_sec,
                yawn_open_thresh=self.cfg.yawn_open_thresh,
                yawn_open_max=self.cfg.yawn_open_max,
                yawn_min_sec=self.cfg.yawn_min_sec,
                yawn_max_sec=self.cfg.yawn_max_sec,
                yawn_window_sec=self.cfg.yawn_window_sec,
            )
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

    def _resolve_va_mode(self, quality: float) -> str:
        requested = str(self.cfg.va_mode).lower().strip()
        if requested == 'vision':
            self._va_mode_active = 'vision'
            return self._va_mode_active
        if requested == 'multimodal':
            self._va_mode_active = 'multimodal'
            return self._va_mode_active

        # Auto mode: use multimodal only when quality is strong enough.
        q = float(quality) if np.isfinite(quality) else 0.0
        high = float(np.clip(getattr(self.cfg, 'va_quality_threshold', 0.55), 0.0, 1.0))
        low = float(np.clip(high - getattr(self.cfg, 'va_quality_hysteresis', 0.08), 0.0, high))
        if self._va_mode_active == 'multimodal':
            if q <= low:
                self._va_mode_active = 'vision'
        else:
            if q >= high:
                self._va_mode_active = 'multimodal'
        return self._va_mode_active

    def reset(self) -> None:
        self.rppg.reset(self.cfg.state_path)
        self.face.reset()
        self.hrv = HRVStream(
            fps=self.cfg.fps,
            window_sec=self.cfg.hrv_window_sec,
            warmup_sec=self.cfg.hrv_warmup_sec,
        )
        self._win_frames.clear()
        self._frames_in_window = 0
        self._last_va = (None, None)
        self._last_fatigue = None
        self._va_mode_active = 'vision'
        # reset landmarks runner state
        try:
            if self._fm_runner is not None:
                self._fm_runner.close()
        except Exception:
            pass
        self._fm_runner = None
        self._last_landmarks = None
        self._eye_tracker = None
        self._last_eye_ear = None
        self._last_eye_closed = None
        self._last_perclos = None
        self._last_blink_rate = None
        self._last_mouth_open = None
        self._last_yawn_prob = None
        self._last_yawn_rate = None
        self._last_fatigue_landmark = None
        if FaceMeshRunner is not None:
            try:
                self._fm_runner = FaceMeshRunner()
                self._fm_runner.ensure_model_loaded()
            except Exception:
                self._fm_runner = None
        if EyeMetricsTracker is not None and self._fm_runner is not None and self.cfg.enable_eye_metrics:
            self._eye_tracker = EyeMetricsTracker(
                perclos_window_sec=self.cfg.perclos_window_sec,
                eye_closed_thresh=self.cfg.eye_closed_thresh,
                blink_min_sec=self.cfg.blink_min_sec,
                blink_max_sec=self.cfg.blink_max_sec,
                blink_window_sec=self.cfg.blink_window_sec,
                yawn_open_thresh=self.cfg.yawn_open_thresh,
                yawn_open_max=self.cfg.yawn_open_max,
                yawn_min_sec=self.cfg.yawn_min_sec,
                yawn_max_sec=self.cfg.yawn_max_sec,
                yawn_window_sec=self.cfg.yawn_window_sec,
            )
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
        hr, rmssd, sdnn, q = self.hrv.estimate()
        self._va_mode_active = self._resolve_va_mode(q)
        # Optional gating: skip VA if HRV+quality looks fine
        run_va = True
        if self.cfg.gate_va:
            # Trigger VA only when signal is reliable AND something looks off:
            # low HRV (RMSSD < 25 ms) or unusually low/high HR.
            if not (q > 0.5 and (rmssd < 25.0 or hr < 50 or hr > 110)):
                run_va = False
        out = None
        if run_va:
            batch = np.stack(self._win_frames, axis=0)
            preds = self.va.predict(batch)  # (N, 2) continuous [valence, arousal]

            def cont_to_logits(x: np.ndarray, centers=(-1.0, 0.0, 1.0), sigma=0.6):
                # x: (N,) continuous in approx [-2,2]; output (N,3) logits
                x = np.asarray(x, dtype=np.float32).reshape(-1)
                centers_a = np.asarray(centers, dtype=np.float32).reshape(1, -1)
                dif = x.reshape(-1, 1) - centers_a
                logits = -0.5 * (dif ** 2) / (sigma ** 2)
                return logits

            # Vision logits per-frame
            vis_v_logits = cont_to_logits(preds[:, 0])
            vis_a_logits = cont_to_logits(preds[:, 1])
            vision_v = vis_v_logits.mean(axis=0)
            vision_a = vis_a_logits.mean(axis=0)

            if self._va_mode_active != 'multimodal':
                # image-only: keep previous behavior (mean continuous preds)
                out = preds.mean(axis=0)
                self._last_va = (float(out[0]), float(out[1]))
            else:
                # Multimodal: derive simple rPPG-based logits from HRV and fuse
                def rppg_to_logits(hr, rmssd, sdnn, centers=(-1.0, 0.0, 1.0)):
                    # Normalize inputs to rough ranges and produce 3-class logits
                    # hr: ~40-120 bpm -> map to [-1,1]
                    h = np.clip((hr - 70.0) / 30.0, -1.0, 1.0)
                    r = 0.0 if not np.isfinite(rmssd) else np.clip((rmssd - 20.0) / 30.0, -1.0, 1.0)
                    s = 0.0 if not np.isfinite(sdnn) else np.clip((sdnn - 30.0) / 30.0, -1.0, 1.0)
                    # For arousal, use HR as primary; for valence, use RMSSD/SDNN as proxy
                    v_val = (r + s) * 0.5
                    a_val = h
                    # convert to logits like cont_to_logits
                    def _to_logits(val):
                        c = np.asarray(centers, dtype=np.float32)
                        dif = val - c
                        return -0.5 * (dif ** 2) / (0.6 ** 2)

                    return _to_logits(v_val), _to_logits(a_val)

                rppg_v_logits, rppg_a_logits = rppg_to_logits(hr, rmssd, sdnn)

                alpha = float(max(0.0, min(1.0, getattr(self.cfg, 'fusion_alpha', 0.5))))
                fused_v_logits = alpha * vision_v + (1.0 - alpha) * rppg_v_logits
                fused_a_logits = alpha * vision_a + (1.0 - alpha) * rppg_a_logits

                # convert fused logits to continuous expected value using softmax over centers
                def logits_to_cont(logits, centers=(-1.0, 0.0, 1.0)):
                    exp = np.exp(logits - np.max(logits))
                    p = exp / (np.sum(exp) + 1e-12)
                    centers_a = np.asarray(centers, dtype=np.float32)
                    return float(np.dot(p, centers_a))

                v_cont = logits_to_cont(fused_v_logits)
                a_cont = logits_to_cont(fused_a_logits)
                self._last_va = (float(v_cont), float(a_cont))
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
            mode = self._resolve_va_mode(q)
            return FrameResult(
                bvp=float('nan'), hr=hr, rmssd=rmssd, sdnn=sdnn, quality=q,
                va_mode=mode,
                valence=self._last_va[0], arousal=self._last_va[1],
                fatigue=self._last_fatigue,
                landmarks=self._last_landmarks,
                eye_ear=self._last_eye_ear,
                eye_closed=self._last_eye_closed,
                perclos=self._last_perclos,
                blink_rate=self._last_blink_rate,
                mouth_open=self._last_mouth_open,
                yawn_prob=self._last_yawn_prob,
                yawn_rate=self._last_yawn_rate,
                fatigue_landmark=self._last_fatigue_landmark,
                face_box=None,
                va_updated=False,
            )
        self._decim_acc -= 1.0

        face, box = self.face.crop(frame_rgb)
        if face is None:
            mode = self._resolve_va_mode(self._hrv_cache[3])
            return FrameResult(bvp=float('nan'), hr=float('nan'), rmssd=float('nan'),
                               sdnn=float('nan'), quality=0.0,
                               va_mode=mode,
                               valence=self._last_va[0], arousal=self._last_va[1],
                               fatigue=self._last_fatigue,
                               landmarks=self._last_landmarks,
                               eye_ear=self._last_eye_ear,
                               eye_closed=self._last_eye_closed,
                               perclos=self._last_perclos,
                               blink_rate=self._last_blink_rate,
                               mouth_open=self._last_mouth_open,
                               yawn_prob=self._last_yawn_prob,
                               yawn_rate=self._last_yawn_rate,
                               fatigue_landmark=self._last_fatigue_landmark,
                               va_updated=False)

        # Per-frame rPPG
        face_36 = cv2.resize(face, (36, 36), interpolation=cv2.INTER_AREA)
        bvp = self.rppg.step(face_36)
        self.hrv.push(bvp)

        # Face landmarks (Mediapipe FaceMesh) - process full-res face RGB
        if self._fm_runner is not None:
            try:
                lm = self._fm_runner.process(face)
                self._last_landmarks = lm
            except Exception:
                # keep previous landmarks on error
                pass
        if self._eye_tracker is not None:
            metrics = self._eye_tracker.update(self._last_landmarks, time.time())
            self._last_eye_ear = metrics.get('ear')
            self._last_eye_closed = metrics.get('eye_closed')
            self._last_perclos = metrics.get('perclos')
            self._last_blink_rate = metrics.get('blink_rate')
            self._last_mouth_open = metrics.get('mouth_open')
            self._last_yawn_prob = metrics.get('yawn_prob')
            self._last_yawn_rate = metrics.get('yawn_rate')
            self._last_fatigue_landmark = self._compute_landmark_fatigue(
                self._last_perclos, self._last_blink_rate, self._last_yawn_rate
            )

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
        mode = self._resolve_va_mode(q)
        return FrameResult(
            bvp=bvp, hr=hr, rmssd=rmssd, sdnn=sdnn, quality=q,
            va_mode=mode,
            valence=self._last_va[0], arousal=self._last_va[1],
            fatigue=self._last_fatigue,
            face_box=box,
            landmarks=self._last_landmarks,
            eye_ear=self._last_eye_ear,
            eye_closed=self._last_eye_closed,
            perclos=self._last_perclos,
            blink_rate=self._last_blink_rate,
            mouth_open=self._last_mouth_open,
            yawn_prob=self._last_yawn_prob,
            yawn_rate=self._last_yawn_rate,
            fatigue_landmark=self._last_fatigue_landmark,
            va_updated=(new_va is not None),
        )

    @staticmethod
    def _compute_landmark_fatigue(
        perclos: Optional[float],
        blink_rate: Optional[float],
        yawn_rate: Optional[float],
    ) -> Optional[float]:
        if perclos is None and blink_rate is None and yawn_rate is None:
            return None
        p = float(perclos) if perclos is not None else 0.0
        # Normalize blink rate: 10-30 blinks/min mapped to 0..1
        if blink_rate is None:
            b = 0.0
        else:
            b = float(np.clip((blink_rate - 10.0) / 20.0, 0.0, 1.0))
        # Normalize yawn rate: 0-4 yawns/min mapped to 0..1
        if yawn_rate is None:
            y = 0.0
        else:
            y = float(np.clip(yawn_rate / 4.0, 0.0, 1.0))
        score = 0.6 * p + 0.25 * y + 0.15 * b
        return float(np.clip(score, 0.0, 1.0))
