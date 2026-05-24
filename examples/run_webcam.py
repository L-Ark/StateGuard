"""Live webcam demo with overlay (BVP plot, HR, V/A). Press 'q' to quit.

This is the reference loop a UI (Tkinter / Qt / Electron) should mimic:
  - Read a frame from the camera
  - pipe.step(rgb)
  - Draw the FrameResult somehow
"""
import argparse
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stateguard import StateGuardPipeline, StateGuardConfig
from stateguard.landmarks import FaceMeshRunner


@dataclass
class FatigueCalibrator:
    duration_sec: float = 90.0
    min_quality: float = 0.45
    min_samples: int = 4
    sample_every_sec: float = 5.0
    started_at: float = field(default_factory=time.time)
    last_sample_at: float = field(default_factory=lambda: -1e9)
    samples: list[float] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    valence_samples: list[float] = field(default_factory=list)
    arousal_samples: list[float] = field(default_factory=list)
    finished: bool = False
    baseline: Optional[float] = None
    spread: Optional[float] = None
    valence_baseline: Optional[float] = None
    valence_spread: Optional[float] = None
    arousal_baseline: Optional[float] = None
    arousal_spread: Optional[float] = None

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def remaining(self) -> float:
        return max(0.0, self.duration_sec - self.elapsed())

    def progress(self) -> float:
        if self.duration_sec <= 1e-6:
            return 1.0
        return min(1.0, self.elapsed() / self.duration_sec)

    def add_sample(
        self,
        fatigue: Optional[float],
        valence: Optional[float],
        arousal: Optional[float],
        quality: float,
        force: bool = False,
    ) -> None:
        if self.finished:
            return
        if fatigue is None or np.isnan(fatigue):
            return
        now = time.time()
        if not force and (now - self.last_sample_at) < self.sample_every_sec:
            return
        self.samples.append(float(fatigue))
        self.qualities.append(float(quality) if np.isfinite(quality) else float('nan'))
        if valence is not None and np.isfinite(valence):
            self.valence_samples.append(float(valence))
        if arousal is not None and np.isfinite(arousal):
            self.arousal_samples.append(float(arousal))
        self.last_sample_at = now

    def maybe_finish(self) -> bool:
        if self.finished:
            return True
        if self.elapsed() >= self.duration_sec:
            if self.samples:
                arr = np.asarray(self.samples, dtype=np.float32)
                self.baseline = float(np.median(arr))
                self.spread = float(np.std(arr) + 1e-6)
            else:
                # Fallback: avoid getting stuck forever if no valid windows were
                # collected during calibration. Keep raw fatigue unchanged.
                self.baseline = 0.5
                self.spread = 0.25
            if self.valence_samples:
                arr_v = np.asarray(self.valence_samples, dtype=np.float32)
                self.valence_baseline = float(np.median(arr_v))
                self.valence_spread = float(np.std(arr_v) + 1e-6)
            else:
                self.valence_baseline = 0.0
                self.valence_spread = 0.6
            if self.arousal_samples:
                arr_a = np.asarray(self.arousal_samples, dtype=np.float32)
                self.arousal_baseline = float(np.median(arr_a))
                self.arousal_spread = float(np.std(arr_a) + 1e-6)
            else:
                self.arousal_baseline = 0.0
                self.arousal_spread = 0.6
            self.finished = True
        return self.finished

    def confidence_text(self) -> str:
        if not self.finished:
            return 'calibrating'
        if len(self.samples) < self.min_samples:
            return f'low-confidence ({len(self.samples)} samples)'
        valid_q = [q for q in self.qualities if np.isfinite(q)]
        if valid_q and float(np.mean(valid_q)) < self.min_quality:
            return f'low-confidence ({len(self.samples)} samples, poor quality)'
        return f'ready ({len(self.samples)} samples)'

    def calibrated_value(self, raw_value: Optional[float]) -> Optional[float]:
        if raw_value is None or np.isnan(raw_value):
            return None
        if not self.finished or self.baseline is None:
            return float(raw_value)
        scale = max(0.18, (self.spread or 0.0) * 3.0)
        adjusted = 0.5 + (float(raw_value) - self.baseline) / scale
        return float(np.clip(adjusted, 0.0, 1.0))

    def calibrated_va_value(
        self,
        raw_value: Optional[float],
        baseline: Optional[float],
        spread: Optional[float],
    ) -> Optional[float]:
        if raw_value is None or np.isnan(raw_value):
            return None
        if not self.finished or baseline is None:
            return float(raw_value)
        scale = max(0.18, (spread or 0.0) * 3.0)
        adjusted = 0.5 + (float(raw_value) - baseline) / scale
        return float(np.clip(adjusted, -1.0, 1.0))


def measure_capture_fps(cap, max_seconds: float = 3.0, sample_frames: int = 90) -> float:
    start = time.time()
    frames = 0
    while frames < sample_frames and (time.time() - start) < max_seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.time() - start
    return frames / elapsed if elapsed > 0 else 0.0


def fmt_metric(value: float, decimals: int = 1) -> str:
    if value is None:
        return '--'
    try:
        if np.isnan(value):
            return '--'
    except TypeError:
        pass
    return f'{value:.{decimals}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--weights', default=str(Path(__file__).parent.parent / 'stateguard' / 'weights'))
    ap.add_argument('--gate-va', action='store_true')
    ap.add_argument('--va-mode', choices=['auto', 'vision', 'multimodal'], default='auto', help='VA mode: auto-switch, vision, or multimodal fusion')
    ap.add_argument('--fusion-alpha', type=float, default=0.5, help='Fusion alpha weight for vision in multimodal mode (0..1)')
    ap.add_argument('--va-quality-threshold', type=float, default=0.55, help='Auto mode threshold for switching to multimodal when HRV quality is strong')
    ap.add_argument('--va-quality-hysteresis', type=float, default=0.08, help='Auto mode hysteresis to avoid VA mode flapping')
    ap.add_argument('--calib-sec', type=float, default=90.0, help='Personal fatigue calibration duration in seconds (recommend 60-180)')
    ap.add_argument('--calib-min-quality', type=float, default=0.45, help='Minimum HRV quality to accept a fatigue sample during calibration')
    ap.add_argument('--calib-min-samples', type=int, default=4, help='Minimum valid fatigue windows required to finish calibration')
    ap.add_argument('--calib-sample-sec', type=float, default=5.0, help='Minimum seconds between fatigue calibration samples')
    ap.add_argument('--txt-log', default='stateguard_webcam.txt', help='Path to txt log file with system timestamps')
    args = ap.parse_args()

    w = Path(args.weights)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(
            f'Could not open camera {args.cam}. Try another index or check Windows camera permissions.'
        )
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    measured_fps = measure_capture_fps(cap)
    if measured_fps >= 1.0:
        src_fps = measured_fps
    elif reported_fps >= 1.0:
        src_fps = reported_fps
    else:
        src_fps = 30.0

    target_fps = 30.0
    if src_fps < target_fps - 0.5:
        target_fps = src_fps
        print(
            f'Warning: camera is slower than 30 fps (measured={measured_fps:.2f}). '
            f'Running pipeline at {target_fps:.2f} fps to avoid upsampling.'
        )

    print(f'Camera FPS: reported={reported_fps:.1f} measured={measured_fps:.2f} using={src_fps:.2f} target={target_fps:.2f}')

    pipe = StateGuardPipeline(StateGuardConfig(
        facephys_path=str(w / 'step.onnx'),
        va_path=str(w / 'va_mbf.onnx'),
        fatigue_path=str(w / 'fatigue.onnx') if (w / 'fatigue.onnx').exists() else None,
        state_path=str(w / 'state.pkl') if (w / 'state.pkl').exists() else None,
        fps=target_fps,
        source_fps=src_fps,
        hrv_warmup_sec=12.0,
        gate_va=args.gate_va,
        va_mode=args.va_mode,
        fusion_alpha=args.fusion_alpha,
        va_quality_threshold=args.va_quality_threshold,
        va_quality_hysteresis=args.va_quality_hysteresis,
    ))

    # Initialize mediapipe FaceMesh for real-time landmarks (optional)
    fm_runner = None
    try:
        fm_runner = FaceMeshRunner()
        fm_runner.ensure_model_loaded()
        print('FaceMesh: loaded')
    except Exception as e:
        fm_runner = None
        print(f'FaceMesh: unavailable ({e})')

    bvp_buf = deque(maxlen=300)  # 10s plot
    print('Press q to quit.')
    print(f'Txt log: {args.txt_log}')
    calib = FatigueCalibrator(
        duration_sec=max(0.0, float(args.calib_sec)),
        min_quality=float(args.calib_min_quality),
        min_samples=max(1, int(args.calib_min_samples)),
        sample_every_sec=max(0.5, float(args.calib_sample_sec)),
    )
    if calib.duration_sec > 0:
        print(
            f'Calibration started: keep your face visible and stay relaxed for {calib.duration_sec:.0f} seconds.'
        )
        if calib.duration_sec < 15.0:
            print(
                'Warning: calibration duration is shorter than the 15-second VA window. '\
                'Fatigue samples may stay at 0 until the first VA window closes.'
            )
        print(f'Calibration samples are collected every ~{calib.sample_every_sec:.1f} seconds (or on new VA windows).')
        print('Please avoid deliberate facial expressions during calibration.')
    last_va = (None, None)
    last_mode = None
    logf = open(args.txt_log, 'w', encoding='utf-8')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = pipe.step(rgb)
        # Strict sync: only collect calibration samples when a VA window closes
        if r.va_updated:
            calib.add_sample(r.fatigue, r.valence, r.arousal, r.quality, force=True)
        calib.maybe_finish()
        if r.va_updated:
            ts = datetime.now().astimezone().isoformat(timespec='milliseconds')
            fatigue_calib = calib.calibrated_value(r.fatigue)
            valence_calib = calib.calibrated_va_value(r.valence, calib.valence_baseline, calib.valence_spread)
            arousal_calib = calib.calibrated_va_value(r.arousal, calib.arousal_baseline, calib.arousal_spread)
            logf.write(
                f'{ts}\tframe={int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)}\tbvp={r.bvp:.6f}\t'
                f'hr={r.hr:.3f}\trmssd={r.rmssd:.3f}\tsdnn={r.sdnn:.3f}\tquality={r.quality:.3f}\t'
                f'va_mode={r.va_mode}\tfatigue_state={"ready" if calib.finished else "calibrating"}\t'
                f'fatigue_raw={"" if r.fatigue is None else f"{r.fatigue:.3f}"}\t'
                f'fatigue_calib={"" if fatigue_calib is None else f"{fatigue_calib:.3f}"}\t'
                f'calib_baseline={"" if calib.baseline is None else f"{calib.baseline:.3f}"}\t'
                f'valence_raw={"" if r.valence is None else f"{r.valence:.3f}"}\t'
                f'arousal_raw={"" if r.arousal is None else f"{r.arousal:.3f}"}\t'
                f'valence_calib={"" if valence_calib is None else f"{valence_calib:.3f}"}\t'
                f'arousal_calib={"" if arousal_calib is None else f"{arousal_calib:.3f}"}\t'
                f'va_calib_state={"ready" if calib.finished else "calibrating"}\n'
            )
            logf.flush()
        bvp_buf.append(r.bvp if not np.isnan(r.bvp) else 0.0)
        if r.va_mode != last_mode:
            last_mode = r.va_mode
            print(f'VA mode now: {r.va_mode}')

        if r.va_updated and (r.valence, r.arousal) != last_va:
            last_va = (r.valence, r.arousal)
            v = '--' if r.valence is None else f'{r.valence:+.2f}'
            a = '--' if r.arousal is None else f'{r.arousal:+.2f}'
            f_str = '--' if r.fatigue is None else f'{r.fatigue:.2f}'
            print(f'New window result: mode={r.va_mode}  V={v}  A={a}  Fatigue={f_str}')

        disp = frame.copy()
        # Draw landmarks if available
        if fm_runner is not None:
            try:
                # fm_runner.process expects RGB
                fm_runner.process(rgb)
                fm_runner.draw_landmarks(disp, None)
            except Exception:
                # don't break UI on landmark errors
                pass
        if r.face_box:
            x1, y1, x2, y2 = r.face_box
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        h, w_ = disp.shape[:2]
        # BVP plot strip
        if len(bvp_buf) > 5:
            arr = np.array(bvp_buf, dtype=np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            arr = np.clip(arr, -3, 3)
            xs = np.linspace(0, w_ - 1, len(arr)).astype(int)
            ys = (h - 30 - (arr + 3) / 6 * 60).astype(int)
            for i in range(1, len(xs)):
                cv2.line(disp, (xs[i-1], ys[i-1]), (xs[i], ys[i]), (0, 200, 255), 1)

        def t(s, y, c=(255, 255, 255)):
            cv2.putText(disp, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        hr = fmt_metric(r.hr)
        rmssd = fmt_metric(r.rmssd)
        sdnn = fmt_metric(r.sdnn)
        q = fmt_metric(r.quality, 2)
        t(f'HR: {hr} bpm   q={q}', 28)
        t(f'RMSSD: {rmssd} ms   SDNN: {sdnn} ms', 52)
        t(f'VA mode: {r.va_mode}', 68, (0, 220, 255) if r.va_mode == 'multimodal' else (255, 200, 0))
        v = '--' if r.valence is None or np.isnan(r.valence) else f'{r.valence:+.2f}'
        a = '--' if r.arousal is None or np.isnan(r.arousal) else f'{r.arousal:+.2f}'
        if calib.finished:
            f_raw = '--' if r.fatigue is None or np.isnan(r.fatigue) else f'{r.fatigue:.2f}'
            f_cal = '--' if fatigue_calib is None or np.isnan(fatigue_calib) else f'{fatigue_calib:.2f}'
            f_color = (0, 255, 255) if fatigue_calib is None or np.isnan(fatigue_calib) or fatigue_calib < 0.5 else (0, 80, 255)
            t(f'Fatigue (calib): {f_cal}   raw: {f_raw}', 92, f_color)
            if calib.baseline is not None:
                t(f'Personal baseline: {calib.baseline:.2f}', 116, (180, 180, 180))
            t(f'Calibration: {calib.confidence_text()}', 140, (255, 220, 120) if len(calib.samples) < calib.min_samples else (120, 255, 120))
            v_cal = '--' if valence_calib is None or np.isnan(valence_calib) else f'{valence_calib:+.2f}'
            a_cal = '--' if arousal_calib is None or np.isnan(arousal_calib) else f'{arousal_calib:+.2f}'
            t(f'V: {v_cal}   A: {a_cal}', 164, (0, 255, 255))
            t(f'Raw V/A: {v} / {a}', 188, (180, 180, 180))
        else:
            remain = int(round(calib.remaining()))
            mm = remain // 60
            ss = remain % 60
            target_windows = max(calib.min_samples, int(round(calib.duration_sec / max(0.5, calib.sample_every_sec))))
            t(f'Calibration in progress: {mm:02d}:{ss:02d} remaining', 92, (0, 200, 255))
            t('Please stay relaxed and keep your face visible', 116, (0, 200, 255))
            t(f'Collected samples: {len(calib.samples)}/{target_windows}', 140, (255, 255, 255))
            t('VA will be calibrated after this phase', 164, (255, 255, 255))
            t(f'Raw V: {v}   Raw A: {a}', 188, (0, 255, 255))

        cv2.imshow('StateGuard', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    logf.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
