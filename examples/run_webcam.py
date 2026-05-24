"""Live webcam demo with overlay (BVP plot, HR, V/A). Press 'q' to quit.

This is the reference loop a UI (Tkinter / Qt / Electron) should mimic:
  - Read a frame from the camera
  - pipe.step(rgb)
  - Draw the FrameResult somehow
"""
import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stateguard import StateGuardPipeline, StateGuardConfig


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
    ap.add_argument('--va-mode', choices=['vision', 'multimodal'], default='vision', help='VA mode: vision or multimodal fusion')
    ap.add_argument('--fusion-alpha', type=float, default=0.5, help='Fusion alpha weight for vision in multimodal mode (0..1)')
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
    ))

    bvp_buf = deque(maxlen=300)  # 10s plot
    print('Press q to quit.')
    last_va = (None, None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = pipe.step(rgb)
        bvp_buf.append(r.bvp if not np.isnan(r.bvp) else 0.0)

        if r.va_updated and (r.valence, r.arousal) != last_va:
            last_va = (r.valence, r.arousal)
            v = '--' if r.valence is None else f'{r.valence:+.2f}'
            a = '--' if r.arousal is None else f'{r.arousal:+.2f}'
            f_str = '--' if r.fatigue is None else f'{r.fatigue:.2f}'
            print(f'New window result: V={v}  A={a}  Fatigue={f_str}')

        disp = frame.copy()
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
        v = '--' if r.valence is None or np.isnan(r.valence) else f'{r.valence:+.2f}'
        a = '--' if r.arousal is None or np.isnan(r.arousal) else f'{r.arousal:+.2f}'
        f_str = '--' if r.fatigue is None or np.isnan(r.fatigue) else f'{r.fatigue:.2f}'
        f_color = (0, 255, 255) if r.fatigue is None or np.isnan(r.fatigue) or r.fatigue < 0.5 else (0, 80, 255)
        t(f'V: {v}   A: {a}', 76, (0, 255, 255))
        t(f'Fatigue: {f_str}', 100, f_color)

        cv2.imshow('StateGuard', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
