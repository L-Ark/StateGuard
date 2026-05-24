"""Live webcam demo with overlay (BVP plot, HR, V/A). Press 'q' to quit.

This is the reference loop a UI (Tkinter / Qt / Electron) should mimic:
  - Read a frame from the camera
  - pipe.step(rgb)
  - Draw the FrameResult somehow
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from stateguard import StateGuardPipeline, StateGuardConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--weights', default=str(Path(__file__).parent.parent / 'stateguard' / 'weights'))
    ap.add_argument('--gate-va', action='store_true')
    args = ap.parse_args()

    w = Path(args.weights)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'Camera FPS: {src_fps:.1f}')

    pipe = StateGuardPipeline(StateGuardConfig(
        facephys_path=str(w / 'step.onnx'),
        va_path=str(w / 'va_mbf.onnx'),
        fatigue_path=str(w / 'fatigue.onnx') if (w / 'fatigue.onnx').exists() else None,
        state_path=str(w / 'state.pkl') if (w / 'state.pkl').exists() else None,
        source_fps=src_fps,
        gate_va=args.gate_va,
    ))

    bvp_buf = deque(maxlen=300)  # 10s plot
    print('Press q to quit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = pipe.step(rgb)
        bvp_buf.append(r.bvp if not np.isnan(r.bvp) else 0.0)

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

        t(f'HR: {r.hr:5.1f} bpm   q={r.quality:.2f}', 28)
        t(f'RMSSD: {r.rmssd:5.1f} ms   SDNN: {r.sdnn:5.1f} ms', 52)
        v = '--' if r.valence is None else f'{r.valence:+.2f}'
        a = '--' if r.arousal is None else f'{r.arousal:+.2f}'
        f_str = '--' if r.fatigue is None else f'{r.fatigue:.2f}'
        f_color = (0, 255, 255) if r.fatigue is None or r.fatigue < 0.5 else (0, 80, 255)
        t(f'V: {v}   A: {a}', 76, (0, 255, 255))
        t(f'Fatigue: {f_str}', 100, f_color)

        cv2.imshow('StateGuard', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
