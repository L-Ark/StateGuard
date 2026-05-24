"""Run StateGuard on a video file (no UI). Saves a CSV of per-frame state."""
import argparse
import time
import csv
from pathlib import Path

import cv2
import numpy as np

from stateguard import StateGuardPipeline, StateGuardConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help='Path to video file')
    ap.add_argument('--weights', default=str(Path(__file__).parent.parent / 'stateguard' / 'weights'))
    ap.add_argument('--out', default='stateguard_out.csv')
    ap.add_argument('--gate-va', action='store_true', help='Only run VA when HRV anomaly')
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'Source FPS: {fps_src:.1f}')

    w = Path(args.weights)
    cfg = StateGuardConfig(
        facephys_path=str(w / 'step.onnx'),
        va_path=str(w / 'va_mbf.onnx'),
        fatigue_path=str(w / 'fatigue.onnx') if (w / 'fatigue.onnx').exists() else None,
        state_path=str(w / 'state.pkl') if (w / 'state.pkl').exists() else None,
        source_fps=fps_src,
        gate_va=args.gate_va,
        num_threads=args.threads,
    )
    pipe = StateGuardPipeline(cfg)

    rows = []
    n = 0
    t0 = time.time()
    with open(args.out, 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['frame', 'bvp', 'hr', 'rmssd', 'sdnn', 'quality', 'valence', 'arousal', 'fatigue'])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = pipe.step(rgb)
            w_.writerow([n, r.bvp, r.hr, r.rmssd, r.sdnn, r.quality,
                         r.valence if r.valence is not None else '',
                         r.arousal if r.arousal is not None else '',
                         r.fatigue if r.fatigue is not None else ''])
            n += 1
            if n % 60 == 0:
                fps_proc = n / (time.time() - t0)
                print(f'  frame {n:5d}  HR={r.hr:5.1f}  RMSSD={r.rmssd:5.1f}  '
                      f'V={r.valence}  A={r.arousal}  F={r.fatigue}  proc {fps_proc:.1f} fps')
    cap.release()
    print(f'Wrote {n} rows to {args.out} ({n/(time.time()-t0):.1f} fps avg)')


if __name__ == '__main__':
    main()
