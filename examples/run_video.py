"""Run StateGuard on a video file (no UI). Saves CSV and txt logs of per-frame state."""
import argparse
import time
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stateguard import StateGuardPipeline, StateGuardConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help='Path to video file')
    ap.add_argument('--weights', default=str(Path(__file__).parent.parent / 'stateguard' / 'weights'))
    ap.add_argument('--out', default='stateguard_out.csv')
    ap.add_argument('--txt-log', default=None, help='Path to txt log file (defaults to the same basename as --out)')
    ap.add_argument('--gate-va', action='store_true', help='Only run VA when HRV anomaly')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--va-mode', choices=['auto', 'vision', 'multimodal'], default='auto', help='VA mode: auto-switch, vision, or multimodal fusion')
    ap.add_argument('--fusion-alpha', type=float, default=0.5, help='Fusion alpha weight for vision in multimodal mode (0..1)')
    ap.add_argument('--va-quality-threshold', type=float, default=0.55, help='Auto mode threshold for switching to multimodal when HRV quality is strong')
    ap.add_argument('--va-quality-hysteresis', type=float, default=0.08, help='Auto mode hysteresis to avoid VA mode flapping')
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
        va_mode=args.va_mode,
        fusion_alpha=args.fusion_alpha,
        va_quality_threshold=args.va_quality_threshold,
        va_quality_hysteresis=args.va_quality_hysteresis,
    )
    pipe = StateGuardPipeline(cfg)

    rows = []
    n = 0
    t0 = time.time()
    txt_path = Path(args.txt_log) if args.txt_log else Path(args.out).with_suffix('.txt')
    with open(args.out, 'w', newline='') as f:
        with open(txt_path, 'w', encoding='utf-8') as logf:
            w_ = csv.writer(f)
            w_.writerow(['frame', 'bvp', 'hr', 'rmssd', 'sdnn', 'quality', 'va_mode', 'valence', 'arousal', 'fatigue'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = pipe.step(rgb)
                ts = datetime.now().astimezone().isoformat(timespec='milliseconds')
                w_.writerow([n, r.bvp, r.hr, r.rmssd, r.sdnn, r.quality, r.va_mode,
                             r.valence if r.valence is not None else '',
                             r.arousal if r.arousal is not None else '',
                             r.fatigue if r.fatigue is not None else ''])
                logf.write(
                    f'{ts}\tframe={n}\tbvp={r.bvp:.6f}\thr={r.hr:.3f}\trmssd={r.rmssd:.3f}\t'
                    f'sdnn={r.sdnn:.3f}\tquality={r.quality:.3f}\tva_mode={r.va_mode}\t'
                    f'valence={"" if r.valence is None else f"{r.valence:.3f}"}\t'
                    f'arousal={"" if r.arousal is None else f"{r.arousal:.3f}"}\t'
                    f'fatigue={"" if r.fatigue is None else f"{r.fatigue:.3f}"}\n'
                )
                logf.flush()
                n += 1
                if n % 60 == 0:
                    fps_proc = n / (time.time() - t0)
                    print(f'  frame {n:5d}  mode={r.va_mode:10s}  HR={r.hr:5.1f}  RMSSD={r.rmssd:5.1f}  '
                          f'V={r.valence}  A={r.arousal}  F={r.fatigue}  proc {fps_proc:.1f} fps')
    cap.release()
    print(f'Wrote {n} rows to {args.out} and {txt_path} ({n/(time.time()-t0):.1f} fps avg)')


if __name__ == '__main__':
    main()
