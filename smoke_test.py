"""End-to-end accuracy smoke test on a CAST clip.

Validates:
  - HR estimate vs GT PPG (should be within 3 bpm)
  - Per-frame BVP signal has Pearson r > 0.3 vs GT PPG (real signal)
  - Pipeline latency stays realtime
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from stateguard import StateGuardPipeline, StateGuardConfig

VID = '/root/shared/CAST-Phys/fold_1/subject_001/Q1_1/vid_crop.avi'
BIO = '/root/shared/CAST-Phys/fold_1/subject_001/Q1_1/bio.csv'
W = Path(__file__).parent / 'stateguard' / 'weights'

cap = cv2.VideoCapture(VID)
src_fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Source FPS: {src_fps}')

cfg = StateGuardConfig(
    facephys_path=str(W / 'step.onnx'),
    va_path=str(W / 'va_mbf.onnx'),
    state_path=None,
    source_fps=src_fps,   # ★ critical: tell pipeline the camera/video rate
)
pipe = StateGuardPipeline(cfg)

latencies = []
rows = []
t0 = time.time()
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = time.perf_counter()
    r = pipe.step(rgb)
    latencies.append((time.perf_counter() - t) * 1000)
    rows.append(r)
    n += 1
cap.release()

lats = np.array(latencies)
bvps = np.array([r.bvp for r in rows])
hr_series = np.array([r.hr for r in rows if not np.isnan(r.hr)])

# Build GT-aligned BVP at 30Hz (pipeline emits one BVP per non-skipped frame).
bvps_30 = bvps[~np.isnan(bvps)]
bio = pd.read_csv(BIO)
gt_ppg = bio['ppg'].values.astype(np.float32)
gt_30 = gt_ppg[::int(round(src_fps/30))]

def bandpass(x, fs=30, low=0.7, high=3.5):
    b, a = sig.butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    return sig.filtfilt(b, a, x)

def hr_welch(x, fs=30):
    f, p = sig.welch(x, fs=fs, nperseg=min(len(x), 256))
    m = (f >= 0.7) & (f <= 3.5)
    return float(f[m][np.argmax(p[m])] * 60)

# Skip 5s warmup
SKIP = 30 * 5
L = min(len(bvps_30) - SKIP, len(gt_30) - SKIP)
b = bandpass(bvps_30[SKIP:SKIP+L]); b = np.clip((b-b.mean())/(b.std()+1e-8), -3, 3)
g = bandpass(gt_30[SKIP:SKIP+L]);   g = np.clip((g-g.mean())/(g.std()+1e-8), -3, 3)
r_pearson, _ = pearsonr(b, g)
hr_pred = hr_welch(b)
hr_gt = hr_welch(g)

print('-' * 60)
print(f'Frames processed: {n}  emitted BVPs: {len(bvps_30)}')
print(f'Per-frame pipeline latency: mean {lats.mean():.1f} ms  P95 {np.percentile(lats, 95):.1f} ms  max {lats.max():.1f} ms')
print(f'HR pred: {hr_pred:5.1f} bpm  |  GT: {hr_gt:5.1f} bpm  |  |Δ| = {abs(hr_pred-hr_gt):.1f} bpm')
print(f'Pearson r (rPPG vs GT PPG): {r_pearson:+.3f}')
va = next((r for r in rows[::-1] if r.valence is not None), None)
if va: print(f'Last VA: V={va.valence:+.2f}  A={va.arousal:+.2f}')

assert abs(hr_pred - hr_gt) < 3.0, f'HR error too large: {abs(hr_pred-hr_gt):.1f} bpm'
assert r_pearson > 0.3, f'Pearson r too low: {r_pearson:.3f}'
assert lats.mean() < 100, f'Pipeline too slow: {lats.mean():.1f} ms/frame'
print('OK - smoke test passed (accuracy + latency)')
