"""Non-GUI smoke test: warm up FaceMesh + pipeline and run a few steps on a dummy frame.

This avoids opening the webcam while validating the integration.
"""
import time
from pathlib import Path
import numpy as np

from stateguard import StateGuardPipeline, StateGuardConfig


def main():
    w = Path(__file__).parent.parent / 'stateguard' / 'weights'
    cfg = StateGuardConfig(
        facephys_path=str(w / 'step.onnx'),
        va_path=str(w / 'va_mbf.onnx'),
        fatigue_path=str(w / 'fatigue.onnx') if (w / 'fatigue.onnx').exists() else None,
        state_path=None,
        fps=30.0,
        source_fps=30.0,
    )
    print('Creating pipeline...')
    pipe = StateGuardPipeline(cfg)
    print('Pipeline created. Warming up...')
    # warm-up a few steps with a dummy RGB frame
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(3):
        r = pipe.step(dummy)
        print(f'step {i}: hr={r.hr:.2f} rmssd={r.rmssd:.2f} q={r.quality:.2f} valence={r.valence} arousal={r.arousal} landmarks={'None' if r.landmarks is None else len(r.landmarks)}')
        time.sleep(0.2)
    print('Smoke test finished.')


if __name__ == '__main__':
    main()
