"""Streaming HR / HRV from per-frame BVP samples.

Maintains a rolling window (default 30s) and recomputes HR/RMSSD/SDNN on
demand. Lightweight enough to call once per second.
"""
from collections import deque
from typing import Tuple

import numpy as np
from scipy import signal as sig


class HRVStream:
    def __init__(self, fps: float = 30.0, window_sec: float = 30.0, warmup_sec: float = 12.0) -> None:
        self.fps = float(fps)
        self.window_sec = float(window_sec)
        self.warmup_sec = float(warmup_sec)
        self._buf: deque = deque(maxlen=int(self.fps * self.window_sec))
        # cached bandpass filter
        self._b, self._a = sig.butter(
            3, [0.7 / (self.fps / 2), 3.5 / (self.fps / 2)], btype='band'
        )

    def push(self, bvp: float) -> None:
        self._buf.append(float(bvp))

    def __len__(self) -> int:
        return len(self._buf)

    def filtered(self) -> np.ndarray:
        if len(self._buf) < int(self.fps * self.warmup_sec):
            return np.array([], dtype=np.float32)
        x = np.asarray(self._buf, dtype=np.float32)
        try:
            x = sig.filtfilt(self._b, self._a, x)
        except Exception:
            return np.array([], dtype=np.float32)
        s = x.std() + 1e-8
        return np.clip((x - x.mean()) / s, -3, 3).astype(np.float32)

    def estimate(self) -> Tuple[float, float, float, float]:
        """Returns (HR_bpm, RMSSD_ms, SDNN_ms, signal_quality 0-1).

        HR comes from the Welch power spectrum (robust under noise).
        RMSSD/SDNN use peak-detected R-R intervals, but only when the
        detected R-R median agrees with the Welch peak — otherwise the
        peaks are likely spurious and HRV is reported as NaN.

        Quality blends spectral concentration and R-R agreement; it is
        NOT just a count ratio (which would inflate under noise).
        """
        x = self.filtered()
        if x.size < int(self.fps * self.warmup_sec):
            return float('nan'), float('nan'), float('nan'), 0.0

        # 1) HR from Welch — primary, drift-resistant
        freqs, psd = sig.welch(x, fs=self.fps, nperseg=min(len(x), 256))
        mask = (freqs >= 0.7) & (freqs <= 3.5)
        if not mask.sum():
            return float('nan'), float('nan'), float('nan'), 0.0
        psd_hr = psd[mask]; freqs_hr = freqs[mask]
        peak_idx = int(np.argmax(psd_hr))
        hr_welch = float(freqs_hr[peak_idx] * 60)
        # spectral concentration: fraction of in-band energy near the peak (±0.3 Hz)
        f_peak = freqs_hr[peak_idx]
        near = (freqs_hr >= f_peak - 0.3) & (freqs_hr <= f_peak + 0.3)
        spectral_q = float(psd_hr[near].sum() / (psd_hr.sum() + 1e-12))

        # 2) HRV from R-R intervals — only trust when consistent with Welch.
        # Stricter prominence (median absolute amplitude * 0.5) makes us
        # less likely to pick up high-freq noise peaks.
        prom = float(np.median(np.abs(x)) * 0.5 + 1e-6)
        peaks, _ = sig.find_peaks(x, distance=int(self.fps * 0.45), prominence=prom)
        rmssd = float('nan'); sdnn = float('nan'); rr_q = 0.0
        if peaks.size >= 4:
            rr = np.diff(peaks) / self.fps * 1000.0  # ms
            rr = rr[(rr >= 350) & (rr <= 1500)]
            if rr.size >= 3:
                hr_rr = 60000.0 / float(np.median(rr))
                # require RR-derived HR to be within 8 bpm of Welch peak
                if abs(hr_rr - hr_welch) <= 8.0:
                    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
                    sdnn = float(rr.std())
                    rr_q = float(np.clip(rr.size / max(1, self.window_sec * hr_welch / 60), 0, 1))

        # combined quality: spectral concentration is the most reliable signal
        quality = float(np.clip(0.7 * spectral_q + 0.3 * rr_q, 0, 1))
        return hr_welch, rmssd, sdnn, quality
