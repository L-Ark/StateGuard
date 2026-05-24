"""Rule-based five-region spectrum prediction.

This module fuses the current rPPG / HRV / landmark-derived metrics into
one of five high-level states:
    - routine
    - flow
  - overload
  - distraction
    - exhaustion

The goal is a stable, real-time estimate, not an intervention engine.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional
import time

import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _normalize(value: Optional[float], low: float, high: float, *, default: float = 0.5) -> float:
    value = _finite(value)
    if value is None:
        return default
    if abs(high - low) < 1e-6:
        return default
    return _clip01((value - low) / (high - low))


def _inverse_normalize(value: Optional[float], low: float, high: float, *, default: float = 0.5) -> float:
    return 1.0 - _normalize(value, low, high, default=default)


SPECTRUM_CENTER_RATIO = 0.34


@dataclass
class QuadrantPrediction:
    key: str
    label: str
    activation: float
    depletion: float
    confidence: float
    scores: Dict[str, float]
    reason: str
    x_axis: str = 'cognitive activation'
    y_axis: str = 'physiological depletion'

    @property
    def x(self) -> float:
        return self.activation

    @property
    def y(self) -> float:
        return self.depletion


class QuadrantClassifier:
    """Stable five-region spectrum predictor.

    The classifier uses two latent axes:
    - activation: high means the user is cognitively engaged / aroused
    - depletion: high means the user is physiologically worn down

    The regions are then mapped as:
    - routine:     close to the center anchor
    - flow:        high activation, low depletion
    - overload:    high activation, high depletion
    - distraction: low activation, low depletion
    - exhaustion:  low activation, high depletion
    """

    LABELS = {
        'routine': '常态工作区 / 稳态',
        'flow': '深心流 / 高效专注',
        'overload': '认知过载 / 焦虑焦躁',
        'distraction': '注意力涣散 / 走神散漫',
        'exhaustion': '生理耗尽 / 疲惫',
    }

    def __init__(
        self,
        window_sec: float = 12.0,
        max_fps: float = 30.0,
        smoothing: float = 0.25,
        min_samples: int = 24,
    ) -> None:
        self.window_sec = float(window_sec)
        self.maxlen = max(8, int(round(self.window_sec * max_fps)))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.min_samples = max(1, int(min_samples))
        self._history: Deque[dict[str, float]] = deque(maxlen=self.maxlen)
        self._smoothed_activation: Optional[float] = None
        self._smoothed_depletion: Optional[float] = None
        self._last_key: str = 'routine'
        self._last_prediction = QuadrantPrediction(
            key='routine',
            label=self.LABELS['routine'],
            activation=0.0,
            depletion=0.0,
            confidence=0.0,
            scores={k: 0.0 for k in self.LABELS},
            reason='等待稳定信号',
        )

    def reset(self) -> None:
        self._history.clear()
        self._smoothed_activation = None
        self._smoothed_depletion = None
        self._last_key = 'routine'
        self._last_prediction = QuadrantPrediction(
            key='routine',
            label=self.LABELS['routine'],
            activation=0.0,
            depletion=0.0,
            confidence=0.0,
            scores={k: 0.0 for k in self.LABELS},
            reason='等待稳定信号',
        )

    def update(self, result: Any, calibrator: Optional[Any] = None) -> Optional[QuadrantPrediction]:
        sample = self._build_sample(result, calibrator)
        self._history.append(sample)
        if not self._is_ready():
            wait_reason = self._waiting_reason()
            self._last_prediction = QuadrantPrediction(
                key='',
                label=wait_reason,
                activation=0.0,
                depletion=0.0,
                confidence=0.0,
                scores={k: 0.0 for k in self.LABELS},
                reason=wait_reason,
            )
            return None
        stats = self._aggregate(self._history)

        activation_raw = self._compute_activation(stats)
        depletion_raw = self._compute_depletion(stats)

        if self._smoothed_activation is None:
            self._smoothed_activation = activation_raw
        else:
            self._smoothed_activation = (
                self.smoothing * activation_raw + (1.0 - self.smoothing) * self._smoothed_activation
            )

        if self._smoothed_depletion is None:
            self._smoothed_depletion = depletion_raw
        else:
            self._smoothed_depletion = (
                self.smoothing * depletion_raw + (1.0 - self.smoothing) * self._smoothed_depletion
            )

        activation = _clip01(self._smoothed_activation)
        depletion = _clip01(self._smoothed_depletion)

        scores = self._compute_scores(stats, activation, depletion)
        best_key = self._region_key(activation, depletion)
        best_score = scores[best_key]
        ranked = sorted(scores.values(), reverse=True)
        second_score = ranked[1] if len(ranked) > 1 else 0.0

        confidence = _clip01((best_score - second_score) * 2.0)
        quality = _clip01(stats['quality'])
        confidence = _clip01(confidence * (0.55 + 0.45 * quality))

        reason = self._reason_for(best_key, stats, activation, depletion)
        prediction = QuadrantPrediction(
            key=best_key,
            label=self.LABELS[best_key],
            activation=activation,
            depletion=depletion,
            confidence=confidence,
            scores=scores,
            reason=reason,
        )
        self._last_key = best_key
        self._last_prediction = prediction
        return prediction

    def _region_key(self, activation: float, depletion: float) -> str:
        if float(np.hypot(activation - 0.5, depletion - 0.5)) <= SPECTRUM_CENTER_RATIO:
            return 'routine'
        if activation >= 0.5 and depletion < 0.5:
            return 'flow'
        if activation >= 0.5 and depletion >= 0.5:
            return 'overload'
        if activation < 0.5 and depletion < 0.5:
            return 'distraction'
        return 'exhaustion'

    def _is_ready(self) -> bool:
        if len(self._history) < self.min_samples:
            return False
        if not self._history:
            return False
        finite_sources = 0
        for key in ('hr', 'rmssd', 'sdnn', 'quality', 'perclos', 'blink_rate', 'yawn_rate', 'fatigue', 'fatigue_landmark', 'arousal', 'valence'):
            values = [item[key] for item in self._history if item.get(key) is not None and np.isfinite(item[key])]
            if values:
                finite_sources += 1
        return finite_sources >= 2

    def _waiting_reason(self) -> str:
        if len(self._history) < self.min_samples:
            return f'等待稳定信号（帧数 {len(self._history)}/{self.min_samples}）'

        available = []
        for key in ('hr', 'rmssd', 'sdnn', 'quality', 'perclos', 'blink_rate', 'yawn_rate', 'fatigue', 'fatigue_landmark', 'arousal', 'valence'):
            values = [item[key] for item in self._history if item.get(key) is not None and np.isfinite(item[key])]
            if values:
                available.append(key)

        if not available:
            return '等待稳定信号（尚无有效特征）'
        return f'等待稳定信号（有效特征 {len(available)}/2：{", ".join(available[:4])})'

    def current(self) -> QuadrantPrediction:
        return self._last_prediction

    def _build_sample(self, result: Any, calibrator: Optional[Any]) -> Dict[str, float]:
        sample = {
            'hr': _finite(getattr(result, 'hr', None)),
            'rmssd': _finite(getattr(result, 'rmssd', None)),
            'sdnn': _finite(getattr(result, 'sdnn', None)),
            'quality': _finite(getattr(result, 'quality', None)) or 0.0,
            'valence': _finite(getattr(result, 'valence', None)),
            'arousal': _finite(getattr(result, 'arousal', None)),
            'fatigue': _finite(getattr(result, 'fatigue', None)),
            'perclos': _finite(getattr(result, 'perclos', None)),
            'blink_rate': _finite(getattr(result, 'blink_rate', None)),
            'yawn_rate': _finite(getattr(result, 'yawn_rate', None)),
            'fatigue_landmark': _finite(getattr(result, 'fatigue_landmark', None)),
        }

        if calibrator is not None and getattr(calibrator, 'finished', False):
            try:
                sample['fatigue'] = _finite(calibrator.calibrated_value(sample['fatigue']))
                sample['perclos'] = _finite(
                    calibrator.calibrated_scalar_value(
                        sample['perclos'], calibrator.perclos_baseline, calibrator.perclos_spread
                    )
                )
                sample['blink_rate'] = _finite(
                    calibrator.calibrated_scalar_value(
                        sample['blink_rate'], calibrator.blink_baseline, calibrator.blink_spread
                    )
                )
                sample['yawn_rate'] = _finite(
                    calibrator.calibrated_scalar_value(
                        sample['yawn_rate'], calibrator.yawn_baseline, calibrator.yawn_spread
                    )
                )
                sample['fatigue_landmark'] = _finite(
                    calibrator.calibrated_scalar_value(
                        sample['fatigue_landmark'],
                        calibrator.lmk_fatigue_baseline,
                        calibrator.lmk_fatigue_spread,
                    )
                )
            except Exception:
                pass

        return sample

    def _aggregate(self, history: Deque[Dict[str, float]]) -> Dict[str, float]:
        keys = history[0].keys() if history else []
        stats: Dict[str, float] = {}
        for key in keys:
            values = [item[key] for item in history if item.get(key) is not None and np.isfinite(item[key])]
            stats[key] = float(np.median(values)) if values else float('nan')
        return stats

    def _compute_activation(self, stats: Dict[str, float]) -> float:
        hr_high = _normalize(stats.get('hr'), 58.0, 98.0)
        arousal_high = _normalize(stats.get('arousal'), -0.2, 0.8)
        blink_suppressed = _inverse_normalize(stats.get('blink_rate'), 8.0, 22.0)
        quality = _normalize(stats.get('quality'), 0.3, 0.9)
        return _clip01(0.35 * arousal_high + 0.30 * hr_high + 0.20 * blink_suppressed + 0.15 * quality)

    def _compute_depletion(self, stats: Dict[str, float]) -> float:
        perclos_high = _normalize(stats.get('perclos'), 0.08, 0.35)
        fatigue_high = _normalize(stats.get('fatigue'), 0.25, 0.80)
        lmk_fatigue_high = _normalize(stats.get('fatigue_landmark'), 0.20, 0.80)
        rmssd_low = _inverse_normalize(stats.get('rmssd'), 18.0, 60.0)
        yawn_high = _normalize(stats.get('yawn_rate'), 0.4, 2.0)
        return _clip01(
            0.30 * perclos_high + 0.25 * fatigue_high + 0.15 * lmk_fatigue_high + 0.20 * rmssd_low + 0.10 * yawn_high
        )

    def _compute_scores(self, stats: Dict[str, float], activation: float, depletion: float) -> Dict[str, float]:
        valence = stats.get('valence')
        arousal = stats.get('arousal')
        valence_positive = _normalize(valence, 0.0, 0.8)
        valence_negative = _normalize(-valence if valence is not None and np.isfinite(valence) else None, 0.0, 0.8)
        arousal_low = _inverse_normalize(arousal, -0.2, 0.6)
        rmssd_high = _normalize(stats.get('rmssd'), 18.0, 60.0)
        perclos_high = _normalize(stats.get('perclos'), 0.08, 0.35)
        fatigue_high = _normalize(stats.get('fatigue'), 0.25, 0.80)
        center_distance = float(np.hypot(activation - 0.5, depletion - 0.5))
        routine = _clip01(1.0 - center_distance / max(0.05, SPECTRUM_CENTER_RATIO))

        flow = activation * (1.0 - depletion) * (0.70 + 0.30 * rmssd_high) * (0.75 + 0.25 * valence_positive)
        overload = activation * depletion * (0.70 + 0.30 * valence_negative) * (0.70 + 0.30 * (1.0 - rmssd_high))
        distraction = (1.0 - activation) * (1.0 - depletion) * (0.75 + 0.25 * arousal_low) * (0.70 + 0.30 * (1.0 - valence_negative))
        exhaustion = (1.0 - activation) * depletion * (0.75 + 0.25 * perclos_high) * (0.75 + 0.25 * fatigue_high)

        return {
            'routine': routine,
            'flow': _clip01(flow),
            'overload': _clip01(overload),
            'distraction': _clip01(distraction),
            'exhaustion': _clip01(exhaustion),
        }

    def _reason_for(self, key: str, stats: Dict[str, float], activation: float, depletion: float) -> str:
        hr = stats.get('hr')
        rmssd = stats.get('rmssd')
        perclos = stats.get('perclos')
        blink = stats.get('blink_rate')
        valence = stats.get('valence')
        arousal = stats.get('arousal')
        fatigue = stats.get('fatigue')

        if key == 'routine':
            return (
                f'常态工作区，中心半径内（r<{SPECTRUM_CENTER_RATIO:.2f}）；'
                f'HR={self._fmt(hr)} / RMSSD={self._fmt(rmssd)} / Blink={self._fmt(blink)}'
            )
        if key == 'flow':
            return (
                f'激活高({activation:.2f})、耗竭低({depletion:.2f})；'
                f'HR={self._fmt(hr)} / RMSSD={self._fmt(rmssd)} / Blink={self._fmt(blink)}'
            )
        if key == 'overload':
            return (
                f'激活高({activation:.2f})且耗竭上升({depletion:.2f})；'
                f'Valence={self._fmt(valence)} / Arousal={self._fmt(arousal)} / RMSSD={self._fmt(rmssd)}'
            )
        if key == 'exhaustion':
            return (
                f'激活低({activation:.2f})、耗竭高({depletion:.2f})；'
                f'PERCLOS={self._fmt(perclos)} / Fatigue={self._fmt(fatigue)} / HR={self._fmt(hr)}'
            )
        return (
            f'激活低({activation:.2f})、耗竭低({depletion:.2f})；'
            f'Arousal={self._fmt(arousal)} / Blink={self._fmt(blink)} / Valence={self._fmt(valence)}'
        )

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        value = _finite(value)
        if value is None:
            return '--'
        return f'{value:.2f}'



# Public spectrum names used by the app; keep legacy quadrant names available too.
SpectrumPrediction = QuadrantPrediction
SpectrumClassifier = QuadrantClassifier
