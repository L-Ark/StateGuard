from __future__ import annotations

import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from stateguard.pipeline import StateGuardConfig, StateGuardPipeline
from stateguard.quadrants import SPECTRUM_CENTER_RATIO, SpectrumClassifier, SpectrumPrediction


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / 'stateguard' / 'weights'


def _default_model_path(name: str) -> str:
    return str(WEIGHTS_DIR / name)


def _fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return '--'
    try:
        if np.isnan(value):
            return '--'
    except TypeError:
        pass
    return f'{float(value):.{decimals}f}'


def measure_capture_fps(cap: cv2.VideoCapture, max_seconds: float = 3.0, sample_frames: int = 90) -> float:
    start = time.time()
    frames = 0
    while frames < sample_frames and (time.time() - start) < max_seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.time() - start
    return frames / elapsed if elapsed > 0 else 0.0


class SpectrumDashboardWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._prediction: Optional[SpectrumPrediction] = None
        self.setMinimumHeight(360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_prediction(self, prediction: Optional[SpectrumPrediction]) -> None:
        self._prediction = prediction
        self.update()

    def clear_prediction(self) -> None:
        self._prediction = None
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(520, 420)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor('#0b1020'))

        panel = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor('#111827'))
        painter.drawRoundedRect(panel, 18, 18)

        plot = panel.adjusted(26, 24, -26, -72)
        if plot.width() < 40 or plot.height() < 40:
            return

        palette = {
            'routine': QtGui.QColor('#2563eb'),
            'flow': QtGui.QColor('#16a34a'),
            'overload': QtGui.QColor('#d73a49'),
            'distraction': QtGui.QColor('#f59e0b'),
            'exhaustion': QtGui.QColor('#7c3aed'),
        }

        def _fill_region(x: int, y: int, w: int, h: int, color: QtGui.QColor) -> None:
            rgba = QtGui.QColor(color)
            rgba.setAlpha(70)
            painter.fillRect(QtCore.QRect(x, y, w, h), rgba)

        half_w = plot.width() // 2
        half_h = plot.height() // 2
        _fill_region(plot.left(), plot.top(), half_w, half_h, palette['overload'])
        _fill_region(plot.left() + half_w, plot.top(), plot.width() - half_w, half_h, palette['flow'])
        _fill_region(plot.left(), plot.top() + half_h, half_w, plot.height() - half_h, palette['distraction'])
        _fill_region(plot.left() + half_w, plot.top() + half_h, plot.width() - half_w, plot.height() - half_h, palette['exhaustion'])

        routine_color = QtGui.QColor(palette['routine'])
        routine_color.setAlpha(115)
        center = plot.center()
        center_radius = int(min(plot.width(), plot.height()) * SPECTRUM_CENTER_RATIO)
        painter.setBrush(routine_color)
        painter.setPen(QtGui.QPen(QtGui.QColor('#dbeafe'), 3))
        painter.drawEllipse(center, center_radius, center_radius)

        painter.setPen(QtGui.QPen(QtGui.QColor('#94a3b8'), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(plot)
        painter.drawLine(plot.center().x(), plot.top(), plot.center().x(), plot.bottom())
        painter.drawLine(plot.left(), plot.center().y(), plot.right(), plot.center().y())

        label_pen = QtGui.QPen(QtGui.QColor('#e2e8f0'))
        painter.setPen(label_pen)
        label_font = painter.font()
        label_font.setPointSize(max(9, label_font.pointSize() - 1))
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.drawText(plot.left() + 12, plot.top() + 24, '认知过载')
        painter.drawText(plot.right() - 72, plot.top() + 24, '深度心流')
        painter.drawText(plot.left() + 12, plot.bottom() - 12, '注意力涣散')
        painter.drawText(plot.right() - 72, plot.bottom() - 12, '生理耗尽')
        text_rect = QtCore.QRectF(
            center.x() - center_radius,
            center.y() - center_radius,
            center_radius * 2,
            center_radius * 2,
        )
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, '常态工作区')

        axis_font = painter.font()
        axis_font.setPointSize(max(8, axis_font.pointSize() - 2))
        axis_font.setBold(False)
        painter.setFont(axis_font)
        painter.setPen(QtGui.QPen(QtGui.QColor('#cbd5e1')))
        painter.drawText(plot.left(), panel.bottom() - 20, 'X: 行为专注度')
        painter.drawText(panel.right() - 128, panel.bottom() - 20, 'Y: 生理激活度')

        if self._prediction is None or not getattr(self._prediction, 'key', ''):
            painter.setPen(QtGui.QPen(QtGui.QColor('#94a3b8')))
            painter.drawText(plot.adjusted(0, 0, -12, -12), QtCore.Qt.AlignCenter, '等待稳定信号')
            return

        x = float(np.clip(getattr(self._prediction, 'x', 0.5), 0.0, 1.0))
        y = float(np.clip(getattr(self._prediction, 'y', 0.5), 0.0, 1.0))
        point_x = plot.left() + x * plot.width()
        point_y = plot.bottom() - y * plot.height()

        point_color = QtGui.QColor(palette.get(getattr(self._prediction, 'key', ''), QtGui.QColor('#e5e7eb')))
        painter.setPen(QtGui.QPen(QtGui.QColor('white'), 2))
        painter.setBrush(point_color)
        painter.drawEllipse(QtCore.QPointF(point_x, point_y), 9, 9)
        painter.setPen(QtGui.QPen(point_color, 1, QtCore.Qt.DashLine))
        painter.drawLine(QtCore.QPointF(point_x, plot.top()), QtCore.QPointF(point_x, plot.bottom()))
        painter.drawLine(QtCore.QPointF(plot.left(), point_y), QtCore.QPointF(plot.right(), point_y))

        info = f'{self._prediction.label}   x={x:.2f}   y={y:.2f}'
        info_rect = QtCore.QRect(plot.left(), panel.bottom() - 54, plot.width(), 22)
        painter.setPen(QtGui.QPen(QtGui.QColor('#e2e8f0')))
        painter.drawText(info_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, info)



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
    perclos_samples: list[float] = field(default_factory=list)
    blink_rate_samples: list[float] = field(default_factory=list)
    yawn_rate_samples: list[float] = field(default_factory=list)
    lmk_fatigue_samples: list[float] = field(default_factory=list)
    finished: bool = False
    baseline: Optional[float] = None
    spread: Optional[float] = None
    perclos_baseline: Optional[float] = None
    perclos_spread: Optional[float] = None
    blink_baseline: Optional[float] = None
    blink_spread: Optional[float] = None
    yawn_baseline: Optional[float] = None
    yawn_spread: Optional[float] = None
    lmk_fatigue_baseline: Optional[float] = None
    lmk_fatigue_spread: Optional[float] = None

    def reset(self) -> None:
        self.started_at = time.time()
        self.last_sample_at = -1e9
        self.samples.clear()
        self.qualities.clear()
        self.perclos_samples.clear()
        self.blink_rate_samples.clear()
        self.yawn_rate_samples.clear()
        self.lmk_fatigue_samples.clear()
        self.finished = False
        self.baseline = None
        self.spread = None
        self.perclos_baseline = None
        self.perclos_spread = None
        self.blink_baseline = None
        self.blink_spread = None
        self.yawn_baseline = None
        self.yawn_spread = None
        self.lmk_fatigue_baseline = None
        self.lmk_fatigue_spread = None

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def remaining(self) -> float:
        return max(0.0, self.duration_sec - self.elapsed())

    def add_sample(
        self,
        fatigue: Optional[float],
        perclos: Optional[float],
        blink_rate: Optional[float],
        yawn_rate: Optional[float],
        lmk_fatigue: Optional[float],
        quality: float,
    ) -> None:
        if self.finished:
            return
        now = time.time()
        if (now - self.last_sample_at) < self.sample_every_sec:
            return
        if fatigue is not None and np.isfinite(fatigue):
            self.samples.append(float(fatigue))
        self.qualities.append(float(quality) if np.isfinite(quality) else float('nan'))
        if perclos is not None and np.isfinite(perclos):
            self.perclos_samples.append(float(perclos))
        if blink_rate is not None and np.isfinite(blink_rate):
            self.blink_rate_samples.append(float(blink_rate))
        if yawn_rate is not None and np.isfinite(yawn_rate):
            self.yawn_rate_samples.append(float(yawn_rate))
        if lmk_fatigue is not None and np.isfinite(lmk_fatigue):
            self.lmk_fatigue_samples.append(float(lmk_fatigue))
        self.last_sample_at = now

    def maybe_finish(self) -> bool:
        if self.finished:
            return True
        if self.elapsed() < self.duration_sec:
            return False

        def _set_baseline(values: list[float], default_base: float, default_spread: float) -> tuple[float, float]:
            if values:
                arr = np.asarray(values, dtype=np.float32)
                return float(np.median(arr)), float(np.std(arr) + 1e-6)
            return default_base, default_spread

        self.baseline, self.spread = _set_baseline(self.samples, 0.5, 0.25)
        self.perclos_baseline, self.perclos_spread = _set_baseline(self.perclos_samples, 0.2, 0.1)
        self.blink_baseline, self.blink_spread = _set_baseline(self.blink_rate_samples, 15.0, 5.0)
        self.yawn_baseline, self.yawn_spread = _set_baseline(self.yawn_rate_samples, 0.5, 0.5)
        self.lmk_fatigue_baseline, self.lmk_fatigue_spread = _set_baseline(self.lmk_fatigue_samples, 0.3, 0.2)
        self.finished = True
        return True

    def calibrated_value(self, raw_value: Optional[float]) -> Optional[float]:
        if raw_value is None or np.isnan(raw_value):
            return None
        if not self.finished or self.baseline is None:
            return float(raw_value)
        scale = max(0.18, (self.spread or 0.0) * 3.0)
        return float(np.clip(0.5 + (float(raw_value) - self.baseline) / scale, 0.0, 1.0))

    def calibrated_scalar_value(
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
        return float(np.clip(0.5 + (float(raw_value) - baseline) / scale, 0.0, 1.0))

    def confidence_text(self) -> str:
        if not self.finished:
            return '校准中'
        if len(self.samples) < self.min_samples:
            return f'低置信度（{len(self.samples)} 个样本）'
        valid_q = [q for q in self.qualities if np.isfinite(q)]
        if valid_q and float(np.mean(valid_q)) < self.min_quality:
            return f'低置信度（{len(self.samples)} 个样本，质量偏低）'
        return f'就绪（{len(self.samples)} 个样本）'


class CameraPipelineWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(object, object, object)
    status_changed = QtCore.Signal(str)
    calibration_changed = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        camera_index: int,
        facephys_path: str,
        va_path: str,
        fatigue_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.facephys_path = facephys_path
        self.va_path = va_path
        self.fatigue_path = fatigue_path
        self._stop = False
        self._calibrator: Optional[FatigueCalibrator] = None
        self._show_landmarks = True
        self._show_face_box = True
        self._last_calib_second: Optional[int] = None
        self._quadrant = SpectrumClassifier(window_sec=12.0, max_fps=30.0, smoothing=0.25)
        self._sampling_profile = 'routine'
        self._sampling_last_switch_at = 0.0

    def request_calibration(self, duration_sec: float, min_quality: float, min_samples: int, sample_every_sec: float) -> None:
        self._calibrator = FatigueCalibrator(
            duration_sec=max(0.0, float(duration_sec)),
            min_quality=float(min_quality),
            min_samples=max(1, int(min_samples)),
            sample_every_sec=max(0.5, float(sample_every_sec)),
        )
        self.calibration_changed.emit(
            f'校准开始：{self._calibrator.duration_sec:.0f}s，间隔 {self._calibrator.sample_every_sec:.1f}s'
        )

    def stop(self) -> None:
        self._stop = True

    def set_display_options(self, show_landmarks: bool, show_face_box: bool) -> None:
        self._show_landmarks = bool(show_landmarks)
        self._show_face_box = bool(show_face_box)

    def _desired_sampling_profile(self, quadrant: object) -> str:
        if quadrant is None or not getattr(quadrant, 'key', None):
            return 'routine'
        if quadrant.key == 'routine':
            return 'routine'
        if float(getattr(quadrant, 'confidence', 0.0)) >= 0.45:
            return 'boosted'
        return 'routine'

    @staticmethod
    def _draw_landmarks(image_bgr: np.ndarray, landmarks, face_box=None) -> None:
        if not landmarks:
            return
        h, w = image_bgr.shape[:2]
        if face_box:
            x1, y1, x2, y2 = face_box
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
        else:
            x1 = y1 = 0
            bw = w
            bh = h
        for lm in landmarks:
            x = int(x1 + float(lm['x']) * bw)
            y = int(y1 + float(lm['y']) * bh)
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 255), -1)

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()

    @staticmethod
    def _open_camera(camera_index: int) -> tuple[Optional[cv2.VideoCapture], str]:
        backends = [
            ('DSHOW', getattr(cv2, 'CAP_DSHOW', None)),
            ('MSMF', getattr(cv2, 'CAP_MSMF', None)),
            ('ANY', getattr(cv2, 'CAP_ANY', None)),
        ]
        for name, backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(int(camera_index))
                else:
                    cap = cv2.VideoCapture(int(camera_index), int(backend))
                if cap.isOpened():
                    return cap, name
                cap.release()
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
        return None, 'NONE'

    def run(self) -> None:
        try:
            cap, backend_name = self._open_camera(int(self.camera_index))
            if cap is None or not cap.isOpened():
                self.error.emit(f'无法打开摄像头 {self.camera_index}')
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            measured_fps = measure_capture_fps(cap)
            if measured_fps >= 1.0:
                source_fps = measured_fps
            elif reported_fps >= 1.0:
                source_fps = reported_fps
            else:
                source_fps = 30.0

            target_fps = min(30.0, source_fps)
            if source_fps < 30.0 - 0.5:
                self.status_changed.emit(
                    f'摄像头实际帧率较低({measured_fps:.2f})，管线降到 {target_fps:.1f} fps 以避免上采样。'
                )

            cfg = StateGuardConfig(
                facephys_path=self.facephys_path,
                va_path=self.va_path,
                fatigue_path=self.fatigue_path,
                fps=target_fps,
                source_fps=source_fps,
            )
            pipeline = StateGuardPipeline(cfg)
            self._quadrant.reset()
            pipeline.set_sampling_profile(self._sampling_profile)
            self.status_changed.emit(
                f'摄像头已启动({backend_name}) | reported={reported_fps:.1f} fps | measured={measured_fps:.2f} fps | source={source_fps:.1f} fps | target={target_fps:.1f} fps'
            )

            warmup_deadline = time.time() + 3.0
            warmup_reads = 0

            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    if time.time() <= warmup_deadline:
                        time.sleep(0.05)
                        warmup_reads += 1
                        continue
                    self.error.emit('摄像头已打开，但连续读取失败。请切换设备索引，或检查 Windows 相机权限/占用情况。')
                    break
                warmup_reads += 1
                if warmup_reads <= 3:
                    time.sleep(0.02)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pipeline.step(rgb)

                if self._calibrator is not None:
                    self._calibrator.add_sample(
                        result.fatigue,
                        result.perclos,
                        result.blink_rate,
                        result.yawn_rate,
                        result.fatigue_landmark,
                        result.quality,
                    )
                    if not self._calibrator.finished:
                        remaining = int(round(self._calibrator.remaining()))
                        if remaining != self._last_calib_second:
                            self._last_calib_second = remaining
                            self.calibration_changed.emit(f'校准中：剩余 {remaining}s')
                    if self._calibrator.maybe_finish() and self._calibrator.finished:
                        self.calibration_changed.emit(
                            f'校准完成：{self._calibrator.confidence_text()}'
                        )

                quadrant = self._quadrant.update(result, self._calibrator)
                desired_profile = self._desired_sampling_profile(quadrant)
                if desired_profile != self._sampling_profile:
                    now = time.time()
                    if (now - self._sampling_last_switch_at) >= 1.5:
                        self._sampling_profile = desired_profile
                        self._sampling_last_switch_at = now
                        pipeline.set_sampling_profile(desired_profile)
                        if desired_profile == 'boosted':
                            self.status_changed.emit('检测到离开常态工作区：已切换到高采样档位（更密集的图像模型采样）')
                        else:
                            self.status_changed.emit('回到常态工作区：已切回低采样档位')

                display = frame.copy()
                if self._show_landmarks and result.landmarks:
                    self._draw_landmarks(display, result.landmarks, result.face_box)
                if self._show_face_box and result.face_box:
                    x1, y1, x2, y2 = result.face_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                self.frame_ready.emit(self._to_qimage(display), result, quadrant)
                time.sleep(max(0.0, 1.0 / max(target_fps, 1.0) * 0.2))

            cap.release()
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('StateGuard')
        self.resize(1400, 860)
        self.worker: Optional[CameraPipelineWorker] = None
        self._pending_model_reload = False
        self._recording = False
        self._record_fp = None
        self._record_path: Optional[Path] = None
        self._last_record_second: Optional[int] = None
        self._last_record_key: Optional[str] = None
        self._quadrant_ui_key: Optional[str] = None
        self._quadrant_ui_candidate_key: Optional[str] = None
        self._quadrant_ui_candidate_seen_at: float = 0.0
        self._quadrant_ui_last_update_at: float = 0.0
        self._quadrant_ui_last_confidence: float = -1.0
        self._quadrant_ui_waiting_shown: bool = False
        self._calibration_ui_text: str = '等待校准'
        self._calibration_ui_active: bool = False

        self._build_ui()
        self._load_default_models()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.video_label = QtWidgets.QLabel('摄像头未启动')
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet('background: #111; color: #ddd; border: 1px solid #333;')
        root.addWidget(self.video_label, 3)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(QtCore.Qt.ElideRight)
        root.addWidget(tabs, 2)

        self._overview_page = QtWidgets.QWidget()
        self._overview_page_layout = QtWidgets.QVBoxLayout(self._overview_page)
        self._overview_page_layout.setContentsMargins(0, 0, 0, 0)
        self._overview_page_layout.setSpacing(0)
        self._overview_scroll = QtWidgets.QScrollArea()
        self._overview_scroll.setWidgetResizable(True)
        self._overview_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._overview_content = QtWidgets.QWidget()
        self._overview_layout = QtWidgets.QVBoxLayout(self._overview_content)
        self._overview_layout.setContentsMargins(0, 0, 0, 0)
        self._overview_layout.setSpacing(10)
        self._overview_scroll.setWidget(self._overview_content)
        self._overview_page_layout.addWidget(self._overview_scroll)

        self._metrics_page = QtWidgets.QWidget()
        self._metrics_page_layout = QtWidgets.QVBoxLayout(self._metrics_page)
        self._metrics_page_layout.setContentsMargins(0, 0, 0, 0)
        self._metrics_page_layout.setSpacing(0)
        self._metrics_scroll = QtWidgets.QScrollArea()
        self._metrics_scroll.setWidgetResizable(True)
        self._metrics_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._metrics_content = QtWidgets.QWidget()
        self._metrics_layout = QtWidgets.QVBoxLayout(self._metrics_content)
        self._metrics_layout.setContentsMargins(0, 0, 0, 0)
        self._metrics_layout.setSpacing(10)
        self._metrics_scroll.setWidget(self._metrics_content)
        self._metrics_page_layout.addWidget(self._metrics_scroll)

        self._quadrant_page = QtWidgets.QWidget()
        self._quadrant_page_layout = QtWidgets.QVBoxLayout(self._quadrant_page)
        self._quadrant_page_layout.setContentsMargins(0, 0, 0, 0)
        self._quadrant_page_layout.setSpacing(0)
        self._quadrant_scroll = QtWidgets.QScrollArea()
        self._quadrant_scroll.setWidgetResizable(True)
        self._quadrant_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._quadrant_content = QtWidgets.QWidget()
        self._quadrant_layout = QtWidgets.QVBoxLayout(self._quadrant_content)
        self._quadrant_layout.setContentsMargins(0, 0, 0, 0)
        self._quadrant_layout.setSpacing(10)
        self._quadrant_scroll.setWidget(self._quadrant_content)
        self._quadrant_page_layout.addWidget(self._quadrant_scroll)

        camera_box = QtWidgets.QGroupBox('摄像头')
        camera_form = QtWidgets.QFormLayout(camera_box)
        self.camera_index = QtWidgets.QSpinBox()
        self.camera_index.setRange(0, 10)
        self.camera_index.setValue(0)
        self.show_landmarks = QtWidgets.QCheckBox('显示 landmark')
        self.show_landmarks.setChecked(True)
        self.show_face_box = QtWidgets.QCheckBox('显示人脸框')
        self.show_face_box.setChecked(True)
        self.start_btn = QtWidgets.QPushButton('开始')
        self.stop_btn = QtWidgets.QPushButton('停止')
        self.stop_btn.setEnabled(False)
        self.record_btn = QtWidgets.QPushButton('开始记录')
        self.record_btn.setCheckable(True)
        self.record_path_label = QtWidgets.QLabel('未选择记录文件')
        self.record_path_label.setWordWrap(True)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        camera_form.addRow('设备索引', self.camera_index)
        camera_form.addRow(self.show_landmarks)
        camera_form.addRow(self.show_face_box)
        camera_form.addRow(btn_row)
        camera_form.addRow(self.record_btn)
        camera_form.addRow('记录文件', self.record_path_label)
        self._overview_layout.addWidget(camera_box)

        model_box = QtWidgets.QGroupBox('模型选择')
        model_form = QtWidgets.QFormLayout(model_box)
        self.facephys_edit = QtWidgets.QLineEdit()
        self.va_edit = QtWidgets.QLineEdit()
        self.fatigue_edit = QtWidgets.QLineEdit()
        self._add_path_row(model_form, 'FacePhys', self.facephys_edit)
        self._add_path_row(model_form, 'VA', self.va_edit)
        self._add_path_row(model_form, 'Fatigue', self.fatigue_edit)
        self.apply_models_btn = QtWidgets.QPushButton('应用模型')
        model_form.addRow(self.apply_models_btn)
        self._overview_layout.addWidget(model_box)

        calib_box = QtWidgets.QGroupBox('校准')
        calib_form = QtWidgets.QFormLayout(calib_box)
        self.calib_sec = QtWidgets.QDoubleSpinBox()
        self.calib_sec.setRange(0.0, 600.0)
        self.calib_sec.setValue(90.0)
        self.calib_min_quality = QtWidgets.QDoubleSpinBox()
        self.calib_min_quality.setRange(0.0, 1.0)
        self.calib_min_quality.setSingleStep(0.05)
        self.calib_min_quality.setValue(0.45)
        self.calib_min_samples = QtWidgets.QSpinBox()
        self.calib_min_samples.setRange(1, 100)
        self.calib_min_samples.setValue(4)
        self.calib_sample_sec = QtWidgets.QDoubleSpinBox()
        self.calib_sample_sec.setRange(0.5, 60.0)
        self.calib_sample_sec.setValue(5.0)
        self.calib_btn = QtWidgets.QPushButton('开始校准')
        self.calib_status = QtWidgets.QLabel('未校准')
        self.calib_progress = QtWidgets.QProgressBar()
        self.calib_progress.setRange(0, 100)
        self.calib_progress.setValue(0)
        calib_form.addRow('时长（秒）', self.calib_sec)
        calib_form.addRow('最低质量', self.calib_min_quality)
        calib_form.addRow('最少样本', self.calib_min_samples)
        calib_form.addRow('采样间隔（秒）', self.calib_sample_sec)
        calib_form.addRow(self.calib_btn)
        calib_form.addRow('进度', self.calib_progress)
        calib_form.addRow('状态', self.calib_status)
        self._overview_layout.addWidget(calib_box)
        self._overview_layout.addStretch(1)

        metrics_box = QtWidgets.QGroupBox('实时指标')
        metrics_grid = QtWidgets.QGridLayout(metrics_box)
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(8)
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        metric_names = [
            ('hr', 'HR'), ('rmssd', 'RMSSD'), ('sdnn', 'SDNN'), ('quality', 'Quality'),
            ('fatigue', 'Fatigue Prob'), ('fatigue_conf', 'Fatigue Conf'),
            ('fatigue_calib', 'Fatigue Calib'),
            ('fatigue_index', 'Fatigue Index'), ('focus_index', 'Focus Index'),
            ('fatigue_landmark', 'Landmark Fatigue'), ('lmk_calib', 'Landmark Calib'),
            ('perclos', 'PERCLOS'), ('blink_rate', 'Blink/min'), ('yawn_rate', 'Yawn/min'),
            ('va_mode', 'VA Mode'), ('valence', 'Valence'), ('arousal', 'Arousal'),
        ]
        for idx, (key, title) in enumerate(metric_names):
            row = idx // 2
            col = (idx % 2) * 2
            metrics_grid.addWidget(QtWidgets.QLabel(title), row, col)
            value = QtWidgets.QLabel('--')
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            metrics_grid.addWidget(value, row, col + 1)
            self.metric_labels[key] = value
        self._metrics_layout.addWidget(metrics_box)

        status_box = QtWidgets.QGroupBox('状态总览')
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.overall_status_label = QtWidgets.QLabel('等待稳定信号')
        self.overall_status_label.setWordWrap(True)
        self.overall_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overall_status_label.setMinimumHeight(88)
        self.overall_status_label.setStyleSheet(
            'background-color: #333; color: white; padding: 12px 16px; border-radius: 10px; '
            'font-size: 16px; font-weight: 700;'
        )
        status_layout.addWidget(self.overall_status_label)
        self._metrics_layout.addWidget(status_box)
        self._metrics_layout.addStretch(1)

        quadrant_box = QtWidgets.QGroupBox('双轴连续光谱')
        quadrant_form = QtWidgets.QFormLayout(quadrant_box)
        self.quadrant_state = QtWidgets.QLabel('--')
        self.quadrant_x = QtWidgets.QLabel('--')
        self.quadrant_y = QtWidgets.QLabel('--')
        self.quadrant_confidence = QtWidgets.QLabel('--')
        self.quadrant_reason = QtWidgets.QLabel('等待稳定信号')
        self.quadrant_state.setAlignment(QtCore.Qt.AlignCenter)
        self.quadrant_reason.setWordWrap(True)
        self.quadrant_reason.setStyleSheet('color: #111111; font-size: 14px; font-weight: 600;')
        self.quadrant_state.setMinimumHeight(54)
        self.quadrant_state.setMinimumWidth(220)
        self.quadrant_state.setStyleSheet(
            'background-color: #444; color: white; padding: 10px 14px; border-radius: 8px; '
            'font-size: 18px; font-weight: 700;'
        )
        quadrant_form.addRow('状态', self.quadrant_state)
        quadrant_form.addRow('行为专注度 X', self.quadrant_x)
        quadrant_form.addRow('生理激活度 Y', self.quadrant_y)
        quadrant_form.addRow('置信度', self.quadrant_confidence)
        quadrant_form.addRow('判定依据', self.quadrant_reason)
        self.spectrum_dashboard = SpectrumDashboardWidget()
        self._quadrant_layout.addWidget(self.spectrum_dashboard)
        self._quadrant_layout.addWidget(quadrant_box)
        self._quadrant_layout.addStretch(1)

        tabs.addTab(self._overview_page, '概览')
        tabs.addTab(self._metrics_page, '指标')
        tabs.addTab(self._quadrant_page, '光谱')

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.apply_models_btn.clicked.connect(self.apply_models)
        self.calib_btn.clicked.connect(self.begin_calibration)
        self.record_btn.toggled.connect(self.toggle_recording)
        self.show_landmarks.toggled.connect(self._sync_display_options)
        self.show_face_box.toggled.connect(self._sync_display_options)

    def _add_path_row(self, form: QtWidgets.QFormLayout, title: str, edit: QtWidgets.QLineEdit) -> None:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        browse = QtWidgets.QPushButton('浏览')
        browse.clicked.connect(lambda: self._browse_model(edit))
        row_layout.addWidget(browse)
        form.addRow(title, row)

    def _browse_model(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            '选择模型文件',
            str(REPO_ROOT),
            'Model files (*.onnx *.task);;All files (*)',
        )
        if path:
            edit.setText(path)

    def _load_default_models(self) -> None:
        self.facephys_edit.setText(_default_model_path('step.onnx'))
        self.va_edit.setText(_default_model_path('va_mbf.onnx'))
        self.fatigue_edit.setText(_default_model_path('fatigue.onnx'))

    def _default_record_path(self) -> Path:
        logs_dir = REPO_ROOT / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return logs_dir / f'stateguard_record_{stamp}.txt'

    def _open_record_file(self, path: Optional[Path] = None) -> None:
        if self._record_fp is not None:
            return
        target = path or self._default_record_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._record_fp = open(target, 'a', encoding='utf-8')
        self._record_path = target
        self.record_path_label.setText(str(target))
        self.status_bar.showMessage(f'记录已开始：{target}')

    def _close_record_file(self) -> None:
        if self._record_fp is not None:
            try:
                self._record_fp.flush()
                self._record_fp.close()
            except Exception:
                pass
        self._record_fp = None
        self._record_path = None

    def toggle_recording(self, checked: bool) -> None:
        if checked:
            path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                '选择记录文件',
                str(self._default_record_path()),
                'Text files (*.txt);;All files (*)',
            )
            record_path = Path(path_text) if path_text else self._default_record_path()
            self._open_record_file(record_path)
            self._recording = True
            self.record_btn.setText('停止记录')
            self._last_record_second = None
            self._last_record_key = None
        else:
            self._recording = False
            self._close_record_file()
            self.record_btn.setText('开始记录')
            self.status_bar.showMessage('记录已停止')

    def _current_model_paths(self) -> tuple[str, str, str]:
        return self.facephys_edit.text().strip(), self.va_edit.text().strip(), self.fatigue_edit.text().strip()

    def _create_worker(self) -> CameraPipelineWorker:
        facephys_path, va_path, fatigue_path = self._current_model_paths()
        worker = CameraPipelineWorker(
            camera_index=int(self.camera_index.value()),
            facephys_path=facephys_path,
            va_path=va_path,
            fatigue_path=fatigue_path,
        )
        worker.set_display_options(self.show_landmarks.isChecked(), self.show_face_box.isChecked())
        return worker

    def _sync_display_options(self, *_args) -> None:
        if self.worker is not None:
            self.worker.set_display_options(self.show_landmarks.isChecked(), self.show_face_box.isChecked())

    def start_camera(self) -> None:
        self.stop_camera()
        self.worker = self._create_worker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_changed.connect(self.on_status)
        self.worker.calibration_changed.connect(self.on_calibration_status)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage('摄像头启动中...')

    def stop_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._calibration_ui_active = False
        if self.record_btn.isChecked():
            self.record_btn.blockSignals(True)
            self.record_btn.setChecked(False)
            self.record_btn.blockSignals(False)
            self.toggle_recording(False)

    def apply_models(self) -> None:
        if self.worker is None:
            self.status_bar.showMessage('模型已更新，点击“开始”生效')
            return

        calib = getattr(self.worker, '_calibrator', None)
        if calib is not None and not calib.finished:
            self._pending_model_reload = True
            self.status_bar.showMessage('校准进行中，模型更新已排队，校准结束后自动生效')
            return

        self._restart_camera_with_current_models()

    def _restart_camera_with_current_models(self) -> None:
        self.stop_camera()
        self.start_camera()
        self._pending_model_reload = False

    def begin_calibration(self) -> None:
        if self.worker is None:
            self.start_camera()
        if self.worker is not None:
            self.calib_progress.setValue(0)
            self._calibration_ui_active = True
            self._calibration_ui_text = '校准中'
            self._refresh_overall_status_label()
            self.worker.request_calibration(
                self.calib_sec.value(),
                self.calib_min_quality.value(),
                self.calib_min_samples.value(),
                self.calib_sample_sec.value(),
            )

    @QtCore.Slot(str)
    def on_status(self, text: str) -> None:
        self.status_bar.showMessage(text)

    @QtCore.Slot(str)
    def on_calibration_status(self, text: str) -> None:
        self.calib_status.setText(text)
        self.status_bar.showMessage(text)
        self._calibration_ui_text = text
        if '校准中：剩余' in text:
            self._calibration_ui_active = True
            try:
                remaining = int(text.rsplit(' ', 1)[-1].removesuffix('s'))
                total = max(1, int(round(self.calib_sec.value())))
                progress = int(np.clip(100 * (1.0 - remaining / total), 0, 100))
                self.calib_progress.setValue(progress)
            except Exception:
                pass
        elif '校准完成' in text:
            self._calibration_ui_active = False
            self.calib_progress.setValue(100)
            if self._pending_model_reload:
                self._pending_model_reload = False
                QtCore.QTimer.singleShot(0, self._restart_camera_with_current_models)
        self._refresh_overall_status_label()

    @QtCore.Slot(str)
    def on_error(self, text: str) -> None:
        self.status_bar.showMessage(text)
        self.calib_status.setText(text)
        self._calibration_ui_text = text
        self._calibration_ui_active = False
        self._refresh_overall_status_label()

    @QtCore.Slot(object, object)
    def on_frame(self, image: object, result: object, quadrant: object) -> None:
        if isinstance(image, QtGui.QImage):
            pix = QtGui.QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pix.scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        if result is None or quadrant is None or not getattr(quadrant, 'key', None):
            return
        self.metric_labels['hr'].setText(_fmt(getattr(result, 'hr', None), 1))
        self.metric_labels['rmssd'].setText(_fmt(getattr(result, 'rmssd', None), 1))
        self.metric_labels['sdnn'].setText(_fmt(getattr(result, 'sdnn', None), 1))
        self.metric_labels['quality'].setText(_fmt(getattr(result, 'quality', None), 2))
        self.metric_labels['fatigue'].setText(_fmt(getattr(result, 'fatigue', None), 2))
        self.metric_labels['fatigue_landmark'].setText(_fmt(getattr(result, 'fatigue_landmark', None), 2))
        self.metric_labels['perclos'].setText(_fmt(getattr(result, 'perclos', None), 2))
        self.metric_labels['blink_rate'].setText(_fmt(getattr(result, 'blink_rate', None), 2))
        self.metric_labels['yawn_rate'].setText(_fmt(getattr(result, 'yawn_rate', None), 2))
        self.metric_labels['va_mode'].setText(str(getattr(result, 'va_mode', '--')))
        self.metric_labels['valence'].setText(_fmt(getattr(result, 'valence', None), 2))
        self.metric_labels['arousal'].setText(_fmt(getattr(result, 'arousal', None), 2))
        self.metric_labels['fatigue_conf'].setText(_fmt(getattr(result, 'fatigue_confidence', None), 2))
        calib = getattr(self.worker, '_calibrator', None) if self.worker is not None else None
        if calib is not None:
            fatigue_calib = calib.calibrated_value(getattr(result, 'fatigue', None))
            lmk_calib = calib.calibrated_scalar_value(
                getattr(result, 'fatigue_landmark', None),
                calib.lmk_fatigue_baseline,
                calib.lmk_fatigue_spread,
            )
            self.metric_labels['fatigue_calib'].setText(_fmt(fatigue_calib, 2))
            self.metric_labels['lmk_calib'].setText(_fmt(lmk_calib, 2))
        else:
            self.metric_labels['fatigue_calib'].setText('--')
            self.metric_labels['lmk_calib'].setText('--')

        fatigue_index, focus_index = self._compute_user_indices(result, quadrant)
        self.metric_labels['fatigue_index'].setText(_fmt(fatigue_index, 1))
        self.metric_labels['focus_index'].setText(_fmt(focus_index, 1))

        # keep last frame result for recorder
        self._last_frame_result = result
        self._update_quadrant_display(quadrant)

    def _update_quadrant_display(self, quadrant: object) -> None:
        if not isinstance(quadrant, SpectrumPrediction) or not quadrant.key:
            self.spectrum_dashboard.clear_prediction()
            if not self._quadrant_ui_waiting_shown:
                self.quadrant_state.setText('--')
                self.quadrant_x.setText('--')
                self.quadrant_y.setText('--')
                self.quadrant_confidence.setText('--')
                self.quadrant_reason.setText('等待稳定信号')
                self._refresh_overall_status_label(quadrant_text='等待稳定信号')
                self._quadrant_ui_waiting_shown = True
            return

        self._quadrant_ui_waiting_shown = False
        now = time.time()
        stable_switch_delay = 0.75
        stable_update_interval = 0.35
        confidence_delta = abs(float(quadrant.confidence) - self._quadrant_ui_last_confidence)

        if self._quadrant_ui_key is None:
            should_update = True
        elif quadrant.key == self._quadrant_ui_key:
            should_update = (
                (now - self._quadrant_ui_last_update_at) >= stable_update_interval
                or confidence_delta >= 0.06
            )
        else:
            if self._quadrant_ui_candidate_key != quadrant.key:
                self._quadrant_ui_candidate_key = quadrant.key
                self._quadrant_ui_candidate_seen_at = now
            should_update = (now - self._quadrant_ui_candidate_seen_at) >= stable_switch_delay

        if not should_update:
            return

        self._quadrant_ui_key = quadrant.key
        self._quadrant_ui_candidate_key = quadrant.key
        self._quadrant_ui_candidate_seen_at = now
        self._quadrant_ui_last_update_at = now
        self._quadrant_ui_last_confidence = float(quadrant.confidence)

        self.quadrant_state.setText(quadrant.label)
        self.quadrant_x.setText(_fmt(quadrant.x, 2))
        self.quadrant_y.setText(_fmt(quadrant.y, 2))
        self.quadrant_confidence.setText(_fmt(quadrant.confidence, 2))
        self.quadrant_reason.setText(quadrant.reason)
        self.spectrum_dashboard.set_prediction(quadrant)
        self._set_quadrant_style(quadrant.key, quadrant.confidence)
        self._append_record(quadrant)
        self._refresh_overall_status_label(quadrant_text=f'{quadrant.label}\n{quadrant.reason}')

    def _refresh_overall_status_label(self, quadrant_text: Optional[str] = None) -> None:
        if self._calibration_ui_active:
            self.overall_status_label.setText(self._calibration_ui_text)
            return
        if quadrant_text is not None:
            self.overall_status_label.setText(quadrant_text)
            return
        if self._quadrant_ui_key:
            self.overall_status_label.setText(f'{self.quadrant_state.text()}\n{self.quadrant_reason.text()}')
            return
        if self._calibration_ui_text:
            self.overall_status_label.setText(self._calibration_ui_text)
            return
        self.overall_status_label.setText('等待稳定信号')

    def _append_record(self, quadrant: SpectrumPrediction) -> None:
        if not self._recording or self._record_fp is None:
            return
        now = datetime.now().astimezone()
        current_second = int(now.timestamp())
        if self._last_record_second == current_second and self._last_record_key == quadrant.key:
            return
        self._last_record_second = current_second
        self._last_record_key = quadrant.key
        # collect last pipeline result if available
        result = getattr(self, '_last_frame_result', None)
        calib = getattr(self.worker, '_calibrator', None) if getattr(self, 'worker', None) is not None else None
        fatigue_calib = None
        lmk_calib = None
        try:
            if calib is not None:
                fatigue_calib = calib.calibrated_value(getattr(result, 'fatigue', None))
                lmk_calib = calib.calibrated_scalar_value(
                    getattr(result, 'fatigue_landmark', None),
                    calib.lmk_fatigue_baseline,
                    calib.lmk_fatigue_spread,
                )
        except Exception:
            fatigue_calib = None
            lmk_calib = None

        hr = _fmt(getattr(result, 'hr', None), 1)
        rmssd = _fmt(getattr(result, 'rmssd', None), 1)
        sdnn = _fmt(getattr(result, 'sdnn', None), 1)
        quality = _fmt(getattr(result, 'quality', None), 2)
        va_mode = str(getattr(result, 'va_mode', '--'))
        valence = _fmt(getattr(result, 'valence', None), 2)
        arousal = _fmt(getattr(result, 'arousal', None), 2)
        fatigue = _fmt(getattr(result, 'fatigue', None), 2)
        fatigue_conf = _fmt(getattr(result, 'fatigue_confidence', None), 2)
        fatigue_calib_s = _fmt(fatigue_calib, 2) if fatigue_calib is not None else '--'
        fatigue_lmk = _fmt(getattr(result, 'fatigue_landmark', None), 2)
        lmk_calib_s = _fmt(lmk_calib, 2) if lmk_calib is not None else '--'
        fatigue_index, focus_index = self._compute_user_indices(result, quadrant, fatigue_calib=fatigue_calib, lmk_calib=lmk_calib)
        fatigue_index_s = _fmt(fatigue_index, 1)
        focus_index_s = _fmt(focus_index, 1)
        perclos = _fmt(getattr(result, 'perclos', None), 2)
        blink_rate = _fmt(getattr(result, 'blink_rate', None), 2)
        yawn_rate = _fmt(getattr(result, 'yawn_rate', None), 2)
        bvp = _fmt(getattr(result, 'bvp', None), 6)
        face_box = str(getattr(result, 'face_box', None))
        landmarks_count = str(len(getattr(result, 'landmarks', []) or []))

        line = (
            f"{now.isoformat(timespec='seconds')}\t"
            f"{quadrant.key}\t{quadrant.label}\t"
            f"confidence={quadrant.confidence:.2f}\t"
            f"x={quadrant.x:.2f}\t y={quadrant.y:.2f}\t"
            f"hr={hr}\trmssd={rmssd}\tsdnn={sdnn}\tquality={quality}\t"
            f"va_mode={va_mode}\tvalence={valence}\tarousal={arousal}\t"
            f"fatigue_prob={fatigue}\tfatigue_conf={fatigue_conf}\tfatigue_calib={fatigue_calib_s}\t"
            f"fatigue_index={fatigue_index_s}\tfocus_index={focus_index_s}\t"
            f"fatigue_lmk={fatigue_lmk}\tlmk_calib={lmk_calib_s}\t"
            f"perclos={perclos}\tblink_rate={blink_rate}\tyawn_rate={yawn_rate}\t"
            f"bvp={bvp}\tface_box={face_box}\tlandmarks={landmarks_count}\n"
        )
        try:
            self._record_fp.write(line)
            self._record_fp.flush()
        except Exception as exc:
            self.status_bar.showMessage(f'记录写入失败：{exc}')
            self.record_btn.blockSignals(True)
            self.record_btn.setChecked(False)
            self.record_btn.blockSignals(False)
            self._recording = False
            self._close_record_file()
            self.record_btn.setText('开始记录')

    def _compute_user_indices(
        self,
        result: object,
        quadrant: object,
        *,
        fatigue_calib: Optional[float] = None,
        lmk_calib: Optional[float] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        fatigue_raw = getattr(result, 'fatigue', None)
        fatigue_landmark = getattr(result, 'fatigue_landmark', None)
        quality = getattr(result, 'quality', None)
        arousal = getattr(result, 'arousal', None)

        fatigue_source = fatigue_calib if fatigue_calib is not None else fatigue_raw
        if fatigue_source is None and lmk_calib is not None:
            fatigue_source = lmk_calib

        fatigue_index = None
        if fatigue_source is not None:
            fatigue_index = float(np.clip(float(fatigue_source), 0.0, 1.0) * 100.0)

        focus_index = None
        score_map = getattr(quadrant, 'scores', None) if quadrant is not None else None
        if isinstance(score_map, dict) and score_map:
            focus_score = (
                0.45 * float(score_map.get('flow', 0.0))
                + 0.25 * float(score_map.get('routine', 0.0))
                - 0.20 * float(score_map.get('distraction', 0.0))
                - 0.15 * float(score_map.get('overload', 0.0))
                - 0.25 * float(score_map.get('exhaustion', 0.0))
            )
            if quality is not None and np.isfinite(quality):
                focus_score += 0.15 * (float(quality) - 0.5)
            if arousal is not None and np.isfinite(arousal):
                focus_score += 0.10 * (float(arousal) - 0.5)
            if fatigue_raw is not None and np.isfinite(fatigue_raw):
                focus_score -= 0.10 * float(fatigue_raw)
            if fatigue_landmark is not None and np.isfinite(fatigue_landmark):
                focus_score -= 0.05 * float(fatigue_landmark)
            focus_index = float(np.clip(0.5 + focus_score, 0.0, 1.0) * 100.0)

        return fatigue_index, focus_index


    def _set_quadrant_style(self, key: str, confidence: float) -> None:
        palette = {
            'routine': '#2563eb',
            'flow': '#16a34a',
            'overload': '#d73a49',
            'distraction': '#f59e0b',
            'exhaustion': '#7c3aed',
        }
        color = palette.get(key, '#666666')
        self.quadrant_state.setStyleSheet(
            f'background-color: {color}; color: white; padding: 6px; border-radius: 6px;'
        )
        self.quadrant_confidence.setStyleSheet(f'color: {color}; font-weight: 600;')
        self.quadrant_reason.setStyleSheet('color: #111111; font-size: 14px; font-weight: 600;')

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_camera()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
