from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from stateguard.pipeline import StateGuardConfig, StateGuardPipeline


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
    frame_ready = QtCore.Signal(object, object)
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

    def run(self) -> None:
        try:
            cap = cv2.VideoCapture(int(self.camera_index))
            if not cap.isOpened():
                self.error.emit(f'无法打开摄像头 {self.camera_index}')
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            source_fps = reported_fps if reported_fps >= 1.0 else 30.0
            target_fps = min(30.0, source_fps)

            cfg = StateGuardConfig(
                facephys_path=self.facephys_path,
                va_path=self.va_path,
                fatigue_path=self.fatigue_path,
                fps=target_fps,
                source_fps=source_fps,
            )
            pipeline = StateGuardPipeline(cfg)
            self.status_changed.emit(
                f'摄像头已启动 | reported={reported_fps:.1f} fps | source={source_fps:.1f} fps | target={target_fps:.1f} fps'
            )

            while not self._stop:
                ok, frame = cap.read()
                if not ok:
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

                display = frame.copy()
                if self._show_landmarks and result.landmarks:
                    self._draw_landmarks(display, result.landmarks, result.face_box)
                if self._show_face_box and result.face_box:
                    x1, y1, x2, y2 = result.face_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                self.frame_ready.emit(self._to_qimage(display), result)
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

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(10)
        root.addLayout(side, 2)

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
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        camera_form.addRow('设备索引', self.camera_index)
        camera_form.addRow(self.show_landmarks)
        camera_form.addRow(self.show_face_box)
        camera_form.addRow(btn_row)
        side.addWidget(camera_box)

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
        side.addWidget(model_box)

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
        side.addWidget(calib_box)

        metrics_box = QtWidgets.QGroupBox('实时指标')
        metrics_grid = QtWidgets.QGridLayout(metrics_box)
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(8)
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        metric_names = [
            ('hr', 'HR'), ('rmssd', 'RMSSD'), ('sdnn', 'SDNN'), ('quality', 'Quality'),
            ('fatigue', 'Fatigue'), ('fatigue_calib', 'Fatigue Calib'),
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
        side.addWidget(metrics_box)
        side.addStretch(1)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.apply_models_btn.clicked.connect(self.apply_models)
        self.calib_btn.clicked.connect(self.begin_calibration)
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
        if '校准中：剩余' in text:
            try:
                remaining = int(text.rsplit(' ', 1)[-1].removesuffix('s'))
                total = max(1, int(round(self.calib_sec.value())))
                progress = int(np.clip(100 * (1.0 - remaining / total), 0, 100))
                self.calib_progress.setValue(progress)
            except Exception:
                pass
        elif '校准完成' in text:
            self.calib_progress.setValue(100)
            if self._pending_model_reload:
                self._pending_model_reload = False
                QtCore.QTimer.singleShot(0, self._restart_camera_with_current_models)

    @QtCore.Slot(str)
    def on_error(self, text: str) -> None:
        self.status_bar.showMessage(text)
        self.calib_status.setText(text)

    @QtCore.Slot(object, object)
    def on_frame(self, image: object, result: object) -> None:
        if isinstance(image, QtGui.QImage):
            pix = QtGui.QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pix.scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        if result is None:
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
