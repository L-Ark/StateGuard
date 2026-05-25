# StateGuard 部署文档（Windows）

StateGuard 是 FacePhys + VA 多模态轻量推理管线的本地部署包。
本目录是**自包含**的：所有权重已转为 ONNX，运行时不依赖 PyTorch / TensorFlow / Keras。

## 1. 设计与性能

**架构**（每路一帧）：
```
摄像头帧 (RGB) ──► 人脸检测+裁剪 ──► 36×36 ──► FacePhys (step.onnx) ──► BVP
                                  └─► 224×224 ──► (按窗口抽 4 帧) ─┬► VA      (va_mbf.onnx)  ──► V/A
                                                                  └► Fatigue (fatigue.onnx) ──► P(fatigue)
        └─► BVP 滚动窗 ──► HR / RMSSD / SDNN
```

- **FacePhys** 是逐帧流式 ONNX (`InfinitePulse.step`)，状态在调用间持续，**无窗口延迟**。
- **VA** 模型每 15 秒只对 4 个均匀采样关键帧推理一次（验证 1–5% 关键帧基本无损）。默认会根据 rPPG 质量自动在 **single-modal(vision)** 和 **multimodal** 之间切换：质量高时使用多模态，画质差或置信度低时回退到单模态。
- **Fatigue** 模型与 VA 共用同一组关键帧（112×112 RGB），同窗口推理，每窗口 +8 ms（4 帧 × ~2 ms）。
- **HRV** 估计 (Welch + peaks) 限制为 ~1 Hz，避免拖慢每帧。

**服务器端基准（CPU, 1 线程）**：

| 组件 | 加载 | 单次推理 | 内存增量 |
|---|---|---|---|
| step.onnx (每帧) | 350 ms | mean 17 ms / P95 34 ms | +18 MB |
| va_mbf.onnx (batch=4) | 120 ms | 200 ms / 窗 (50 ms/帧) | +36 MB |
| fatigue.onnx (batch=4) | 60 ms | 8 ms / 窗 (2 ms/帧) | +6 MB |
| **端到端 pipeline** | <600 ms | **mean 7 ms / 帧** | ~160 MB RSS |

以 30 fps 输入，**~4× 实时余量**（实测 ≈64 fps 处理 60 fps 视频），CPU 占用单核 60% 以下。可直接在普通笔记本上跑。

## 2. 状态评估依据与公式

StateGuard 采用双轴连续光谱：先把多源信号压成两条连续轴，再用这两条轴对当前状态打分。

### 2.1 就绪门槛

状态判定前，必须同时满足：

- 历史样本数达到 `24` 帧；
- 历史中至少有 `2` 类有效特征源非空。

有效特征源包括：`hr`、`rmssd`、`sdnn`、`quality`、`perclos`、`blink_rate`、`yawn_rate`、`fatigue`、`fatigue_landmark`、`arousal`、`valence`。



2.2 连续轴
行为专注度轴：
$$A = \text{clip}_{[0,1]}(0.35\hat{a} + 0.30\hat{h} + 0.20\hat{b} + 0.15\hat{q})$$
其中：

$  \hat{a} = \text{norm}(arousal; -0.2, 0.8)  $
$  \hat{h} = \text{norm}(hr; 58, 98)  $
$  \hat{b} = 1 - \text{norm}(blink\_rate; 8, 22)  $
$  \hat{q} = \text{norm}(quality; 0.3, 0.9)  $

生理耗竭轴：
$$D = \text{clip}_{[0,1]}(0.30\hat{p} + 0.25\hat{f} + 0.15\hat{l} + 0.20\hat{r} + 0.10\hat{y})$$
其中：

$  \hat{p} = \text{norm}(perclos; 0.08, 0.35)  $
$  \hat{f} = \text{norm}(fatigue; 0.25, 0.80)  $
$  \hat{l} = \text{norm}(fatigue\_landmark; 0.20, 0.80)  $
$  \hat{r} = 1 - \text{norm}(rmssd; 18, 60)  $
$  \hat{y} = \text{norm}(yawn\_rate; 0.4, 2.0)  $
---

### 2.3 五区域打分

在连续轴 $A/D$ 上分别计算区域分数，取**最大者**作为当前状态：

$$
S_{\text{focus}} = A(1-D)(0.70 + 0.30\widehat{rmssd})(0.75 + 0.25\widehat{valence}_{+})
$$

$$
S_{\text{overload}} = AD(0.70 + 0.30\widehat{valence}_{-})(0.70 + 0.30(1-\widehat{rmssd}))
$$

$$
S_{\text{distraction}} = (1-A)(1-D)(0.75 + 0.25\widehat{arousal}_{\text{low}})(0.70 + 0.30(1-\widehat{valence}_{-}))
$$

$$
S_{\text{fatigue}} = (1-A)D(0.75 + 0.25\widehat{perclos})(0.75 + 0.25\widehat{fatigue})
$$

其中：
- $\widehat{rmssd} = \text{norm}(rmssd; 18, 60)$
- $\widehat{valence}_{+} = \text{norm}(valence; 0, 0.8)$
- $\widehat{valence}_{-} = \text{norm}(-valence; 0, 0.8)$
- $\widehat{arousal}_{\text{low}} = 1 - \text{norm}(arousal; -0.2, 0.6)$
- $\widehat{perclos} = \text{norm}(perclos; 0.08, 0.35)$
- $\widehat{fatigue} = \text{norm}(fatigue; 0.25, 0.80)$

---

### 2.4 输出字段

- **`x`**：行为专注度轴 $A$
- **`y`**：生理耗竭轴 $D$
- **`confidence`**：由最大分数与次大分数的差值结合 `quality` 调整得到
- **`reason`**：展示当前状态的主导特征组合，便于界面展示和日志追踪




### 2.5 说明

- `quality` 主要来自 HRV / rPPG 观测质量，用于辅助激活轴和最终置信度。
- `fatigue_landmark` 由 PERCLOS、眨眼频率、打哈欠频率融合得到，先参与校准，再进入耗竭轴。
- 光谱页中的仪表盘会把实时坐标点画到五区域平面上，中心为“常态工作区”。

## 3. 目录结构

```
deploy/
├── README.md                  # 本文档
├── requirements.txt
├── smoke_test.py              # 端到端冒烟测试
├── stateguard/                # 核心包
│   ├── __init__.py
│   ├── pipeline.py            # StateGuardPipeline + StateGuardConfig + FrameResult
│   ├── hrv.py                 # HRVStream (HR/RMSSD/SDNN)
│   ├── models/
│   │   ├── face.py            # FaceCropper (MediaPipe → Haar fallback)
│   │   ├── facephys_runner.py # 逐帧 rPPG
│   │   ├── va_runner.py       # batch VA
│   │   └── fatigue_runner.py  # batch fatigue (binary drowsy/normal)
│   └── weights/
│       ├── step.onnx          # FacePhys InfinitePulse, ~2 MB
│       ├── state.pkl          # 预热的递归状态 (可选, 提高早期收敛)
│       ├── va_mbf.onnx        # mbf_va_mtl + 6-logit dual head, ~8 MB
│       └── fatigue.onnx       # MobileNetV3-Small (0.75x) 二分类, ~4 MB
└── examples/
    ├── run_video.py           # 处理离线视频 → CSV
    └── run_webcam.py          # 实时摄像头 OpenCV demo
```

## 4. Windows 安装与运行

### 4.1 准备环境

推荐 **Python 3.10 或 3.11**（与 mediapipe 兼容）。

```powershell
# 拷贝 deploy/ 整个目录到目标机器（路径不能含中文/空格）
cd C:\path\to\deploy

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell;  cmd 用 .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 mediapipe 安装失败，可以忽略 → 自动回退到 OpenCV Haar 人脸检测（精度略低，仍可用）。

### 4.2 验证安装

```powershell
python smoke_test.py
```
注意：脚本默认读取一个 Linux 路径下的 CAST 视频。在 Windows 上请把 `smoke_test.py` 顶部的 `VID` 改成本地视频文件，或运行下面的离线脚本：

```powershell
python examples\run_video.py path\to\some\video.mp4 --out result.csv
```
默认会同时生成一个同名 txt 日志，例如 `result.txt`，每一行包含本机时间戳和对应状态值。

### 4.3 实时摄像头 demo

```powershell
python examples\run_webcam.py --cam 0
```
启动后会先进入个人疲劳基线校准，默认约 90 秒，建议保持自然放松、正视摄像头，不要刻意做表情。校准完成后再进入正常监测。
按 `q` 退出。窗口里会显示人脸框、BVP 波形条、HR / RMSSD / SDNN、最新 V/A、校准后的 Fatigue 以及个人基线。
同时会生成 `stateguard_webcam.txt`，按帧记录本机系统时间、原始 fatigue、校准后 fatigue 和基线值。

如果你已经创建了 `.venv`，也可以直接双击根目录的 `run_webcam.bat` 启动。

参数：
- `--cam N`：摄像头索引（默认 0）。
- `--gate-va`：开启 VA 门控（HRV 异常时才激活），降低 CPU。
- `--va-mode auto|vision|multimodal`：默认 `auto`，会根据 rPPG quality 自动切换。
- `--va-quality-threshold` / `--va-quality-hysteresis`：控制 auto 模式下切换到/退出 multimodal 的阈值。
- `--calib-sec`：个人基线校准时长，默认 90 秒，建议 60-180 秒。
- `--calib-min-quality`：校准时允许收集 fatigue 样本的最低 HRV quality。
- `--calib-min-samples`：完成校准所需的最少有效窗口数。
- `--txt-log PATH`：指定 txt 日志文件路径。每帧会记录系统时间、bvp、hr、rmssd、sdnn、quality、va_mode、V/A、fatigue。

启动时会先测一次摄像头实际 FPS；如果摄像头低于 30 fps，程序会自动降到实际帧率运行，避免上采样报错。

## 5. 在 UI 中集成（关键 API）

建议把 StateGuard 作为后台线程跑，UI 只消费 `FrameResult`：

```python
from pathlib import Path
from stateguard import StateGuardPipeline, StateGuardConfig

W = Path('stateguard/weights')
pipe = StateGuardPipeline(StateGuardConfig(
    facephys_path=str(W / 'step.onnx'),
    va_path=str(W / 'va_mbf.onnx'),
    fatigue_path=str(W / 'fatigue.onnx'),   # 传 None 则禁用 fatigue 头
    state_path=None,
    fps=30.0,             # FacePhys 训练域，不要改
    source_fps=60.0,      # ★ 摄像头/视频实际帧率，>30 时自动降采样
    va_window_sec=15.0,
    va_keyframes=4,
    gate_va=False,        # True = 仅在异常时跑 VA（同时也节流 fatigue）
    num_threads=1,
))

# 主循环（30Hz）
import cv2
cap = cv2.VideoCapture(0)
while True:
    ok, frame_bgr = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r = pipe.step(rgb)
    #   r.bvp      每帧 rPPG 标量
    #   r.hr / r.rmssd / r.sdnn / r.quality   ~1Hz 刷新的生理指标
    #   r.valence / r.arousal                 每 15s 刷新一次（首次 15s 内为 None）
    #   r.fatigue                             每 15s 刷新一次的 P(fatigue) ∈ [0,1]
    #   r.face_box (x1,y1,x2,y2) 或 None
```

**线程模型**：`pipeline.step()` 不是线程安全的。UI 线程只读，推理线程独占一个 pipeline 实例即可。常见做法：

```
┌───────────┐  frames  ┌─────────────────┐  FrameResult  ┌─────┐
│ Camera Thr│ ───────► │ Inference Thread│ ────────────► │ UI  │
└───────────┘ queue(1) └─────────────────┘ queue(1)      └─────┘
```
队列容量 1、丢弃旧帧，避免延迟堆积（实时系统的标准做法）。

### 4.1 Fatigue 模型 I/O（独立调用）

如果不通过 pipeline，希望直接对一张人脸图片做 fatigue 判定，使用 `FatigueRunner`：

```python
from stateguard import FatigueRunner
import numpy as np

fat = FatigueRunner('stateguard/weights/fatigue.onnx', num_threads=1)

# 输入: (N, H, W, 3) uint8 RGB；任意 H/W，内部自动 resize 到 112×112
faces = np.zeros((4, 224, 224, 3), dtype=np.uint8)
p = fat.predict(faces)        # (N,) float32 — P(fatigue) ∈ [0, 1]
logits = fat.predict_logits(faces)   # (N, 2) — [normal_logit, fatigue_logit]
```

**ONNX 模型规格**

| 项 | 值 |
|---|---|
| 文件 | `stateguard/weights/fatigue.onnx` (3.9 MB) |
| 架构 | timm `mobilenetv3_small_075`（ImageNet 预训练 → 二分类微调） |
| 参数量 | **1.02 M**（部署预算 < 2 M） |
| 输入 `frames` | `(N, 3, 112, 112)` float32, RGB, 已归一化 mean=std=0.5（即 `(x/255 - 0.5) / 0.5`） |
| 输出 `logits` | `(N, 2)` float32, 顺序 `[normal, fatigue]` |
| 输出 `prob`   | `(N,)` float32, 等价于 `softmax(logits)[:, 1]`，即 **P(fatigue)** |
| 类别约定 | `0 = normal`, `1 = fatigue/drowsy` |
| Opset | 17 |
| 单帧推理 (CPU, 1 线程) | ~2.0 ms |

**训练数据与精度**（详见 `AttnEmo/fatigue/`）

| 数据集 | acc | F1 |
|---|---|---|
| Kaggle fatigue 留出验证 (220+220) | **0.864** | 0.860 |
| Kaggle drowsy_detection 测试 (757+726) | **0.959** | 0.958 |

**典型阈值与建议用法**

- 输出是单帧概率，建议在 UI 侧再做时间平滑（pipeline 已经在 15s 窗口内对 4 帧取均值，`r.fatigue` 即窗口均值）。
- 默认 0.5 作为告警阈值；若希望更稳健，可累计连续 N 个窗口（≥ 30s）的 `r.fatigue > 0.5` 才触发干预。
- 与 rPPG 联动：`r.fatigue > 0.5 且 r.rmssd 低于个人基线` 是更可靠的疲劳信号（生理 + 表情两证）。

## 6. 调优开关

| 选项 | 默认 | 说明 |
|---|---|---|
| `StateGuardConfig.fps` | 30 | **不要改**——FacePhys 训练域。模型 dt = 1/fps |
| `source_fps` | None (=fps) | 摄像头/视频实际帧率。>30 时 pipeline 自动降采样 |
| `va_window_sec` | 15 | VA 输出周期 |
| `va_keyframes` | 4 | 每窗口送入 VA 的帧数（实测 4 已足够，最多 8） |
| `gate_va` | False | 节能模式，HRV 异常时才跑 VA（fatigue 同样仅在该窗口运行） |
| `fatigue_path` | None | 设为 `weights/fatigue.onnx` 启用 fatigue 头；和 VA 共用关键帧，每窗口 +8 ms |
| `num_threads` | 1 | 模型小，单线程最优；多线程反而引入调度抖动 |
| `va_mode` | auto | `auto` = 按 rPPG quality 自动在 vision / multimodal 间切换；也可固定为 `vision` 或 `multimodal` |
| `fusion_alpha` | 0.5 | When `va_mode=multimodal`, vision weight in fusion (0..1) |
| `va_quality_threshold` | 0.55 | `auto` 模式下，quality 高于该值时切换到 multimodal |
| `va_quality_hysteresis` | 0.08 | `auto` 模式下的回切滞回，避免 mode 在阈值附近抖动 |
| `FaceCropper(detect_every_n=5)` | 5 | 检测稀疏度；帧间用平滑插值 |

> ⚠️ **关于 fps 配置**：FacePhys 在 30 fps 上训练。如果你的摄像头是 60 fps，**必须**把 `source_fps=60` 传进来，否则模型会以为时间流逝速率是真实的 2×，HR 估计会错。修复前 smoke test 上 HR 错了 ~33%。在 Windows 上摄像头默认通常是 30 fps，在某些笔记本/外接设备上会是 60 fps，请用 `cap.get(cv2.CAP_PROP_FPS)` 实际查询。

## 7. 已知坑

- **Windows 摄像头权限**：首次运行需在系统设置里授权 Python。
- **MediaPipe wheel**：3.13+ 暂无官方 wheel；用 3.10/3.11，或忽略它走 Haar。
- **OpenCV 与 conda**：如果同时装了 `opencv-python` 和 `opencv-python-headless` 会冲突，只装前者。
- **首 15 秒 V/A 为 None**：第一个 VA 窗口尚未关闭，正常。UI 显示 `--`。
- **首 15 秒 Fatigue 为 None**：与 VA 同步，第一个窗口关闭前为 None。
- **首 5 秒 HR 为 NaN**：HRV 窗未填够，正常。
- **VA mode 显示为 vision**：通常表示当前 rPPG 质量偏低，auto 模式已回退到单模态；画质恢复后会自动切回 multimodal。

## 8. 未来扩展指引

- **打包为 .exe**：用 `pyinstaller --collect-all mediapipe --collect-all onnxruntime examples/run_webcam.py`。注意把 `stateguard/weights/` 加入 `--add-data`。
- **GPU 加速（非必须）**：把 ONNX Runtime 换成 `onnxruntime-gpu`，providers 里加 `CUDAExecutionProvider`。当前 CPU 已 3× 实时，不需要 GPU。
- **接入用户界面**：建议 Tauri / Electron + Python 后端（WebSocket 推 `FrameResult` JSON），或直接 PySide6 / Tkinter。
- **干预 / 报告模块**：基于 `FrameResult` 的时间序列在上层累积；本包只负责传感+推理。
