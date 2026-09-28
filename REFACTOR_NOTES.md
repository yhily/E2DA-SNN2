# E2DA-SNN 重构说明（PyTorch 2.0+ 适配）

本仓库原代码基于 PyTorch 1.10 / Python 3.8 编写，且存在多处会使模型**根本无法构建/运行的确定性 Bug**。
本次重构以「**保留所有类名与模块路径，旧 checkpoint 可直接加载**」为原则，修复并优化如下。

验证环境：Python 3.13.5 + PyTorch 2.9.1 + torchvision 0.24.1 + NumPy 2.1.3（macOS CPU）。

---

## 一、PyTorch 2.x API 适配（14 处）

| 位置 | 修改 |
| --- | --- |
| `train.py` / `models/modules.py` / `utils/autobatch.py` | `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda', ...)`；`amp.GradScaler` → `torch.amp.GradScaler('cuda', ...)`（2.4+ 弃用旧路径） |
| `train.py` / `models/experimental.py` / `utils/general.py` | 所有 `torch.load` 增加 `weights_only=False`（PyTorch 2.6+ 默认改为 True，否则无法加载含 nn.Module 的 checkpoint） |
| `models/model.py` | `_make_grid` 直接使用 `indexing='ij'`，删除 1.10 版本兼容分支 |
| `models/model.py` | `_profile_one_layer` 中 `x.copy()` → `x.clone()`（tensor.copy() 已弃用） |
| `utils/general.py` | `pkg_resources.parse_version/parse_requirements` → `packaging` + `importlib.metadata`（pkg_resources 已弃用） |
| `models/modules.py` | coreml 分支 `np.float` → `np.float64`（NumPy 1.24+ 已移除 `np.float`） |
| `utils/metrics.py` | `np.trapz` → `np.trapezoid`（NumPy 2.0 改名） |
| `utils/plots.py` | `ImageFont.getsize` → `getbbox`（Pillow 10+ 移除） |
| `utils/datasets.py` | macOS 下 DataLoader `workers` 默认降为 0（避免解释器退出时 waitpid 挂起） |

## 二、确定性运行时 Bug 修复（11 处，均为原代码无法运行/行为错误的根因）

| 位置 | 原问题 | 修复 |
| --- | --- | --- |
| `models/modules.py` | `from snn_layers import *` 找不到模块 | 改为 `from models.snn_layers import *` |
| `models/modules.py` | yaml 引用的 `Snn_Conv2d_BN` **从未定义**，模型构建即 NameError | 补全定义（Snn_Conv2d + 时空 BatchNorm） |
| `models/model.py` | `parse_model` 引用了大量本仓库不存在的类（GhostConv/SPP/C3…） | 裁剪为实际存在的类 |
| `models/model.py` | Fusion 被通用逻辑错误前置 c1 参数（且 `ch[f]` 多路输入为列表） | Fusion 特判，args 保持 `(c2, k, ratio)` |
| `models/modules.py` | `Fusion.forward(x0, x1)` 与 YOLO 多路输入列表约定不符 | 改为 `forward(x)` 接收列表 |
| `models/modules.py` | `_model_type` 依赖仓库中不存在的 `export.py`，DetectMultiBackend 无法加载任何模型 | 内联后缀表 |
| `models/experimental.py` | `from models.model import Conv` 造成循环导入，星号导入丢失 CrossConv/MixConv2d（依赖导入顺序） | 改为 `from models.modules import Conv` |
| `train.py` | `Loggers` 未定义（日志集成被裁剪后残留调用） | 补最小兼容层 |
| `train.py` | 默认 `--hyp` 指向不存在的 `data/hyps/hyp.yaml` | 改为 `data/hyp.yaml` |
| `models/snn_layers.py` 等 | `time_window` 在三个模块分别定义为 2/2/4，互不相同 → 模型平铺 4 步却只消费前 2 步 | 统一为 `TIME_WINDOW`（默认 4），各层从输入第 0 维动态推导时间步；yaml 可加 `time_window:` 覆盖 |
| `models/model.py` | `fuse()` 引用未定义的 `DWConv` | 只融合标准 2D `Conv`+BN |

## 三、性能优化（3 处）

- `Snn_Conv2d`：时间维卷积由 T 次循环改为一次批量卷积（(T,B) 合并后单次 `F.conv2d`）。**数值完全等价（最大误差 0），CPU 实测快 1.31×**，GPU 上省去 T 次 kernel launch 收益更大。
- `Pools` / `zeropad` / `Sample`：同样批量化为单次池化/填充/插值。
- `Model.forward` 平铺改为 `x.unsqueeze(0).repeat(T,1,1,1,1)`，无逐步拷贝开销。

## 四、死代码与副作用清理（6 处）

- `detect.py`：删除每次运行都在当前目录写空数组的 `firerate10_5.npy` / `size10_5.npy`；删除 2000 张图的隐性截断；删除强制 `save_img=True`（`--nosave` 现在真正生效）。
- `models/experimental.py`：删除被注释掉的旧版 `attempt_load` 死代码。
- `train.py` / `val.py` / `detect.py`：`ROOT` 不再相对 cwd 化，默认路径在任何目录下运行都有效；删除调试打印。

---

## 验证结果（全部在本机 PyTorch 2.9.1 CPU 实测）

1. `e2da.yaml`（34.65M 参数）与 `e2da_lite.yaml`（16.05M 参数）：构建 → forward → loss → backward 全部通过，**全部参数均获得有效梯度**。
2. checkpoint 往返：保存 → `strip_optimizer` → `attempt_load` → `fuse` → 推理，正常。
3. `DetectMultiBackend` + NMS：推理输出形状正确（如 640×640 下 `(1, 25200, 9)`）。
4. `val.run` 完整评估流程通过（mAP/PR 计算正常）。
5. `detect.py` CLI 20 张样本图全部处理完成（320px 下约 192ms/图）。
6. `train.py` 1 个 epoch 端到端跑通：训练 → 存权重 → 剥离优化器 → 重载 → 最终验证。
7. 开启 `warnings.filterwarnings('error')` 回归：无任何弃用/兼容性警告。

## 使用

```bash
# 训练（PyTorch 2.0+ 环境）
python train.py --cfg models/e2da.yaml --data data.yaml --weights ''

# 可选：torch.compile（PyTorch 2.0 特性，默认关闭；SNN 5D 张量+自定义 autograd 在 CPU 上可能挂起，建议仅 CUDA 尝试，失败自动回退）
python train.py --cfg models/e2da.yaml --data data.yaml --compile default

# 验证 / 推理
python val.py --weights path/to/weights.pt --data data.yaml
python detect.py --weights path/to/weights.pt --source data/images --device 0
```

## 已知限制

- `torch.compile`：本 SNN 架构含 5D 时间维张量与就地膜电位写入，CPU 编译可能长时间挂起；`--compile` 为可选项且默认关闭，建议在 CUDA 环境实验。
- 时间窗统一为 `TIME_WINDOW=4`（原 `model.py` 的意图值）。如需复现旧代码实际等效的 2 步行为，在 yaml 中加 `time_window: 2` 即可。
- 所有类名与模块路径保持不变，官方 Google Drive 预训练权重可直接 `torch.load(..., weights_only=False)` 加载。
