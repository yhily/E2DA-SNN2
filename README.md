<div align="center">

## Event-Driven Dual-Attention Spiking Neural Network for Energy-Efficient Detection of Coffee Fruit Maturity
</div>

This repository provides the official implementation of **E2DA-SNN**, an event-driven spiking neural network for energy-efficient object detection.  

The code is released to support reproducibility of the experimental results reported in the paper.

---

### Requirements

The code has been tested with the following environment:

- Python: >= 3.8 (推荐 3.10+)
- PyTorch: >= 2.0（已适配 2.0+，验证环境：PyTorch 2.9 / Python 3.13）
- CUDA: 11.7+（2.x 对应版本）

<details open>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>

### SNN 时间窗（Time Window）

所有 SNN 层（`mem_update`、`Snn_Conv2d`、`Pools`、`Sample` 等）的时间步数现在统一从
输入张量第 0 维动态推导，唯一的时间窗常量为 `models/snn_layers.py` 中的 `TIME_WINDOW`（默认 4），
可通过模型 yaml 的 `time_window` 字段覆盖（如 `time_window: 2`）。
原代码中 `snn_layers.py`(2)、`modules.py`(2)、`model.py`(4) 各有一份互不相同的 `time_window`，
会导致模型只消费前 2 个时间步，本仓库已统一修复。

</details>

### Datasets

This project uses two public datasets for fruit maturity detection:

- **COFFEE_FOB**: Primary benchmark dataset, containing 2365 images captured under real-world orchard conditions, including dense fruit distributions, non-uniform illumination, and partial occlusions. [Download link](https://universe.roboflow.com/nata-zlj1h/coffee_fob-gekr0/dataset)

- **Cherry**: Secondary dataset, containing 2700 images with similar characteristics. [Download link](https://universe.roboflow.com/project-k2yri/cherry-tnrjs/dataset)

A small subset of sample images and labels are included in this repository (`dataset/images_samples` and `dataset/labels_samples`) to illustrate the data format. Please download the full datasets from the links above for training and evaluation.

### Pretrained Checkpoints

We provide pretrained [E2DA](https://drive.google.com/file/d/1VsOd_sk0Wf6R8SFKw9sLLP9xopR7hcHw/view?usp=sharing) and [E2DA-Lite](https://drive.google.com/file/d/1Kmgp-MgIiW2n2igNH68Xw-21OB3YuJxx/view?usp=sharing) models on the COFFEE_FOB dataset.

### Training 
<details open>
<summary>Train</summary>

Train the standard E2DA model:
```bash
python train.py --cfg models/e2da.yaml --data data.yaml --weights path/to/weights.pt
```
Train the lightweight E2DA-Lite model:
```bash
python train.py --cfg models/e2da_lite.yaml --data data.yaml --weights path/to/weights.pt
```
可选（PyTorch 2.0+）：使用 `torch.compile` 加速训练（单卡/CPU，失败自动回退）：
```bash
python train.py --cfg models/e2da.yaml --data data.yaml --compile default
```

</details>

### Evaluation
<details open> <summary>Validate / Evaluate Models</summary>

Evaluate a trained model:
```bash
python val.py --weights path/to/weights.pt --data data.yaml
```

</details>
