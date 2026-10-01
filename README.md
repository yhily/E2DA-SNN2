# E2DA-SNN

**Energy-Efficient Dual-Attention Spiking Detection via Cross-Level Feature Interaction**

This repository contains the official PyTorch implementation of E2DA-SNN, a
directly trained spiking object detector designed for dense, small, and
partially occluded targets. Coffee and cherry maturity datasets are used as
validation testbeds; they do not define the scope of the method.

## Method overview

E2DA-SNN preserves weak spatial evidence under a short simulation horizon with
three complementary components:

- **DAEB — Dual-channel Attention Enhanced Block:** combines parameter-free
  SimAM reweighting with LIF-encoded average and max pooling.
- **CFIM — Cross-Level Feature Interaction Module:** learns content-dependent
  gates between aligned shallow and deep features.
- **P3 detection branch:** retains a high-resolution prediction surface for
  small and crowded objects, alongside the P4 and P5 branches.

The implementation uses the tensor convention `[time, batch, channels,
height, width]` inside the spiking network. Static RGB images are replicated
over `T=2` simulation steps before feature extraction.

| Paper component | Implementation |
| --- | --- |
| LIF dynamics and surrogate gradient | `models/snn_layers.py` |
| DAEB | `BasicBlock_3` in `models/modules.py` |
| CFIM | `Fusion` in `models/modules.py` |
| P3–P5 temporal detection head | `Detect` in `models/model.py` |
| Full and lightweight architectures | `models/e2da.yaml`, `models/e2da_lite.yaml` |

## Repository layout

```text
E2DA-SNN/
├── data/                  # dataset and hyperparameter YAML files
├── models/                # E2DA-SNN architecture and spiking layers
├── utils/                 # data loading, losses, metrics, and plotting
├── train.py               # training entry point
├── val.py                 # validation and test entry point
├── detect.py              # image/video inference entry point
└── requirements.txt
```

## Environment

The experiments reported in the paper used:

- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3
- cuDNN 8.2.0

Install a PyTorch build that matches your CUDA environment first, then install
the remaining dependencies:

```bash
pip install -r requirements.txt
```

Newer PyTorch versions may work, but they have not been used to reproduce the
reported results.

## Datasets

Two public datasets are used for evaluation:

- [COFFEE_FOB](https://universe.roboflow.com/nata-zlj1h/coffee_fob-gekr0/dataset)
  — the primary benchmark for ablation and comparison.
- [Cherry](https://universe.roboflow.com/project-k2yri/cherry-tnrjs/dataset)
  — a secondary transfer benchmark.

Download a dataset in YOLO format and update `path`, `train`, `val`, `test`,
`nc`, and `names` in `data/data.yaml`. A typical layout is:

```text
dataset_root/
├── train/images/          # training images
├── train/labels/          # YOLO labels
├── val/images/
├── val/labels/
├── test/images/
└── test/labels/
```

Use forward slashes in YAML paths, including on Windows. The class order in
`names` must match the integer class identifiers stored in the label files.

## Pretrained checkpoints

- [E2DA-SNN](https://drive.google.com/file/d/1VsOd_sk0Wf6R8SFKw9sLLP9xopR7hcHw/view?usp=sharing)
- [E2DA-SNN-Lite](https://drive.google.com/file/d/1Kmgp-MgIiW2n2igNH68Xw-21OB3YuJxx/view?usp=sharing)

Download a checkpoint and place it in a local directory such as `weights/`.

## Training

Train the full model from scratch:

```bash
python train.py \
  --cfg models/e2da.yaml \
  --data data/data.yaml \
  --hyp data/hyp.yaml \
  --weights '' \
  --epochs 100 \
  --batch-size 4 \
  --imgsz 640 \
  --device 0
```

Train the lightweight model by replacing the configuration:

```bash
python train.py --cfg models/e2da_lite.yaml --data data/data.yaml --weights ''
```

To fine-tune a checkpoint, pass its path to `--weights`. Training outputs are
written to `runs/train/` by default.

## Evaluation

```bash
python val.py \
  --weights weights/e2da.pt \
  --data data/data.yaml \
  --task test \
  --imgsz 640 \
  --device 0
```

The evaluator reports precision, recall, mAP@0.5, and mAP@0.5:0.95.

## Inference

```bash
python detect.py \
  --weights weights/e2da.pt \
  --source path/to/images_or_video \
  --imgsz 640 \
  --conf-thres 0.25 \
  --device 0
```

Predictions are saved under `runs/detect/` unless another output directory is
specified.

## Reproducibility notes

- The public datasets contain static images; they are replicated across time
  and are not event-camera recordings.
- The paper's energy values are operation-level estimates under a shared
  reference process. They are not wall-plug or neuromorphic-chip measurements.
- Dataset splits, image size, hyperparameters, model YAML, and checkpoint must
  be kept fixed when comparing ablations.
- Random seeds improve repeatability but do not guarantee bitwise-identical
  CUDA results across hardware or software versions.

## Acknowledgements

The training, evaluation, and detection utilities follow the configuration-
driven design of the YOLO family. E2DA-SNN adds the spiking layers, DAEB, CFIM,
and the associated multi-scale architecture used in the paper.
