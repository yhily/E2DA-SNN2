# E2DA-SNN

**Evidence-Preserving Low-Timestep Spiking Detection via Attention and Cross-Level Interaction**

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
E2DA-SNN2/
├── data/                  # dataset and hyperparameter YAML files
├── experiments/           # repeated-run and device-benchmark protocol
├── models/                # E2DA-SNN architecture and spiking layers
├── tools/                 # experiment orchestration and benchmarking
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

- [COFFEE_FOB provider page](https://universe.roboflow.com/nata-zlj1h/coffe_fobv5)
  — the primary benchmark family for ablation and comparison. The provider's
  current versions are not file-identical to the historical 2,365-image split.
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
  --device 0 \
  --seed 1
```

Train the lightweight model by replacing the configuration:

```bash
python train.py --cfg models/e2da_lite.yaml --data data/data.yaml --weights ''
```

To fine-tune a checkpoint, pass its path to `--weights`. Training outputs are
written to `runs/train/` by default. Every completed run also writes a
`metrics.json` file containing the final precision, recall, mAP, validation
losses, seed, environment, and elapsed time.

## Repeated-run uncertainty

Run three independent trainings while keeping every setting except the random
seed fixed:

```bash
python tools/repeat_train.py \
  --seeds 1 2 3 \
  --cfg models/e2da.yaml \
  --data data/data.yaml \
  --hyp data/hyp.yaml \
  --epochs 100 \
  --batch-size 4 \
  --device 0 \
  --deterministic
```

The runner creates per-seed records, mean and sample-standard-deviation
summaries, and a generated LaTeX table under `runs/repeated/`. See
[`experiments/README.md`](experiments/README.md) for the fixed-control protocol
and interpretation limits.

### Retained-sample recovery test

The original file-level split is no longer available. The repository therefore
includes a separate, deliberately limited recovery workflow for the 20 retained
format samples. It must not be treated as a reconstruction of the manuscript's
primary benchmark:

```bash
python tools/reconstruct_sample_split.py
python tools/repeat_train.py \
  --seeds 1 2 3 \
  --cfg models/e2da_lite.yaml \
  --data data/reconstructed_sample.yaml \
  --hyp data/hyp_reconstructed_finetune.yaml \
  --weights weights/e2da_lite_download/best.pt \
  --epochs 15 \
  --batch-size 2 \
  --imgsz 256 \
  --device cpu \
  --deterministic \
  --output-dir runs/reconstructed_finetune \
  --run-prefix e2da_lite_ft \
  --extra-args --noautoanchor
```

The split builder records sample identifiers, class counts, and SHA-256 hashes
in `dataset/reconstructed_sample/manifest.json`. Because the public checkpoint
may have seen some retained images during its historical training, the recovery
test measures pipeline and seed stability only; its accuracy is not an
independent estimate of generalization. The completed per-seed and summary
records are published under `experiments/reconstructed_finetune/`.

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

## Device latency and power

Measure synchronized, model-forward latency with a trained checkpoint:

```bash
python tools/benchmark_device.py \
  --weights weights/e2da.pt \
  --device 0 \
  --imgsz 640 \
  --batch-size 1
```

On NVIDIA Jetson, add `--power-backend tegrastats` to sample `VDD_IN` board
input power. On a discrete NVIDIA GPU, `--power-backend nvidia-smi` samples GPU
power only. The generated JSON and CSV files record the device, software,
precision, warm-up, number of timed iterations, latency distribution, power
scope, and energy per image when power samples are available.

## Reproducibility notes

- The public datasets contain static images; they are replicated across time
  and are not event-camera recordings.
- The manuscript's primary comparison and ablation tables remain the original
  single-seed experiments. A separate three-seed retained-sample recovery test
  is reported with mean and sample standard deviation; it does not replace the
  unavailable full-benchmark repeated runs.
- The exact file-level manifests of the historical 7:2:1 dataset splits were
  not preserved. Reproducing the published partition therefore requires a new,
  explicitly versioned split manifest rather than assuming a file-identical
  reconstruction.
- The retained-sample recovery test contains only 20 images. It is reported
  separately from the main tables and cannot establish statistical
  significance or replace full-benchmark retraining.
- The paper's energy values are operation-level estimates under a shared
  reference process. They are not wall-plug or neuromorphic-chip measurements.
- Dataset splits, image size, hyperparameters, model YAML, and checkpoint must
  be kept fixed when comparing ablations.
- Random seeds improve repeatability but do not guarantee bitwise-identical
  CUDA results across hardware or software versions.
- Operation-count energy, Jetson board-input power, and discrete-GPU power are
  different quantities and must not be presented as interchangeable.

## Acknowledgements

The training, evaluation, and detection utilities follow the configuration-
driven design of the YOLO family. E2DA-SNN adds the spiking layers, DAEB, CFIM,
and the associated multi-scale architecture used in the paper.
