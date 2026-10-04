# Repeated-run and device-efficiency experiments

This directory documents repeated-run uncertainty and deployment measurement.
The scripts generate machine-readable records, and
`reconstructed_finetune/` contains the completed retained-sample recovery
experiment reported in the manuscript.

## 1. Three independent training runs

Keep the dataset split, image size, hyperparameters, architecture, training
epochs, and software environment fixed. Change only the random seed. The
default seeds are `1`, `2`, and `3`.

```bash
python tools/repeat_train.py \
  --cfg models/e2da.yaml \
  --data data/data.yaml \
  --hyp data/hyp.yaml \
  --weights '' \
  --epochs 100 \
  --batch-size 4 \
  --imgsz 640 \
  --device 0 \
  --deterministic
```

Each run writes `metrics.json`. After all runs finish, the tool creates:

- `runs/repeated/runs.csv`: one row per seed;
- `runs/repeated/summary.csv` and `summary.json`: arithmetic mean and sample
  standard deviation (denominator `n-1`);
- `runs/repeated/summary_table.tex`: a paper-ready LaTeX table.

Interrupted runs are not silently overwritten. Completed runs are reused, and
the command can collect existing results without training:

```bash
python tools/repeat_train.py --collect-only --seeds 1 2 3
```

For ablations, repeat the same seed set for the baseline and the full model.
Three runs provide an uncertainty estimate, but they are usually too few for a
strong claim of statistical significance. Report observed mean differences
and standard deviations without turning a small gap into a significance claim.

## 2. Device latency and power

The benchmark measures model-forward latency after warm-up with explicit
device synchronization. The default report uses batch size 1, 50 warm-up
iterations, and 200 timed iterations.

Latency only:

```bash
python tools/benchmark_device.py \
  --weights weights/e2da.pt \
  --device 0 \
  --imgsz 640 \
  --batch-size 1
```

Jetson board-input power (`VDD_IN`) through `tegrastats`:

```bash
python tools/benchmark_device.py \
  --weights weights/e2da.pt \
  --device 0 \
  --power-backend tegrastats
```

Discrete NVIDIA GPU power only through `nvidia-smi`:

```bash
python tools/benchmark_device.py \
  --weights weights/e2da.pt \
  --device 0 \
  --power-backend nvidia-smi
```

The JSON, CSV, and generated LaTeX table state the measurement scope. `tegrastats` `VDD_IN`
approximates Jetson board-input power, whereas `nvidia-smi` reports GPU power
only. Neither should be described as neuromorphic-chip energy. The reported
model-forward latency excludes image decoding, preprocessing, NMS, and result
serialization; an end-to-end application benchmark must measure those stages
separately.

## 3. Paper reporting template

After the scripts have produced real values, report the protocol in this form:

> We repeated training with random seeds 1, 2, and 3 while holding the dataset
> split, hyperparameters, architecture, and training schedule fixed. Results
> are reported as arithmetic mean plus or minus sample standard deviation.
> Device latency was measured at batch size 1 after 50 warm-up iterations over
> 200 synchronized forward passes. Power was sampled using [backend] on
> [device]; the measurement scope was [scope]. Operation-count energy remains
> a hardware-independent estimate and is reported separately from measured
> device power.

Do not insert this paragraph into the manuscript until the bracketed device
fields and the generated numerical table have been checked against the output
files.

## 4. Completed retained-sample recovery experiment

The historical 2,365-image file-level split and its original run logs are no
longer available. To test whether the released training and evaluation path
can still be executed repeatedly, we built a deterministic 12/4/4 split from
the 20 retained annotated samples, fine-tuned the public E2DA-SNN-Lite
checkpoint for 15 epochs at 256 x 256 pixels, and changed only the random seed
across seeds 1, 2, and 3. All three runs used CPU execution, batch size 2, and
the fixed hyperparameters in `data/hyp_reconstructed_finetune.yaml`.

On the four-image held-out split (80 boxes), the mean plus or minus sample
standard deviation was:

| Metric | Mean +/- sample SD |
| --- | ---: |
| Precision | 28.10 +/- 6.65% |
| Recall | 42.50 +/- 7.35% |
| mAP@0.5 | 27.05 +/- 1.32% |
| mAP@0.5:0.95 | 15.27 +/- 1.38% |

The per-seed records and summaries are under `reconstructed_finetune/`; the
split identities, SHA-256 hashes, and class counts are recorded in
`dataset/reconstructed_sample/manifest.json`. These numbers are a recovery
stress test, not a replacement for the paper's historical benchmark. The test
set has only four images, and the public checkpoint may have encountered some
of the retained samples during its original training. The result therefore
supports executable-path and seed-stability claims only. It does not provide
an independent estimate of generalization or validate the historical accuracy
table.
