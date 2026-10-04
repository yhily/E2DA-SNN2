# Retained-sample three-seed recovery test

This directory contains the compact, machine-readable outputs of the recovery
experiment described in `experiments/README.md`.

## Fixed protocol

- model: E2DA-SNN-Lite;
- initialization: released E2DA-SNN-Lite checkpoint;
- samples: 20 retained annotated images;
- split: 12 train, 4 validation, and 4 test images;
- split seed: 20261003;
- training seeds: 1, 2, and 3;
- epochs: 15;
- image size: 256 x 256;
- batch size: 2;
- device: CPU;
- deterministic-training flag: enabled;
- changing factor across runs: random seed only.

`runs.csv` and `summary.*` contain terminal validation metrics from training.
`test_runs.csv` and `test_summary.*` contain the held-out test results. The
summary standard deviations use the sample definition with denominator n-1.

## Interpretation boundary

This experiment was reconstructed after the original full-dataset run records
and file-level split were lost. It cannot reproduce the manuscript's primary
benchmark. The four-image test subset is small, and overlap between the
retained samples and the historical training set of the released checkpoint is
unknown. Use these files to audit the executable workflow and the dispersion
across three seeds, not to infer full-dataset generalization or compare the
recovery scores directly with the main paper tables.
