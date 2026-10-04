# Dataset notes

The full datasets are not redistributed in this repository. The files under
`images_samples/` and `labels_samples/` are a small COFFEE_FOB subset provided
only to illustrate the expected image and YOLO-label formats.

## Public benchmarks

### COFFEE_FOB

- Task: four-class coffee-fruit maturity detection
- Images used in the paper: 2,365
- Image resolution: 640 × 640
- Annotation format: YOLO
- Current provider page: [Roboflow Universe](https://universe.roboflow.com/nata-zlj1h/coffe_fobv5)
- Dataset license reported by the source: CC BY 4.0

COFFEE_FOB is the primary benchmark used for ablation and model comparison.
The provider's current versions differ in image count and split definition from
the historical 2,365-image study subset; they must not be presented as a
file-identical recovery of the published benchmark.

## Retained-sample recovery test

`tools/reconstruct_sample_split.py` creates a deterministic 12/4/4 split from
the 20 retained image/label pairs. Its manifest records SHA-256 hashes and
per-class box counts. This split is only a reproducibility stress test. It is
too small to replace the historical benchmark, and accuracy obtained from the
public checkpoint may be affected by overlap with its original training data.

### Cherry

- Task: three-class cherry maturity detection
- Approximate image count: 2,700
- Image resolution: 640 × 640
- Annotation format: YOLO
- Source: [Roboflow Universe](https://universe.roboflow.com/project-k2yri/cherry-tnrjs/dataset)
- Dataset license reported by the source: CC BY 4.0

Cherry is used as a secondary transfer benchmark.

## Expected layout

After downloading and extracting a dataset, arrange it as follows or update
the paths in `data/data.yaml` to match your export:

```text
dataset_root/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label row must use normalized YOLO format:

```text
class_id x_center y_center width height
```

Class identifiers must follow the `names` order in the selected dataset YAML.
