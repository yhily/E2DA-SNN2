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
- Source: [Roboflow Universe](https://universe.roboflow.com/nata-zlj1h/coffee_fob-gekr0/dataset)
- Dataset license reported by the source: CC BY 4.0

COFFEE_FOB is the primary benchmark used for ablation and model comparison.

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
