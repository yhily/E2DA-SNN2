# Datasets

This repository provides the implementation of the proposed **E2DA-SNN** model.
Due to GitHub storage limitations, the full datasets are hosted on external platforms.
Only a small number of sample images and annotation files are provided in this repository
for illustration and reproducibility purposes.

The **COFFEE_FOB** and **Cherry** datasets are public datasets obtained from Roboflow.
They contain images captured under real-world orchard environments, including dense fruit
distributions, non-uniform illumination, and partial occlusions.
Among them, **COFFEE_FOB** serves as the primary benchmark dataset in our experiments.

---

## COFFEE_FOB Dataset

- Task: Coffee fruit maturity detection
- Number of images: 2365
- Image resolution: 640x640
- Annotations: YOLO format
- Roboflow workspace: nata-zlj1h
- Roboflow project: coffee_fob-gekr0
- License: CC BY 4.0
- Download link: https://universe.roboflow.com/nata-zlj1h/coffee_fob-gekr0/dataset

---

## Cherry Dataset

- Task: Fruit maturity detection  
- Number of images: 2700
- Image resolution: 640x640
- Annotations: YOLO format
- Roboflow workspace: project-k2yri
- Roboflow project: cherry-tnrjs
- License: CC BY 4.0
- Download link: https://universe.roboflow.com/project-k2yri/cherry-tnrjs/dataset

---

## Sample Files

The `images_samples/` and `labels_samples/` directories provide a small subset of example
images and corresponding annotation files from the **COFFEE_FOB** dataset. These samples
are included to illustrate the data format and labeling style used in our experiments.

---

## Dataset Structure

After downloading and extracting the datasets, the directory structure should follow the
standard YOLO format:
```text
COFFEE_FOB/
|-- images/
|   |-- train/
|   |-- val/
|   |-- test/
|-- labels/
|   |-- train/
|   |-- val/
|   |-- test/
|-- data.yaml
```

The Cherry dataset follows the same directory structure.

