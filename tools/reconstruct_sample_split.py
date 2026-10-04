"""Build a deterministic train/validation/test split from the retained samples.

This utility is intentionally conservative: the 20 retained images are not a
replacement for the historical COFFEE_FOB benchmark.  They are used only for
a small reproducibility stress test when the original file-level split is no
longer available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("overripe", "ripe", "semi_ripe", "unripe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in path.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def split_score(groups: dict[str, list[str]], counts: dict[str, Counter[int]]) -> float:
    total = sum((counts[name] for name in counts), Counter())
    score = 0.0
    for split, names in groups.items():
        observed = sum((counts[name] for name in names), Counter())
        target_fraction = len(names) / len(counts)
        for class_id in range(len(CLASS_NAMES)):
            target = total[class_id] * target_fraction
            score += ((observed[class_id] - target) / max(target, 1.0)) ** 2
            if observed[class_id] == 0:
                score += 100.0
    return score


def choose_split(names: list[str], counts: dict[str, Counter[int]], seed: int):
    """Search deterministic candidate permutations for a class-balanced 12/4/4 split."""
    rng = random.Random(seed)
    best = None
    best_score = float("inf")
    for _ in range(100_000):
        candidate = names[:]
        rng.shuffle(candidate)
        groups = {
            "train": sorted(candidate[:12], key=lambda value: int(value)),
            "val": sorted(candidate[12:16], key=lambda value: int(value)),
            "test": sorted(candidate[16:], key=lambda value: int(value)),
        }
        score = split_score(groups, counts)
        if score < best_score:
            best, best_score = groups, score
    return best, best_score


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261003,
                        help="fixed seed used only to construct the data split")
    parser.add_argument("--source", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dataset" / "reconstructed_sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.source / "images_samples"
    label_dir = args.source / "labels_samples"
    names = sorted((path.stem for path in image_dir.glob("*.jpg")), key=int)
    if len(names) != 20:
        raise ValueError(f"Expected 20 retained JPG samples, found {len(names)}")
    missing = [name for name in names if not (label_dir / f"{name}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing labels for retained samples: {missing}")

    counts = {name: label_counts(label_dir / f"{name}.txt") for name in names}
    groups, score = choose_split(names, counts, args.seed)

    records = []
    for split, split_names in groups.items():
        for name in split_names:
            image_source = image_dir / f"{name}.jpg"
            label_source = label_dir / f"{name}.txt"
            image_target = args.output / split / "images" / image_source.name
            label_target = args.output / split / "labels" / label_source.name
            link_or_copy(image_source, image_target)
            link_or_copy(label_source, label_target)
            records.append({
                "split": split,
                "sample_id": name,
                "image_sha256": sha256(image_source),
                "label_sha256": sha256(label_source),
                "class_box_counts": {
                    CLASS_NAMES[class_id]: counts[name][class_id]
                    for class_id in range(len(CLASS_NAMES))
                },
            })

    summary = {}
    for split, split_names in groups.items():
        split_counts = sum((counts[name] for name in split_names), Counter())
        summary[split] = {
            "images": len(split_names),
            "boxes": sum(split_counts.values()),
            "class_box_counts": {
                CLASS_NAMES[class_id]: split_counts[class_id]
                for class_id in range(len(CLASS_NAMES))
            },
        }

    manifest = {
        "purpose": "retained-sample reproducibility stress test; not a reconstruction of the historical benchmark split",
        "split_seed": args.seed,
        "search_objective": "class-box-distribution balance with every class represented in each split",
        "search_score": score,
        "class_order": list(CLASS_NAMES),
        "summary": summary,
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
