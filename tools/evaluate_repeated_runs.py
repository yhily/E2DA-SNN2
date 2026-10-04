"""Evaluate completed repeated runs on a held-out split and summarize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import val  # noqa: E402


METRICS = ("precision", "recall", "map50", "map50_95")


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the artifact is inside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--data", default="data/reconstructed_sample.yaml")
    parser.add_argument("--task", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    records = []
    for seed in args.seeds:
        run_dir = runs_dir / f"{args.run_prefix}_seed{seed}"
        checkpoint = run_dir / "weights" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        results, _, speed = val.run(
            data=args.data,
            weights=str(checkpoint),
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            task=args.task,
            device=args.device,
            workers=0,
            project=runs_dir / "evaluation",
            name=f"{args.run_prefix}_seed{seed}_{args.task}",
            exist_ok=True,
            plots=False,
        )
        record = {name: float(value) for name, value in zip(METRICS, results[:4])}
        record.update({
            "seed": seed,
            "split": args.task,
            "checkpoint": portable_path(checkpoint),
            "preprocess_ms_per_image": float(speed[0]),
            "inference_ms_per_image": float(speed[1]),
            "nms_ms_per_image": float(speed[2]),
        })
        records.append(record)

    summary = {}
    for metric in METRICS:
        values = [record[metric] for record in records]
        summary[metric] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    payload = {
        "scope": "retained-sample recovery test; not an independent full-benchmark estimate",
        "split": args.task,
        "seeds": args.seeds,
        "metrics": summary,
        "runs": records,
    }
    (runs_dir / f"{args.task}_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (runs_dir / f"{args.task}_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with (runs_dir / f"{args.task}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "n", "mean", "std", "min", "max"))
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"metric": metric, **values})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
