"""Run independent E2DA-SNN trainings and summarize final metrics.

The script keeps the dataset split, model configuration, and hyperparameters
fixed while changing only the random seed. Each training run must emit the
``metrics.json`` file produced by ``train.py``.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRICS = ("precision", "recall", "map50", "map50_95")
ALL_METRICS = PRIMARY_METRICS + ("val_box_loss", "val_obj_loss", "val_cls_loss")
CONTROL_FIELDS = (
    "configuration", "dataset", "hyperparameters", "initial_weights",
    "image_size", "batch_size", "epochs_requested", "deterministic",
    "device", "torch_version", "cuda_version",
)
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP@0.5",
    "map50_95": "mAP@0.5:0.95",
    "val_box_loss": "Validation box loss",
    "val_obj_loss": "Validation objectness loss",
    "val_cls_loss": "Validation classification loss",
}


def portable_path(path):
    """Return a repository-relative path when the artifact is inside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cfg", default="models/e2da.yaml")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--hyp", default="data/hyp.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "repeated")
    parser.add_argument("--run-prefix", default="e2da")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--collect-only", action="store_true",
                        help="summarize completed run directories without launching training")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="additional arguments passed verbatim to train.py")
    return parser.parse_args()


def run_directory(output_dir, prefix, seed):
    return output_dir / f"{prefix}_seed{seed}"


def training_command(args, seed):
    command = [
        args.python,
        str(ROOT / "train.py"),
        "--cfg", args.cfg,
        "--data", args.data,
        "--hyp", args.hyp,
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--imgsz", str(args.imgsz),
        "--device", args.device,
        "--workers", str(args.workers),
        "--seed", str(seed),
        "--project", str(args.output_dir),
        "--name", f"{args.run_prefix}_seed{seed}",
        "--exist-ok",
    ]
    if args.deterministic:
        command.append("--deterministic")
    command.extend(args.extra_args)
    return command


def launch_runs(args):
    for seed in args.seeds:
        run_dir = run_directory(args.output_dir, args.run_prefix, seed)
        metrics_file = run_dir / "metrics.json"
        command = training_command(args, seed)
        print(" ".join(command))
        if args.dry_run:
            continue
        if metrics_file.exists():
            print(f"Reusing completed run: {metrics_file}")
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete run directory exists: {run_dir}. Resume or move it before rerunning."
            )
        subprocess.run(command, cwd=ROOT, check=True)


def read_records(args):
    records = []
    for seed in args.seeds:
        metrics_file = run_directory(args.output_dir, args.run_prefix, seed) / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing completed-run metrics: {metrics_file}")
        record = json.loads(metrics_file.read_text())
        if int(record["seed"]) != seed:
            raise ValueError(f"Seed mismatch in {metrics_file}: expected {seed}, found {record['seed']}")
        missing = [name for name in ALL_METRICS if name not in record]
        if missing:
            raise ValueError(f"Missing metrics in {metrics_file}: {missing}")
        record["metrics_file"] = portable_path(metrics_file)
        records.append(record)
    return records


def summarize(records):
    summary = {}
    for name in ALL_METRICS:
        values = [float(record[name]) for record in records]
        summary[name] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def validate_controls(records):
    """Reject summaries that mix different experimental controls."""
    reference = records[0]
    for field in CONTROL_FIELDS:
        if field not in reference:
            continue
        for record in records[1:]:
            if record.get(field) != reference[field]:
                raise ValueError(
                    f"Experimental control differs across runs: {field}="
                    f"{reference[field]!r} versus {record.get(field)!r}"
                )


def write_outputs(args, records, summary):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_fields = ["seed", *ALL_METRICS, "epochs_completed", "elapsed_seconds", "device",
                  "torch_version", "cuda_version", "metrics_file"]
    with open(args.output_dir / "runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with open(args.output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "n", "mean", "std", "min", "max"))
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"metric": metric, **values})

    payload = {
        "seeds": [int(record["seed"]) for record in records],
        "standard_deviation": "sample (n-1 denominator)",
        "metrics": summary,
        "runs": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for metric in PRIMARY_METRICS:
        values = summary[metric]
        rows.append(
            f"{METRIC_LABELS[metric]} & "
            f"{100 * values['mean']:.2f} $\\pm$ {100 * values['std']:.2f} \\\\"
        )
    latex = "\n".join([
        "% Generated by tools/repeat_train.py; values are percentages.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Repeated-run results over {len(records)} independent random seeds.}}",
        "  \\label{tab:repeated_runs}",
        "  \\begin{tabular}{lc}",
        "    \\toprule",
        "    Metric & Mean $\\pm$ standard deviation (\\%) \\\\ ",
        "    \\midrule",
        *(f"    {row}" for row in rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    (args.output_dir / "summary_table.tex").write_text(latex)


def main():
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if len(args.seeds) < 3:
        print("WARNING: fewer than three seeds is insufficient for the planned paper report.")
    args.output_dir = args.output_dir.resolve()
    if not args.collect_only:
        launch_runs(args)
    if args.dry_run:
        return
    records = read_records(args)
    validate_controls(records)
    summary = summarize(records)
    write_outputs(args, records, summary)
    print(f"Repeated-run summary written to {args.output_dir}")


if __name__ == "__main__":
    main()
