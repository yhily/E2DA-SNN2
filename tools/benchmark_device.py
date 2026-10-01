"""Benchmark model-only latency and optional device power for E2DA-SNN."""

import argparse
import csv
import json
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.torch_utils import select_device


class PowerSampler:
    """Collect timestamped power samples from tegrastats or nvidia-smi."""

    def __init__(self, backend, interval_ms):
        self.backend = backend
        self.interval_ms = interval_ms
        self.process = None
        self.thread = None
        self.samples = []

    @property
    def scope(self):
        if self.backend == "tegrastats":
            return "Jetson VDD_IN board input power"
        if self.backend == "nvidia-smi":
            return "NVIDIA GPU power.draw only"
        return "latency only; no hardware power samples"

    def command(self):
        executable = shutil.which(self.backend)
        if executable is None:
            raise FileNotFoundError(f"Power sampler not found on PATH: {self.backend}")
        if self.backend == "tegrastats":
            return [executable, "--interval", str(self.interval_ms)]
        return [
            executable,
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]

    def parse_watts(self, line):
        if self.backend == "tegrastats":
            match = re.search(r"VDD_IN\s+(\d+(?:\.\d+)?)mW", line)
            return float(match.group(1)) / 1000.0 if match else None
        match = re.search(r"(\d+(?:\.\d+)?)", line)
        return float(match.group(1)) if match else None

    def start(self):
        if self.backend == "none":
            return
        self.process = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        def read_output():
            for line in self.process.stdout:
                watts = self.parse_watts(line)
                if watts is not None:
                    self.samples.append((time.perf_counter(), watts))

        self.thread = threading.Thread(target=read_output, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=1)

    def values_between(self, start, end):
        return [watts for timestamp, watts in self.samples if start <= timestamp <= end]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="trained .pt checkpoint")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--half", action="store_true", help="use FP16 on CUDA")
    parser.add_argument("--power-backend", choices=("none", "tegrastats", "nvidia-smi"), default="none")
    parser.add_argument("--power-interval-ms", type=int, default=100)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "benchmarks")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def latex_escape(value):
    return str(value).replace("\\", "\\textbackslash{}") \
        .replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def latex_table(result):
    rows = [
        ("Mean latency (ms/image)", f"{result['latency_ms_per_image_mean']:.3f}"),
        ("Latency standard deviation (ms/image)", f"{result['latency_ms_per_image_std']:.3f}"),
        ("Median latency (ms/image)", f"{result['latency_ms_per_image_median']:.3f}"),
        ("90th-percentile latency (ms/image)", f"{result['latency_ms_per_image_p90']:.3f}"),
        ("Throughput (images/s)", f"{result['throughput_images_per_second']:.2f}"),
    ]
    optional = (
        ("Idle power (W)", "idle_power_w_mean"),
        ("Active power (W)", "active_power_w_mean"),
        ("Dynamic power (W)", "dynamic_power_w_mean"),
        ("Total energy (J/image)", "energy_per_image_j_total"),
        ("Dynamic energy (J/image)", "energy_per_image_j_dynamic"),
    )
    for label, key in optional:
        if result[key] is not None:
            rows.append((label, f"{result[key]:.6f}"))
    body = "\n".join(f"    {label} & {value} \\\\" for label, value in rows)
    return "\n".join([
        "% Generated by tools/benchmark_device.py.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Model-forward benchmark on {latex_escape(result['device_name'])}; "
        f"batch size {result['batch_size']}, {result['precision'].upper()}.}}",
        "  \\label{tab:device_benchmark}",
        "  \\begin{tabular}{lc}",
        "    \\toprule",
        "    Measurement & Value \\\\ ",
        "    \\midrule",
        body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "% Latency scope: " + result["latency_scope"],
        "% Power scope: " + result["power_scope"],
        "",
    ])


def main():
    args = parse_args()
    if args.iterations < 2:
        raise ValueError("Use at least two timed iterations.")
    if args.half and args.device == "cpu":
        raise ValueError("FP16 benchmarking is only supported on CUDA in this repository.")

    device = select_device(args.device)
    model = attempt_load(args.weights, device=device, fuse=True).eval()
    use_half = args.half and device.type == "cuda"
    model.half() if use_half else model.float()
    dtype = torch.float16 if use_half else torch.float32
    image = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(image)
        synchronize(device)

        sampler = PowerSampler(args.power_backend, args.power_interval_ms)
        sampler.start()
        idle_start = time.perf_counter()
        if args.power_backend != "none":
            time.sleep(args.idle_seconds)
        idle_end = time.perf_counter()

        latencies_ms = []
        active_start = time.perf_counter()
        for _ in range(args.iterations):
            synchronize(device)
            start = time.perf_counter()
            model(image)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        active_end = time.perf_counter()
        sampler.stop()

    per_image_ms = np.asarray(latencies_ms, dtype=np.float64) / args.batch_size
    batch_mean_seconds = statistics.fmean(latencies_ms) / 1000.0
    idle_power = sampler.values_between(idle_start, idle_end)
    active_power = sampler.values_between(active_start, active_end)
    idle_power_mean = mean_or_none(idle_power)
    active_power_mean = mean_or_none(active_power)
    dynamic_power_mean = None
    energy_total_j = None
    energy_dynamic_j = None
    if active_power_mean is not None:
        energy_total_j = active_power_mean * batch_mean_seconds / args.batch_size
        if idle_power_mean is not None:
            dynamic_power_mean = max(active_power_mean - idle_power_mean, 0.0)
            energy_dynamic_j = dynamic_power_mean * batch_mean_seconds / args.batch_size

    device_name = platform.processor() or platform.machine()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    result = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(Path(args.weights).resolve()),
        "device": str(device),
        "device_name": device_name,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "fp16" if use_half else "fp32",
        "batch_size": args.batch_size,
        "image_size": args.imgsz,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_scope": "model forward only; excludes image decoding, preprocessing, NMS, and result serialization",
        "latency_ms_per_image_mean": float(per_image_ms.mean()),
        "latency_ms_per_image_std": float(per_image_ms.std(ddof=1)),
        "latency_ms_per_image_median": float(np.median(per_image_ms)),
        "latency_ms_per_image_p90": float(np.percentile(per_image_ms, 90)),
        "latency_ms_per_image_p95": float(np.percentile(per_image_ms, 95)),
        "throughput_images_per_second": float(1000.0 / per_image_ms.mean()),
        "power_backend": args.power_backend,
        "power_scope": sampler.scope,
        "idle_power_w_mean": idle_power_mean,
        "active_power_w_mean": active_power_mean,
        "dynamic_power_w_mean": dynamic_power_mean,
        "energy_per_image_j_total": energy_total_j,
        "energy_per_image_j_dynamic": energy_dynamic_j,
        "power_sample_count_idle": len(idle_power),
        "power_sample_count_active": len(active_power),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("device_benchmark_%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    tex_path = args.output_dir / f"{stem}.tex"
    json_path.write_text(json.dumps(result, indent=2))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    tex_path.write_text(latex_table(result))

    print(json.dumps(result, indent=2))
    print(f"Benchmark outputs: {json_path}, {csv_path}, and {tex_path}")


if __name__ == "__main__":
    main()
