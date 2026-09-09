"""Benchmark single-image inference latency for trained models.

Usage: python scripts/benchmark_latency.py [--runs 200]
Reports mean / p50 / p95 / p99 latency (ms) per model on CPU and MPS.
"""
import argparse
import json
import statistics
import time

import torch

from model.Meso4.model import build_model as build_meso4
from model.SwinTransformer.model import build_model as build_swin


def get_available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def benchmark(model, device, runs, image_size=256):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "mps":
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    n = len(latencies)
    return {
        "device": device.type,
        "runs": n,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[int(n * 0.95)], 2),
        "p99_ms": round(latencies[int(n * 0.99)], 2),
        "throughput_img_per_s": round(1000 / latencies[n // 2], 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    results = {}
    models = {
        "meso4_28k_params": build_meso4(),
        "swin_t_27M_params": build_swin(num_classes=1),
    }
    for name, model in models.items():
        ckpt_path = f"models/{name.split('_')[0]}/best.pt"
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
        except FileNotFoundError:
            print(f"[warn] {ckpt_path} not found — benchmarking untrained weights")
        for device in get_available_devices():
            key = f"{name}/{device}"
            results[key] = benchmark(model, torch.device(device), args.runs)
            print(f"{key}: {results[key]}")

    with open("models/latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote models/latency_benchmark.json")


if __name__ == "__main__":
    main()
