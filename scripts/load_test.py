"""Load test /predict with concurrent clients and report latency percentiles.

Usage:
    python scripts/load_test.py [--url http://127.0.0.1:8000] [--image path] [--n 64]
Phases: concurrency 1, 4, 16, 32 — each sends --n requests.
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

CONCURRENCY_PHASES = [1, 4, 16, 32]


def send_request(url: str, image_bytes: bytes):
    boundary = "----loadtestboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="img.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{url}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            return time.perf_counter() - t0, resp.status
    except urllib.error.HTTPError as e:
        e.read()
        return time.perf_counter() - t0, e.code
    except Exception:
        return time.perf_counter() - t0, 0


def run_phase(url: str, image_bytes: bytes, concurrency: int, n: int):
    latencies, statuses = [], Counter()
    lock = threading.Lock()
    per_thread = n // concurrency

    def worker():
        local_lat, local_status = [], Counter()
        for _ in range(per_thread):
            lat, status = send_request(url, image_bytes)
            local_lat.append(lat * 1000)
            local_status[status] += 1
        with lock:
            latencies.extend(local_lat)
            statuses.update(local_status)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0

    latencies.sort()
    k = len(latencies)
    total = sum(statuses.values())
    return {
        "concurrency": concurrency,
        "requests": total,
        "p50_ms": round(latencies[k // 2], 1),
        "p95_ms": round(latencies[int(k * 0.95)], 1),
        "p99_ms": round(latencies[min(int(k * 0.99), k - 1)], 1),
        "throughput_rps": round(total / wall, 1),
        "status_counts": dict(statuses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", default="dataset/test/fake")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--phases", default=None)
    args = parser.parse_args()
    if args.phases:
        global CONCURRENCY_PHASES
        CONCURRENCY_PHASES = [int(c) for c in args.phases.split(",")]

    import glob
    import os

    if os.path.isfile(args.image):
        img_path = args.image
    else:
        found = sorted(glob.glob(os.path.join(args.image, "*.jpg")))
        if not found:
            raise SystemExit(f"no images under {args.image}")
        img_path = found[0]
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    print(f"load testing {args.url}/predict with {img_path} ({len(image_bytes)} bytes)")

    print("warmup...")
    for _ in range(10):
        send_request(args.url, image_bytes)

    results = [run_phase(args.url, image_bytes, c, args.n) for c in CONCURRENCY_PHASES]
    print(json.dumps(results, indent=2))
    with open("models/load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote models/load_test_results.json")


if __name__ == "__main__":
    main()
