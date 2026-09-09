# DL-Based Detection of GAN-Generated Face Images

End-to-end deepfake-image detection service: train CNN and Vision Transformer detectors on the
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
benchmark (70k real FFHQ photographs vs 70k StyleGAN-generated faces), evaluate them honestly on a
held-out test split, export to ONNX, and serve them behind a load-tested FastAPI API with bounded
concurrency and load shedding.

A live demo of an earlier iteration was hosted on Render; see [Reproducing](#reproducing) to run
everything locally.

---

## Results

All numbers are from held-out **test** splits the models never saw during training or model
selection. Splits are the dataset's official ones; seed = 42; positive class = fake.

| Model | Params | Test Acc | ROC-AUC | F1 | Latency p50 (MPS / CPU) | Throughput |
|---|---|---|---|---|---|---|
| Meso4 (Afchar et al., 2018) | 28K | 88.6% | 0.9555 | 0.889 | 1.8 ms / 3.6 ms | ~540 img/s (MPS) |
| Swin-T (fine-tuned) | 27.5M | **99.7%** | **1.000** | **0.997** | 31 ms / 96 ms | ~32 img/s (MPS) |

Swin-T confusion matrix on 10,000 test images (rows = truth, cols = prediction, order fake/real):

```
              pred fake   pred real
truth fake        4996          4
truth real          28       4972
```

### Serving performance (FastAPI + ONNX Runtime CPU, M4 Mac)

Measured with `scripts/load_test.py` while training another model was running — i.e. under
realistic contention.

| Concurrent clients | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| 1 | 3.2 ms | 3.7 ms | 4.2 ms | 305 rps |
| 4 | 5.8 ms | 7.6 ms | 8.6 ms | 679 rps |
| 16 | 20 ms | 23 ms | 23 ms | 756 rps |
| 32 | 37 ms | 44 ms | 46 ms | 738 rps |
| 128 | 98 ms | 120 ms | 123 ms | 1053 rps |

Throughput saturates around ~750–1000 rps with 4 inference workers; beyond saturation, queueing
latency grows while throughput stays flat, as expected.

### Backpressure & load shedding

`api/app.py` implements explicit admission control:

- A bounded semaphore caps in-flight requests at `INFERENCE_WORKERS + MAX_QUEUE` (default 4 + 64).
- Requests wait at most `QUEUE_TIMEOUT_S` (default 2 s) for a slot; past that they get
  **503 + Retry-After** instead of piling up unboundedly.
- The ONNX Runtime session is pinned to `intra_op_num_threads=INFERENCE_WORKERS` so concurrent
  inferences don't oversubscribe the CPU.

With Meso4 (~3 ms/request) the shed path never triggers — the server drains faster than requests
arrive even at 128 concurrent clients. The mechanism is sized for heavier models (Swin-T is
~100 ms/request on CPU) where it becomes necessary.

---

## Honest scope statement

This detects **StyleGAN-generated faces**, not face-swap deepfakes generally (FaceForensics++,
Celeb-DF, DFDC). The 1.000 ROC-AUC is *in-domain* — same generator family as training. Published
work shows such models drop sharply cross-dataset; evaluating that generalization gap is the
natural next step and is why the README does not claim universal deepfake detection.

---

## Architecture

```
dataset/ (ImageFolder: train|val|test × real|fake)
   │
   ├── scripts/prepare_data.py      parquet → JPEG folder tree (official splits preserved)
   │
train.py                            PyTorch training loop (Meso4 or Swin-T), seeded,
   │                                best checkpoint by val ROC-AUC → models/<name>/best.pt
   │
evaluate.py                         test-set metrics → JSON (acc, P/R/F1, ROC-AUC, confusion matrix)
   │
scripts/export_onnx.py              ONNX export, opset 17, dynamic batch axis
   │
api/app.py                          FastAPI: in-memory preprocessing (no disk writes),
                                    ONNX Runtime inference, bounded concurrency + load shedding
   │
scripts/load_test.py                concurrent client load test → latency percentiles
scripts/benchmark_latency.py        single-request device benchmarks (CPU/MPS/CUDA)

tests/test_smoke.py                 pytest suite (shapes, transform parity, ONNX round-trip)
.github/workflows/ci.yml            ruff + pytest + Docker build & container smoke test
docker/Dockerfile                   CPU-only, non-root, healthchecked serving image
```

Models live in `model/Meso4/model.py` and `model/SwinTransformer/model.py`. Both output raw logits;
sigmoid gives P(fake). Loss is `BCEWithLogitsLoss` (no double-sigmoid bugs).

### Experiment tracking (Weights & Biases)

Training is W&B-integrated but opt-in — the pipeline runs fully offline without an account:

```bash
pip install wandb
wandb login                      # or skip and use offline mode:
WANDB_MODE=offline python train.py --model meso4 --epochs 50 --wandb
```

Logged per run: config + hyperparameters, epoch-level loss/metric curves, best-val-AUC marker,
final test metrics with an interactive confusion matrix, and the trained checkpoint as a versioned
W&B artifact.

### Serving configuration

The API picks its model via environment variables (`docker/docker-compose.yml` sets them too):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/meso4/model.onnx` | ONNX file to serve (`models/swin/model.onnx` for Swin-T) |
| `INFERENCE_WORKERS` | `4` | inference threads / concurrent inferences |
| `MAX_QUEUE` | `64` | extra waiting slots before shedding |
| `QUEUE_TIMEOUT_S` | `2.0` | max wait before 503 |

### Tests & CI

`tests/test_smoke.py` covers model output shapes, parameter budget, exact train/serve transform
parity, ONNX round-trip, and dataset loader splits. `.github/workflows/ci.yml` runs lint (ruff) +
pytest on every push, then builds the Docker image and smoke-tests the container's `/docs`
endpoint — the image build is gated on tests passing.

### Docker

```bash
docker build -f docker/Dockerfile -t deepfake-api .
docker run --rm -p 8000:8000 deepfake-api                                   # Meso4 (~3ms)
docker run --rm -p 8000:8000 -e MODEL_PATH=models/swin/model.onnx deepfake-api   # Swin-T
# or
docker compose -f docker/docker-compose.yml up
```

CPU-only torch wheels, non-root user, healthcheck; ~2.5 GB with both models baked in.


---

## Reproducing

Requires Python 3.12+. On Apple Silicon, training uses MPS automatically.

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision scikit-learn pillow \
    pyarrow tqdm onnx onnxruntime onnxscript "fastapi[standard]" python-multipart jinja2 uvicorn

# 2. Data (~1 GB download → 36,667 images)
curl -L -o data/raw_parquet/train-0.parquet \
  https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces/resolve/main/data/train-00000-of-00006.parquet
curl -L -o data/raw_parquet/validation-0.parquet \
  https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces/resolve/main/data/validation-00000-of-00002.parquet
curl -L -o data/raw_parquet/test-0.parquet \
  https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces/resolve/main/data/test-00000-of-00002.parquet
python scripts/prepare_data.py

# 3. Train (Meso4 minutes; Swin-T ~7 min/epoch on M4 Pro-class GPU)
python train.py --model meso4 --epochs 50 --batch-size 64
python train.py --model swin  --epochs 8  --batch-size 32 --lr 1e-4

# 4. Evaluate + export + serve + load test
python evaluate.py --model swin
PYTHONPATH=. python scripts/export_onnx.py --model meso4
uvicorn api.app:app --port 8000
python scripts/load_test.py --url http://127.0.0.1:8000 --n 64

# Single-image CLI inference
python inference.py --image_path some_face.jpg --model_path models/meso4/best.pt
```

Artifacts per run land in `models/<name>/`: `best.pt`, `results.json`, `test_metrics.json`,
`model.onnx`, plus `models/latency_benchmark.json` and `models/load_test_results.json`.

---

## References

1. Afchar et al., *MesoNet: a Compact Facial Video Forgery Detection Network* (2018) —
   <https://arxiv.org/pdf/1809.00888>
2. Liu et al., *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows* (2021) —
   <https://arxiv.org/pdf/2103.14030>
3. Dataset: 70k FFHQ real faces vs StyleGAN fakes —
   <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>
