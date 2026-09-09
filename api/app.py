import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms as tv_transforms

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/meso4/model.onnx")
INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "4"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "64"))
QUEUE_TIMEOUT_S = float(os.environ.get("QUEUE_TIMEOUT_S", "2.0"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

_eval_transform = tv_transforms.Compose([
    tv_transforms.Resize((256, 256)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(MEAN, STD),
])

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="frontend")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = INFERENCE_WORKERS
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(MODEL_PATH, sess_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

executor = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS, thread_name_prefix="infer")
inference_slots = asyncio.BoundedSemaphore(INFERENCE_WORKERS + MAX_QUEUE)


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _eval_transform(image)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def predict(image_bytes: bytes) -> dict:
    tensor = preprocess(image_bytes)
    logit = session.run(["logits"], {input_name: tensor})[0][0][0]
    prob_fake = float(1.0 / (1.0 + np.exp(-logit)))
    label = "Fake" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake if prob_fake >= 0.5 else 1.0 - prob_fake
    return {"label": label, "confidence": round(confidence * 100, 2)}


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(QUEUE_TIMEOUT_S):
            async with inference_slots:
                result = await loop.run_in_executor(executor, predict, image_bytes)
    except TimeoutError:
        logger.warning("load shedding: queue full after %.1fs", QUEUE_TIMEOUT_S)
        raise HTTPException(
            status_code=503,
            detail="Server at capacity, retry shortly",
            headers={"Retry-After": "1"},
        )
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return result
