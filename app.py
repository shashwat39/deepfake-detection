from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from inference_onnx import DeepfakeONNXPredictor


app = FastAPI(title="Deepfake Detection API")


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "./models/model_3.onnx"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
predictor = DeepfakeONNXPredictor(MODEL_PATH)


def convert_to_standard_type(obj):
    
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_standard_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_type(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_standard_type(item) for item in obj)
    return obj


@app.get("/", response_class=HTMLResponse)
async def home_page():
    """
    Serve the homepage with the Deepfake Detection interface.
    """
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    
    try:
        result = predictor.predict(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        
        if os.path.exists(file_path):
            os.remove(file_path)
    
    
    result = convert_to_standard_type(result)
    
    
    try:
        highest_confidence_prediction = max(result, key=lambda x: x['score'])
    except ValueError:
        raise HTTPException(status_code=400, detail="No predictions were returned from the model.")
    
   
    return {
        "label": highest_confidence_prediction['label'],
        "confidence": highest_confidence_prediction['score'] * 100  
    }
