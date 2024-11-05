from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference_onnx import DeepfakeONNXPredictor
from typing import List
import os
import numpy as np

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:5500"] if serving from a specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_standard_type(obj):
    # Recursively convert numpy types to Python standard types
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, dict):
        return {k: convert_to_standard_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_type(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_standard_type(item) for item in obj)
    return obj  # Return unchanged if it's already a standard Python type

# Instantiate the predictor
predictor = DeepfakeONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>Welcome to the Deepfake Detection API</h2><p>Upload an image to determine if it is a deepfake or real.</p>"

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    
    os.makedirs('./temp', exist_ok=True)
    
    # Save the uploaded file temporarily
    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Get prediction
    result = predictor.predict(file_path)
    
    result = convert_to_standard_type(result)
        
    return {"predictions": result}