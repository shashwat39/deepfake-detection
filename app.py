from fastapi import FastAPI, File, UploadFile
from inference_onnx import DeepfakeONNXPredictor
from typing import List

app = FastAPI(title="Deepfake Detection API")

# Instantiate the predictor
predictor = DeepfakeONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>Welcome to the Deepfake Detection API</h2><p>Upload an image to determine if it is a deepfake or real.</p>"

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Get prediction
    result = predictor.predict(file_path)
    
    return {"predictions": result}
