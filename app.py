from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from inference_onnx import DeepfakeONNXPredictor

# Initialize FastAPI app
app = FastAPI(title="Deepfake Detection API")

# Serve static files (CSS, JavaScript) and templates (HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:5500"] if serving from a specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model predictor
MODEL_PATH = "./models/model.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
predictor = DeepfakeONNXPredictor(MODEL_PATH)


def convert_to_standard_type(obj):
    """
    Recursively convert numpy types to Python standard types.
    This ensures JSON serialization compatibility.
    """
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
    """
    Accept an image file, save it temporarily, and run prediction using the ONNX model.
    
    Args:
        file (UploadFile): Uploaded image file to be classified.

    Returns:
        JSONResponse: Predicted label and confidence score.
    """
    # Ensure the temp directory exists for temporary file storage
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded file temporarily
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Run prediction
    try:
        result = predictor.predict(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Convert result to standard Python types
    result = convert_to_standard_type(result)
    
    # Get the prediction with the highest score
    try:
        highest_confidence_prediction = max(result, key=lambda x: x['score'])
    except ValueError:
        raise HTTPException(status_code=400, detail="No predictions were returned from the model.")
    
    # Return the prediction result
    return {
        "label": highest_confidence_prediction['label'],
        "confidence": highest_confidence_prediction['score'] * 100  # Convert to percentage
    }