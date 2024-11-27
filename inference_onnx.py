import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

#from utils import timing


class DeepfakeONNXPredictor:
    def __init__(self, model_path):
        # Initialize the ONNX runtime session
        self.ort_session = ort.InferenceSession(model_path)
        self.labels = ["Deepfake", "Real"]

        # Define the image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return np.expand_dims(image.numpy(), axis=0)  # Add batch dimension

    #@timing
    def predict(self, image_path):
        # Preprocess the image and prepare ONNX inputs
        processed_image = self.preprocess_image(image_path)
        ort_inputs = {"input": processed_image}

        # Run inference
        ort_outs = self.ort_session.run(None, ort_inputs)
        score = ort_outs[0][0][0]  # Access the output score directly

        # Interpret the result
        predictions = []
        for label_idx, label in enumerate(self.labels):
            predictions.append({"label": label, "score": 1 - score if label_idx == 0 else score})

        return predictions


if __name__ == "__main__":
    # Example usage
    image_path = r'C:\Users\Shashwat\Downloads\10c57b68-3bf8-4073-a4f2-748420c97134.jpg'
    predictor = DeepfakeONNXPredictor("./models/model_3.onnx")
    print(predictor.predict(image_path))

    # Batch inference example
    from more_dummy import top_images
    # Initialize the counter
    real_higher_count = 0

    # Iterate through all the image paths in top_images
    for img_path in top_images:
        # Get prediction result for the image
        prediction = predictor.predict(img_path)
        
        # Extract the 'Real' and 'Deepfake' scores
        deepfake_score = next(item['score'] for item in prediction if item['label'] == 'Deepfake')
        real_score = next(item['score'] for item in prediction if item['label'] == 'Real')
        
        # Check if the 'Real' score is greater than the 'Deepfake' score
        if real_score > deepfake_score:
            real_higher_count += 1

    # Output the result
    print(f'Number of images where "Real" score is higher than "Deepfake": {real_higher_count}')