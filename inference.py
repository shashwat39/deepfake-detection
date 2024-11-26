import torch
from torchvision import transforms
from PIL import Image
from model import Meso4
import time

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def inference(image_path, model_path):
    model = Meso4.load_from_checkpoint(model_path)
    model.eval()
    
    image = load_image(image_path)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    
    if prediction < 0.5:
        return "DeepFake"
    else:
        return "Real"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFake Detection Inference")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image for inference")
    parser.add_argument('--model_path', type=str, help="Path to the trained model checkpoint")
    
    args = parser.parse_args()

    args.model_path = "C:\\Users\\Shashwat\\OneDrive\\Documents\\deepfake-detection\\models\\epoch=4-step=890.ckpt"

    t_start = time.time()
    result = inference(args.image_path, args.model_path)
    t_end = time.time()
    print(t_end - t_start)
    print(f"Inference result: {result}")
