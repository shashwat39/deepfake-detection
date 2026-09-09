import argparse
import time

import torch
from PIL import Image
from torchvision import transforms

from model.Meso4.model import Meso4


def load_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def inference(image_path, model_path, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Meso4().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image = load_image(image_path).to(device)
    with torch.no_grad():
        logit = model(image).squeeze(1)
        prob_fake = torch.sigmoid(logit).item()

    label = "Fake" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake if prob_fake >= 0.5 else 1 - prob_fake
    return {"label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/meso4/best.pt")
    args = parser.parse_args()

    t_start = time.time()
    result = inference(args.image_path, args.model_path)
    print(f"Inference result: {result} ({time.time() - t_start:.2f}s)")
