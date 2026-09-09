import argparse
import json

import torch

from data.data_loader import get_data_loaders
from model.Meso4.model import build_model as build_meso4
from model.SwinTransformer.model import build_model as build_swin
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split")
    parser.add_argument("--model", choices=["meso4", "swin"], default="meso4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"models/{args.model}/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    _, _, test_loader, meta = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=256,
    )
    pos_class_idx = meta["class_to_idx"]["fake"]

    model = build_meso4() if args.model == "meso4" else build_swin(num_classes=1)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, test_loader, device, pos_class_idx)
    cm = metrics.pop("confusion_matrix")
    print(json.dumps(metrics, indent=2))
    print(f"confusion matrix [rows=truth fake/real, cols=pred fake/real]: {cm}")

    out = f"models/{args.model}/test_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics | {"confusion_matrix": cm}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
