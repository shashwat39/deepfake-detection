"""Export a trained checkpoint to ONNX with a dynamic batch axis."""
import argparse

import torch

from model.Meso4.model import build_model as build_meso4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["meso4", "swin"], default="meso4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"models/{args.model}/best.pt"
    out_path = args.out or f"models/{args.model}/model.onnx"

    model = build_meso4() if args.model == "meso4" else __import__(
        "model.SwinTransformer.model", fromlist=["build_model"]
    ).build_model(num_classes=1)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    import onnx
    proto = onnx.load(out_path, load_external_data=True)
    onnx.save(proto, out_path)
    print(f"exported {out_path} (opset {args.opset}, dynamic batch, self-contained)")


if __name__ == "__main__":
    main()
