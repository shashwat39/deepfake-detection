import argparse
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from data.data_loader import get_data_loaders
from model.Meso4.model import build_model as build_meso4
from model.SwinTransformer.model import build_model as build_swin


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, pos_class_idx):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu()
        pos_idx = (labels == pos_class_idx).long()
        all_probs.append(probs)
        all_labels.append(pos_idx)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n_samples": int(len(labels)),
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, seen = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = (labels == 0).float().unsqueeze(1).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, images)
        loss = criterion(logits, targets.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return running_loss / seen


def forward_logits(model, images):
    out = model(images)
    return out.squeeze(1)


def main():
    parser = argparse.ArgumentParser(description="Train a deepfake detector")
    parser.add_argument("--model", choices=["meso4", "swin"], default="meso4")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--wandb", action="store_true",
                        help="log experiment to Weights & Biases (requires wandb login, or WANDB_MODE=offline)")
    parser.add_argument("--wandb-project", default="deepfake-detection")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"device: {device} | model: {args.model}")

    wb_run = None
    if args.wandb:
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            name=f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={**vars(args), "device": str(device)},
        )

    train_loader, val_loader, test_loader, meta = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
    )
    print(f"splits: {meta['sizes']} | class_to_idx: {meta['class_to_idx']}")
    pos_class_idx = meta["class_to_idx"]["fake"]

    if args.model == "meso4":
        model = build_meso4()
        lr = args.lr or 1e-3
    else:
        model = build_swin(num_classes=1, fine_tune=True)
        lr = args.lr or 1e-4

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")
    if wb_run is not None:
        wb_run.config.update({"trainable_params": n_params, "splits": meta["sizes"]})

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(run_dir, exist_ok=True)
    history = []
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, pos_class_idx)
        dt = time.time() - t0
        print(f"epoch {epoch}/{args.epochs} ({dt:.0f}s) | train_loss={train_loss:.4f} "
              f"| val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        if wb_run is not None:
            wb_run.log({
                "train/loss": train_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "val/roc_auc": val_metrics["roc_auc"],
            }, step=epoch)

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            torch.save({"model_state_dict": model.state_dict(),
                        "class_to_idx": meta["class_to_idx"],
                        "args": vars(args)}, os.path.join(run_dir, "best.pt"))
            if wb_run is not None:
                wb_run.log({"best_val_auc": best_auc}, step=epoch)
            print(f"  saved new best (val_auc={best_auc:.4f})")

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, pos_class_idx)
    print(f"TEST | acc={test_metrics['accuracy']:.4f} auc={test_metrics['roc_auc']:.4f} f1={test_metrics['f1']:.4f}")

    results = {"model": args.model, "best_val_auc": best_auc, "test": test_metrics,
               "history": history, "splits": meta["sizes"]}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(run_dir, 'results.json')}")

    if wb_run is not None:
        import wandb
        cm = test_metrics["confusion_matrix"]
        wb_run.log({
            "test/accuracy": test_metrics["accuracy"],
            "test/roc_auc": test_metrics["roc_auc"],
            "test/f1": test_metrics["f1"],
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=[0] * (cm[0][0] + cm[0][1]) + [1] * (cm[1][0] + cm[1][1]),
                preds=[0] * cm[0][0] + [1] * cm[0][1] + [0] * cm[1][0] + [1] * cm[1][1],
                class_names=["fake", "real"]),
        })
        test_table = wandb.Table(
            columns=["metric", "value"],
            data=[[k, v] for k, v in test_metrics.items() if k != "confusion_matrix"])
        wb_run.log({"test/metrics": test_table})
        artifact = wandb.Artifact(f"{args.model}-checkpoint", type="model")
        artifact.add_file(os.path.join(run_dir, "best.pt"))
        wb_run.log_artifact(artifact)
        wb_run.finish()


if __name__ == "__main__":
    main()
