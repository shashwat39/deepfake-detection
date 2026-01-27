import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from model.Meso4.model import Meso4
from data.data_loader import get_data_loaders
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
import warnings
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            images, labels = next(iter(self.val_dataloader))

            images = images.to(pl_module.device)
            labels = labels.to(pl_module.device)

            outputs = pl_module(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            df = pd.DataFrame({
                "Image": [f"Image_{i}" for i in range(len(labels))],
                "True Label": labels.cpu().numpy().flatten(),
                "Predicted Label": preds.cpu().numpy().flatten()
            })

            wrong_preds = df[df["True Label"] != df["Predicted Label"]]

            if len(wrong_preds) > 0:
                trainer.logger.experiment.log({
                    "Incorrect Predictions": wandb.Table(dataframe=wrong_preds)
                })


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    os.makedirs("models", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="valid/loss",
        mode="min",
        filename="meso4-{epoch:02d}-{valid_loss:.4f}"
    )

    wandb_logger = WandbLogger(
        project="deepfake-detection",
        log_model=True
    )

    train_loader, val_loader = get_data_loaders()

    model = Meso4()

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(val_loader)],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        accelerator="auto",
        devices="auto"
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
