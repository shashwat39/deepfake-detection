import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from model import Meso4
from data import get_data_loaders
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Custom Callback for logging incorrect predictions
class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get a validation batch (you can also process the full dataset here)
        val_batch = next(iter(self.val_dataloader))
        images, labels = val_batch

        # Forward pass
        outputs = pl_module(images)
        preds = (torch.sigmoid(outputs) > 0.5).int()

        # Prepare a DataFrame for analysis
        df = pd.DataFrame({
            "Image": [f"Image_{i}" for i in range(len(labels))],
            "True Label": labels.cpu().numpy().flatten(),
            "Predicted Label": preds.cpu().numpy().flatten()
        })

        # Filter incorrect predictions
        wrong_preds = df[df["True Label"] != df["Predicted Label"]]

        # Log to WandB
        trainer.logger.experiment.log({
            "Incorrect Predictions": wandb.Table(dataframe=wrong_preds)
        })


# NOTE: Need to provide the path for configs folder and the config file name
@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: Meso4")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )

    wandb_logger = WandbLogger(project="deepfake-detection")

    # DataLoaders
    train_loader, val_loader = get_data_loaders()

    # Model
    model = Meso4()

    # Instantiate the trainer with the custom callback
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(val_loader)],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()