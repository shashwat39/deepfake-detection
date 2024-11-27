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


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        
        val_batch = next(iter(self.val_dataloader))
        images, labels = val_batch

        
        outputs = pl_module(images)
        preds = (torch.sigmoid(outputs) > 0.5).int()

       
        df = pd.DataFrame({
            "Image": [f"Image_{i}" for i in range(len(labels))],
            "True Label": labels.cpu().numpy().flatten(),
            "Predicted Label": preds.cpu().numpy().flatten()
        })

       
        wrong_preds = df[df["True Label"] != df["Predicted Label"]]

        
        trainer.logger.experiment.log({
            "Incorrect Predictions": wandb.Table(dataframe=wrong_preds)
        })


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: Meso4")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )

    wandb_logger = WandbLogger(project="deepfake-detection")

    
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
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()