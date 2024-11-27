import torch
import hydra
import logging
from omegaconf import OmegaConf
from model import Meso4  # Assuming Meso4 is your model class from model.py
from data import get_data_loaders  # Assuming get_data_loaders is the equivalent function

import warnings
warnings.filterwarnings("ignore")

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    # Define the paths
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/epoch=150-step=26878.ckpt"
    onnx_output_path = f"{root_dir}/models/model_3.onnx"
    
    logger.info(f"Loading pre-trained model from: {model_path}")
    
    # Load the model from checkpoint
    cola_model = Meso4.load_from_checkpoint(model_path)  # Assuming Meso4 is defined in model.py

    # Get data loader for input batch sample
    train_loader, _ = get_data_loaders(cfg.data.dir, batch_size=cfg.processing.batch_size)

    # Prepare a sample input batch from the train loader
    input_batch = next(iter(train_loader))
    input_sample = input_batch[0][0].unsqueeze(0)  # Extract the first image and add batch dimension

    # Convert the model to ONNX format
    logger.info(f"Converting the model into ONNX format")
    cola_model.to_onnx(
        onnx_output_path,                    # ONNX model output path
        input_sample,                        # Example input
        export_params=True,                  # Export parameters
        opset_version=10,                    # ONNX version
        input_names=["input"],               # Model input names
        output_names=["output"],             # Model output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    logger.info(f"Model converted successfully. ONNX model is at: {onnx_output_path}")


if __name__ == "__main__":
    convert_model()
