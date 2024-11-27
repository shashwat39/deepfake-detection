import torch
import hydra
import logging
from omegaconf import OmegaConf
from model import Meso4 
from data import get_data_loaders 

import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/epoch=150-step=26878.ckpt"
    onnx_output_path = f"{root_dir}/models/model_3.onnx"
    
    logger.info(f"Loading pre-trained model from: {model_path}")
    
    
    cola_model = Meso4.load_from_checkpoint(model_path)  
    
    train_loader, _ = get_data_loaders(cfg.data.dir, batch_size=cfg.processing.batch_size)

    
    input_batch = next(iter(train_loader))
    input_sample = input_batch[0][0].unsqueeze(0)  

    
    logger.info(f"Converting the model into ONNX format")
    cola_model.to_onnx(
        onnx_output_path,                   
        input_sample,                        
        export_params=True,                  
        opset_version=10,                    
        input_names=["input"],               
        output_names=["output"],             
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    logger.info(f"Model converted successfully. ONNX model is at: {onnx_output_path}")


if __name__ == "__main__":
    convert_model()
