from torchvision import models
import torch.nn as nn
def build_model(fine_tune=True, num_classes=10):
    model = models.swin_t(weights='DEFAULT')
    print(model)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    
    model.head = nn.Linear(
        in_features=768, 
        out_features=num_classes, 
        bias=True
    )
    return model