import torch.nn as nn
from torchvision import models


def build_model(num_classes=1, fine_tune=True, weights="DEFAULT"):
    """Swin-T with a binary head. Outputs raw logits of shape (B, 1)."""
    model = models.swin_t(weights=weights)

    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = False

    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    return model
