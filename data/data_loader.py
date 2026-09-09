import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(data_dir="dataset", batch_size=64, num_workers=4, image_size=256):
    """Load the official train/val/test splits written by scripts/prepare_data.py.

    ImageFolder sorts classes alphabetically: fake -> 0, real -> 1.
    All downstream code treats 'fake' as the positive class via class_to_idx.
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transform)

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=num_workers > 0)

    meta = {
        "class_to_idx": train_ds.class_to_idx,
        "sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    return train_loader, val_loader, test_loader, meta
