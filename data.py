import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir='C:\\Users\\Shashwat\\OneDrive\\Documents\\deepfake-detection\\data', batch_size=32):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    print(f"Loading data from: {data_dir}")
    print(os.path.isdir(data_dir))

    # Load the dataset using ImageFolder, assuming two subdirectories 'DeepFake' and 'Real'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Splitting dataset into training and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)

    return train_loader, val_loader