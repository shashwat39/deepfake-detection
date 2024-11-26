import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Required constants.
DATA_DIR = 'C:\\Users\\Shashwat\\OneDrive\\Documents\\deepfake-detection\\data'  # Path to the data folder.
DEEPFAKE_DIR = 'C:\\Users\\Shashwat\\OneDrive\\Documents\\deepfake-detection\\data\\DeepFake'  # Path to DeepFake images.
REAL_DIR = 'C:\\Users\\Shashwat\\OneDrive\\Documents\\deepfake-detection\\data\\Real'  # Path to Real images.

IMAGE_SIZE = 224  # Image size of resize when applying transforms.
NUM_WORKERS = 4  # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform

def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    # Assuming the dataset is divided into two directories: DeepFake and Real
    dataset_train = datasets.ImageFolder(
        root=DATA_DIR,  # Root directory where 'DeepFake' and 'Real' folders are located.
        transform=get_train_transform(IMAGE_SIZE)
    )
    dataset_valid = datasets.ImageFolder(
        root=DATA_DIR,  # Root directory where 'DeepFake' and 'Real' folders are located.
        transform=get_valid_transform(IMAGE_SIZE)
    )
    return dataset_train, dataset_valid, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader