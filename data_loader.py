import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='./challenge_data', batch_size=32):
    """
    Creates and returns PyTorch DataLoaders for the source and target domains.
    Strictly follows the ImageNet normalization and augmentation requirements.
    """
    
    # Standard ImageNet statistics required by the challenge guidelines
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # 1. Training transforms with Data Augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # 2. Evaluation transforms (Strictly deterministic, no random augmentations)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Define paths for the Source domain (Real)
    source_train_dir = os.path.join(data_dir, 'real', 'train')
    source_val_dir = os.path.join(data_dir, 'real', 'val')
    source_test_dir = os.path.join(data_dir, 'real', 'test')
    
    # Define paths for the Target domain (Sketch)
    target_test_dir = os.path.join(data_dir, 'sketch', 'test')
    
    # 3. Create PyTorch Datasets using ImageFolder
    source_train_ds = datasets.ImageFolder(source_train_dir, transform=train_transforms)
    source_val_ds = datasets.ImageFolder(source_val_dir, transform=eval_transforms)
    source_test_ds = datasets.ImageFolder(source_test_dir, transform=eval_transforms)
    target_test_ds = datasets.ImageFolder(target_test_dir, transform=eval_transforms)
    
    # 4. Wrap Datasets in DataLoaders
    # pin_memory=True speeds up host-to-device memory transfer if using a GPU
    dataloaders = {
        'source_train': DataLoader(source_train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        'source_val': DataLoader(source_val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'source_test': DataLoader(source_test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        'target_test': DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    }
    
    # Return mapping of class indices to class names for later visualization
    class_names = source_train_ds.classes
    
    return dataloaders, class_names

# Quick test execution block
if __name__ == '__main__':
    dl, classes = get_dataloaders()
    print(f"Classes found: {classes}")
    
    # Fetch a single batch to verify tensor dimensions
    images, labels = next(iter(dl['source_train']))
    # Expected shape: [Batch_Size, Channels, Height, Width] -> [32, 3, 224, 224]
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")