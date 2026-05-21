import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
# Import the data transforms from our data_loader module
from data_loader import get_dataloaders 

def evaluate_domain_shift(model, dataloaders, device):
    """
    Calculates the domain shift penalty: Delta_shift = Acc_source - Acc_target.
    Evaluates the model without any adaptation (Baseline strategy).
    """
    model.eval()
    
    def get_accuracy(phase):
        running_corrects = 0
        total_samples = len(dataloaders[phase].dataset)
        
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        return (running_corrects.double() / total_samples).item()

    acc_source = get_accuracy('source_test')
    acc_target = get_accuracy('target_test')
    
    # The mathematical formulation required by the challenge
    delta_shift = acc_source - acc_target
    
    print(f"Source Test Accuracy: {acc_source:.4f}")
    print(f"Target Test Accuracy: {acc_target:.4f}")
    print(f"Domain Shift Penalty (Delta): {delta_shift:.4f}")
    
    return acc_source, acc_target, delta_shift

def prepare_adaptation_dataloaders(strategy, base_data_dir, synth_data_dir, batch_size=32):
    """
    Prepares DataLoaders for Phase C adaptation strategies independently.
    Guarantees no KeyError from data_loader.py.
    """
    # 1. Definir las mismas transformaciones estándar que requiere el backbone ResNet-50
    # Con Data Augmentation para evitar sobreajuste en el set pequeño de bocetos
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if strategy == 'target_ft':
        # Ruta estricta hacia los bocetos de entrenamiento (50 por clase)
        target_train_dir = os.path.join(base_data_dir, 'sketch', 'train')
        if not os.path.exists(target_train_dir):
            raise FileNotFoundError(f"No se encontró la carpeta objetivo: {target_train_dir}")
            
        target_train_ds = datasets.ImageFolder(target_train_dir, transform=train_transforms)
        return DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        
    elif strategy == 'synth_aug':
        # Cargar las fotos reales de entrenamiento (50 por clase)
        source_train_dir = os.path.join(base_data_dir, 'real', 'train')
        if not os.path.exists(source_train_dir):
            raise FileNotFoundError(f"No se encontró la carpeta origen: {source_train_dir}")
        source_train_ds = datasets.ImageFolder(source_train_dir, transform=train_transforms)
        
        # Cargar las imágenes estilizadas generadas en la Fase B (30 por clase)
        if not os.path.exists(synth_data_dir):
            raise FileNotFoundError(f"No se encontró la carpeta de imágenes sintéticas: {synth_data_dir}")
        synth_ds = datasets.ImageFolder(synth_data_dir, transform=train_transforms)
        
        # Concatenación científica: Real (50/clase) + Sintético (30/clase) = 80 imágenes por clase
        augmented_ds = ConcatDataset([source_train_ds, synth_ds])
        return DataLoader(augmented_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    
    else:
        raise ValueError(f"Estrategia '{strategy}' no reconocida.")

# Example usage for Target Fine-Tuning:
# 1. Load best Phase A model
# 2. Extract adaptation DataLoader: dl_adapt = prepare_adaptation_dataloaders('target_ft', ...)
# 3. Unfreeze layer4 and head. Set LR = 1e-4.
# 4. Train for 10-20 epochs using train_model() logic.
# 5. Evaluate with evaluate_domain_shift()