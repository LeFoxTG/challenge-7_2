import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import copy
import time

def build_resnet50(num_classes=6, use_mlp_head=False):
    """
    Constructs a ResNet-50 model with pretrained ImageNet weights.
    Replaces the final fully connected layer with a custom classification head.
    """
    # Load state-of-the-art pretrained weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    
    # Challenge requirement: Compare single linear layer vs two-layer MLP
    if use_mlp_head:
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
        
    return model

def set_parameter_requires_grad(model, strategy='feature_extraction'):
    """
    Configures gradient computation for different Transfer Learning strategies.
    strategy can be 'feature_extraction' or 'fine_tuning'.
    """
    if strategy == 'feature_extraction':
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
    elif strategy == 'fine_tuning':
        # Freeze all initially
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze top convolutional blocks (layer3 and layer4) for fine-tuning
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True

    # Always ensure the classification head remains trainable
    for param in model.fc.parameters():
        param.requires_grad = True

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    """
    Standard PyTorch training loop with validation tracking.
    Returns the model with the best validation accuracy.
    """
    since = time.time()
    
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['source_train', 'source_val']:
            if phase == 'source_train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == 'source_train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward pass + optimize only if in training phase
                    if phase == 'source_train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improves
            if phase == 'source_val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            # Track metrics for plotting (Challenge requirement)
            if phase == 'source_train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            elif phase == 'source_val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    history = {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history
    }
    
    return model, history