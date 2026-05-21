import torch
import torch.nn as nn
from data_loader import get_dataloaders
from classifier import build_resnet50, set_parameter_requires_grad, train_model
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 1. Cargar Datos
dataloaders, class_names = get_dataloaders(batch_size=32)

# 2. Construir Modelo
model = build_resnet50(num_classes=6, use_mlp_head=False).to(device)
criterion = nn.CrossEntropyLoss()

# 3. Fase 1: Feature Extraction
print("\n--- INICIANDO FEATURE EXTRACTION ---")
set_parameter_requires_grad(model, 'feature_extraction')
optimizer_fe = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
model, history_fe = train_model(model, dataloaders, criterion, optimizer_fe, device, num_epochs=20)

# 4. Fase 2: Fine-Tuning
print("\n--- INICIANDO FINE-TUNING ---")
set_parameter_requires_grad(model, 'fine_tuning')
optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
model, history_ft = train_model(model, dataloaders, criterion, optimizer_ft, device, num_epochs=30)

# Guardar los pesos finales
torch.save(model.state_dict(), 'resnet50_phaseA.pt')