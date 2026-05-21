import torch
import torch.nn as nn
import copy
import os

# Importaciones de los módulos que ya creamos
from data_loader import get_dataloaders
from classifier import build_resnet50, train_model
from domain_adaptation import evaluate_domain_shift, prepare_adaptation_dataloaders

def run_comprehensive_phase_c():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALIZING PHASE C: DOMAIN ADAPTATION ===")
    print(f"Device detected: {device}\n")

    # 1. Cargar Dataloaders base
    dataloaders, class_names = get_dataloaders(batch_size=32)
    criterion = nn.CrossEntropyLoss()

    # 2. Cargar Modelo Base de la Fase A
    base_model = build_resnet50(num_classes=6, use_mlp_head=False)
    if not os.path.exists('resnet50_phaseA.pt'):
        print("[-] ERROR: Can't find 'resnet50_phaseA.pt'. Execute Phase A first.")
        return
    
    base_model.load_state_dict(torch.load('resnet50_phaseA.pt', map_location=device))
    base_model = base_model.to(device)

    # ---------------------------------------------------------
    # ESTRATEGIA 1: BASELINE (Sin Adaptación)
    # ---------------------------------------------------------
    print("\n>>> [STRATEGY 1] Evaluating Baseline (Pure model from Phase A)")
    acc_src_bl, acc_tgt_bl, delta_bl = evaluate_domain_shift(base_model, dataloaders, device)
    
    # Guardamos los resultados para el cuadro comparativo del paper
    resultados_paper = {
        "Baseline": {"source_acc": acc_src_bl, "target_acc": acc_tgt_bl, "delta": delta_bl}
    }

    # ---------------------------------------------------------
    # ESTRATEGIA 2: TARGET FINE-TUNING
    # ---------------------------------------------------------
    print("\n>>> [STRATEGY 2] Initializing Target Fine-Tuning (Adjust in Sketches)")
    model_tft = copy.deepcopy(base_model).to(device)
    
    # El PDF exige congelar capas tempranas y entrenar solo las profundas (ej. layer4 y fc)
    for param in model_tft.parameters():
        param.requires_grad = False
    for param in model_tft.layer4.parameters():
        param.requires_grad = True
    for param in model_tft.fc.parameters():
        param.requires_grad = True

    # El dataloader saca estrictamente las 50 imágenes de sketch/train
    dl_target_train = prepare_adaptation_dataloaders('target_ft', base_data_dir='./challenge_data', synth_data_dir=None)
    tft_dataloaders = {'source_train': dl_target_train, 'source_val': dataloaders['source_val']}
    
    optimizer_tft = torch.optim.Adam(filter(lambda p: p.requires_grad, model_tft.parameters()), lr=1e-4)
    # Entrenamos por pocas épocas (10-15 es suficiente para pocos datos y evitar overfitting masivo)
    model_tft, _ = train_model(model_tft, tft_dataloaders, criterion, optimizer_tft, device, num_epochs=15)
    
    print("\nEvaluating strategy 2 (Target Fine-Tuning):")
    acc_src_tft, acc_tgt_tft, delta_tft = evaluate_domain_shift(model_tft, dataloaders, device)
    resultados_paper["Target FT"] = {"source_acc": acc_src_tft, "target_acc": acc_tgt_tft, "delta": delta_tft}
    torch.save(model_tft.state_dict(), 'resnet50_target_ft.pt')

    # ---------------------------------------------------------
    # ESTRATEGIA 3: STYLE-TRANSFER AUGMENTATION
    # ---------------------------------------------------------
    print("\n>>> [ESTRATEGY 3] Initializing Style-Transfer Augmentation")
    synth_dir = './data/synthetic_target/'
    
    if not os.path.exists(synth_dir) or len(os.listdir(synth_dir)) == 0:
        print("[-] WARNING: Synthetic images not found. Execute generate_synthetic.py first.")
    else:
        model_sta = copy.deepcopy(base_model).to(device)
        
        # Descongelamos igual las capas pesadas finales
        for param in model_sta.parameters():
            param.requires_grad = False
        for param in model_sta.layer4.parameters():
            param.requires_grad = True
        for param in model_sta.fc.parameters():
            param.requires_grad = True

        # DataLoader que concatena las Fotos Reales + Fotos Estilizadas Sintéticas (80 por clase)
        dl_synth_aug = prepare_adaptation_dataloaders('synth_aug', base_data_dir='./challenge_data', synth_data_dir=synth_dir)
        sta_dataloaders = {'source_train': dl_synth_aug, 'source_val': dataloaders['source_val']}
        
        optimizer_sta = torch.optim.Adam(filter(lambda p: p.requires_grad, model_sta.parameters()), lr=1e-4)
        model_sta, _ = train_model(model_sta, sta_dataloaders, criterion, optimizer_sta, device, num_epochs=15)
        
        print("\nEvaluating Strategy 3 (Style-Transfer Augmentation):")
        acc_src_sta, acc_tgt_sta, delta_sta = evaluate_domain_shift(model_sta, dataloaders, device)
        resultados_paper["Synth Aug"] = {"source_acc": acc_src_sta, "target_acc": acc_tgt_sta, "delta": delta_sta}
        torch.save(model_sta.state_dict(), 'resnet50_synth_aug.pt')

    # 3. Imprimir el resumen ejecutivo final para la sección de resultados del paper
    print("\n========================================================")
    print("METRICS SUMMARY TABLE")
    print("========================================================")
    print(f"{'Strategy':<15} | {'Acc Source (Real)':<18} | {'Acc Objective (Sketch)':<20} | {'Delta Shift':<12}")
    print("-" * 75)
    for est, m in resultados_paper.items():
        print(f"{est:<15} | {m['source_acc']:<18.4f} | {m['target_acc']:<20.4f} | {m['delta']:<12.4f}")
    print("========================================================")

if __name__ == '__main__':
    run_comprehensive_phase_c()