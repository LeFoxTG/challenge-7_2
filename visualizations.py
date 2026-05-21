import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from captum.attr import LayerGradCam
from data_loader import get_dataloaders
from classifier import build_resnet50

def generate_tsne(model, dataloader, device, save_name='tsne_plot.png'):
    """Extracts features before the classifier layer and plots a 2D t-SNE projection."""
    model.eval()
    features = []
    labels_list = []
    domain_list = []

    print("[+] Extracting features for t-SNE...")
    
    feature_activation = {}
    def hook_fn(module, input, output):
        feature_activation['pool'] = output.detach()
    
    hook = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            feats = feature_activation['pool'].view(inputs.size(0), -1).cpu().numpy()
            features.append(feats)
            labels_list.append(labels.numpy())

    hook.remove()
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    print("[+] Calculating t-SNE proyection...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Clases (0-5)')
    plt.title('t-SNE Backbone features proyection')
    plt.savefig(save_name)
    plt.close()
    print(f"[+] t-SNE graphic saved as '{save_name}'")

def generate_gradcam(model, image_tensor, label, device, save_name='gradcam_output.png'):
    """Generates a Grad-CAM activation heatmap over the chosen image layer."""
    model.eval()
    
    lgc = LayerGradCam(model, model.layer4)
    
    input_tensor = image_tensor.unsqueeze(0).to(device)
    attribution = lgc.attribute(input_tensor, target=label)
    
    attribution = attribution.squeeze(0).cpu().detach().numpy()
    attribution = np.maximum(attribution, 0) # ReLU sobre el mapa
    heatmap = np.mean(attribution, axis=0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1.0

    img_show = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(img_show)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f'Grad-CAM para Clase {label}')
    plt.savefig(save_name)
    plt.close()
    print(f"[+] Mapa Grad-CAM guardado como '{save_name}'")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, _ = get_dataloaders(batch_size=32)
    
    model = build_resnet50(num_classes=6).to(device)
    if os.path.exists('resnet50_phaseA.pt'):
        model.load_state_dict(torch.load('resnet50_phaseA.pt', map_location=device))
        
        generate_tsne(model, dataloaders['source_val'], device, 'tsne_faseA.png')
        
        images, labels = next(iter(dataloaders['source_val']))
        generate_gradcam(model, images[0], labels[0].item(), device, 'gradcam_sample.png')