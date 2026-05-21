import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights
from PIL import Image

def gram_matrix(feat):
    """
    Calculates the Gram Matrix to capture style (feature correlations).
    """
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)
    # Batch matrix multiplication: feature maps multiplied by their transpose
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)

class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a pre-trained VGG19 network.
    """
    def __init__(self):
        super().__init__()
        # Load VGG-19 with ImageNet weights
        vgg_pretrained = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # Challenge requirement: specific layers for content and style [cite: 192, 193]
        self.content_layers = {'21': 'relu4_2'} 
        self.style_layers = {'1': 'relu1_1', '6': 'relu2_1', '11': 'relu3_1', '20': 'relu4_1', '29': 'relu5_1'}
        
        # Slice the network up to the deepest layer needed (relu5_1 is at index 29)
        self.net = nn.Sequential(*list(vgg_pretrained.children())[:30])
        
        # Freeze all VGG parameters (we optimize the image, not the network)
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_feats = {}
        style_feats = {}
        
        for name, layer in self.net.named_children():
            x = layer(x)
            if name in self.content_layers:
                content_feats[self.content_layers[name]] = x
            if name in self.style_layers:
                style_feats[self.style_layers[name]] = x
                
        return content_feats, style_feats

def run_style_transfer(content_img_tensor, style_img_tensor, device, num_steps=300, alpha=1.0, beta=1e4):
    """
    Optimizes a target image to blend content and style using L-BFGS[cite: 180, 188].
    """
    extractor = VGGFeatureExtractor().to(device)
    extractor.eval()
    
    # Extract target features once
    target_content_feats, _ = extractor(content_img_tensor)
    _, target_style_feats = extractor(style_img_tensor)
    
    # Initialize the image to be optimized (start with content image clone)
    # requires_grad_(True) is critical: we backpropagate to the pixels! [cite: 188]
    generated_img = content_img_tensor.clone().requires_grad_(True)
    
    # Challenge recommended optimizer: L-BFGS [cite: 180, 188]
    optimizer = torch.optim.LBFGS([generated_img], lr=1.0, max_iter=20)
    
    step_idx = [0]
    
    while step_idx[0] <= num_steps:
        def closure():
            # Ensure pixel values stay valid after each optimization step
            with torch.no_grad():
                generated_img.clamp_(0, 1)
                
            optimizer.zero_grad()
            
            gen_content_feats, gen_style_feats = extractor(generated_img)
            
            # Compute Content Loss (MSE) [cite: 184]
            c_loss = 0
            for layer in target_content_feats:
                c_loss += nn.functional.mse_loss(gen_content_feats[layer], target_content_feats[layer])
                
            # Compute Style Loss (MSE of Gram Matrices) [cite: 186, 187]
            s_loss = 0
            for layer in target_style_feats:
                gen_gram = gram_matrix(gen_style_feats[layer])
                target_gram = gram_matrix(target_style_feats[layer])
                s_loss += nn.functional.mse_loss(gen_gram, target_gram)
                
            # Total Loss [cite: 189]
            loss = alpha * c_loss + beta * s_loss
            loss.backward()
            
            step_idx[0] += 1
            if step_idx[0] % 50 == 0:
                print(f'Step {step_idx[0]}: Total Loss: {loss.item():.4f}')
                
            return loss
            
        optimizer.step(closure)
        
    # Final clamp to ensure valid image
    with torch.no_grad():
        generated_img.clamp_(0, 1)
        
    return generated_img