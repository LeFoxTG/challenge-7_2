import os
import random
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
# Assuming the previous code is saved as style_transfer.py
from style_transfer import run_style_transfer 

def generate_synthetic_dataset(data_dir='./challenge_data', output_dir='./data/synthetic_target/', images_per_class=30, device='cuda'):
    """
    Automates the generation of 180 synthetic images using Neural Style Transfer.
    Strictly follows the requirement to blend Real content with Sketch style.
    """
    classes = ['airplane', 'bicycle', 'bus', 'car', 'motorcycle', 'train']
    
    # Standard transforms for VGG-19 input (NST does not use Data Augmentation)
    nst_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # VGG expects standard ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Denormalize to save valid images (Inverse of ImageNet normalization)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # Paths for source (Real) and target (Sketch)
        real_dir = os.path.join(data_dir, 'real', 'train', cls)
        sketch_dir = os.path.join(data_dir, 'sketch', 'train', cls)
        
        real_imgs = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        sketch_imgs = [os.path.join(sketch_dir, img) for img in os.listdir(sketch_dir)]
        
        print(f"Generating synthetic data for class: {cls}")
        
        for i in range(images_per_class):
            # Randomly pair a content image and a style image
            c_path = random.choice(real_imgs)
            s_path = random.choice(sketch_imgs)
            
            c_tensor = nst_transform(Image.open(c_path).convert('RGB')).unsqueeze(0).to(device)
            s_tensor = nst_transform(Image.open(s_path).convert('RGB')).unsqueeze(0).to(device)
            
            # Execute L-BFGS optimization (alpha=1.0, beta=1e4 as starting point)
            gen_tensor = run_style_transfer(c_tensor, s_tensor, device=device, num_steps=300)
            
            # Denormalize and save
            final_img = inv_normalize(gen_tensor.squeeze(0).cpu())
            final_img = final_img.clamp(0, 1) # Ensure valid pixel range
            
            save_path = os.path.join(output_dir, cls, f'synth_{i}.jpg')
            save_image(final_img, save_path)
            
    print("Synthetic dataset generation completed!")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_synthetic_dataset(device=device)