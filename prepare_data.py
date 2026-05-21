import os
import shutil
import random
from pathlib import Path

def setup_challenge_data(raw_dir, output_dir, seed=42):
    """
    Filters and splits the DomainNet dataset for Challenge 7 (Group 2).
    Handles naming inconsistencies between documentation and actual folders.
    """
    # 1. Scientific rigor: Fix seed for absolute reproducibility
    random.seed(seed)
    
    # Map target class name (PDF) to source folder name (Disk)
    class_mapping = {
        'airplane': 'airplane',
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'motorcycle': 'motorbike',
        'train': 'train'
    }
    
    domains = ['real', 'sketch']
    
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    
    # Create base directory structure using the target names
    for domain in domains:
        for split in ['train', 'val', 'test']:
            for target_cls in class_mapping.keys():
                (out_path / domain / split / target_cls).mkdir(parents=True, exist_ok=True)
                
    # 2. Process Source Domain (Real)
    print("Processing Real domain (Source)...")
    for target_cls, source_cls in class_mapping.items():
        src_folder = raw_path / 'real' / source_cls
        if not src_folder.exists():
            print(f"Warning: Source folder {src_folder} not found!")
            continue
            
        # Robust extension search (case-insensitive)
        images = []
        for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
            images.extend(list(src_folder.glob(ext)))
            
        images.sort() # Sort before shuffle ensures determinism
        random.shuffle(images)
        
        # Strict split requirement: 50 train / 50 val / rest test
        train_imgs = images[:50]
        val_imgs = images[50:100]
        test_imgs = images[100:]
        
        for img in train_imgs: shutil.copy(img, out_path / 'real' / 'train' / target_cls / img.name)
        for img in val_imgs: shutil.copy(img, out_path / 'real' / 'val' / target_cls / img.name)
        for img in test_imgs: shutil.copy(img, out_path / 'real' / 'test' / target_cls / img.name)
            
    # 3. Process Target Domain (Sketch)
    print("Processing Sketch domain (Target)...")
    for target_cls, source_cls in class_mapping.items():
        src_folder = raw_path / 'sketch' / source_cls
        if not src_folder.exists():
            print(f"Warning: Source folder {src_folder} not found!")
            continue
            
        images = []
        for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
            images.extend(list(src_folder.glob(ext)))
            
        images.sort()
        random.shuffle(images)
        
        # 50 for Target fine-tuning (Phase C), the rest for full target-domain test split
        train_imgs_target = images[:50]
        test_imgs_target = images[50:]
        
        for img in train_imgs_target: shutil.copy(img, out_path / 'sketch' / 'train' / target_cls / img.name)
        for img in test_imgs_target: shutil.copy(img, out_path / 'sketch' / 'test' / target_cls / img.name)

    print("Data structure completed successfully!")

if __name__ == "__main__":
    setup_challenge_data(raw_dir='./raw_data', output_dir='./challenge_data')