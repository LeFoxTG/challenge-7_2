# Challenge 7: Few-Shot Classification, Neural Style Transfer, and Domain Shift Adaptation

**Course:** Machine Learning  
**Institution:** Universidad Distrital Francisco José de Caldas  
**Group:** 2

This repository contains the complete reproducible pipeline for Challenge 7. The project investigates the Domain Shift phenomenon by transferring deep representations from real photographs to textureless sketches, utilizing a ResNet-50 backbone, L-BFGS Neural Style Transfer (NST), and sparse target-domain fine-tuning.

---

## CHECKLIST

As requested by the challenge guidelines, the following summarizes the core experimental setup and findings:

* **Assigned Dataset Pair and Category Subset:** * **Source Domain:** Real (DomainNet)
    * **Target Domain:** Sketch (DomainNet)
    * **Categories (6):** airplane, bicycle, bus, car, motorcycle (mapped from 'motorbike' in raw data), train.
* **Pretrained Backbone Chosen and Trainable Parameters:**
    * **Backbone:** `ResNet-50` (pretrained on `IMAGENET1K_V2`).
    * **Feature Extraction (Phase A1):** ~12,294 trainable parameters (only the 6-way linear head is unfrozen).
    * **Fine-Tuning (Phase A2 & C):** ~14.9M trainable parameters (unfrozen `layer3`, `layer4`, and `fc` head).
* **Performance Across All 5 Model Variants (Mean ± Std):**

| Model Variant | Source-Domain Acc (Real) | Target-Domain Acc (Sketch) |
| :--- | :--- | :--- |
| 1. From-Scratch Baseline | 0.3540 ± 0.0210 | 0.1520 ± 0.0180 |
| 2. Feature Extraction | 0.8850 ± 0.0150 | 0.4120 ± 0.0110 |
| 3. Fine-Tuning (Best Part A) | **0.9365 ± 0.0080** | 0.5545 ± 0.0120 |
| 4. Target Fine-Tuning | 0.9329 ± 0.0090 | **0.8135 ± 0.0100** |
| 5. Style-Transfer Augmentation| 0.9433 ± 0.0070 | 0.6486 ± 0.0140 |

* **Domain Shift Penalty ($\Delta shift$):**
    * **Best Part A Model (Baseline):** $\Delta shift = 0.3819$
    * **Best Part C Model (Target FT):** $\Delta shift = 0.1194$
* **Neural Style Transfer Settings and Visual Quality:**
    * **Ratio:** $\alpha/\beta = 10^{-4}$ (Optimized with L-BFGS for 300 steps).
    * **Quality Assessment:** The generated images successfully preserve the geometric content (chassis, wheels, windows) of the real photographs while adopting the stark, high-contrast, edge-only style of the sketches. No structural collapse was observed, making them highly viable for data augmentation.
* **Best Adaptation Strategy (Conclusion):**
    For the Real $\to$ Sketch domain pair, **Target Fine-Tuning** proved to be the most effective strategy, recovering over 25% of target accuracy and reducing the $\Delta shift$ to 0.1194. Sketches entirely lack the color and gradient textures present in real photos. Unfreezing `layer4` and training directly on 50 sketch samples allowed the CNN filters to rapidly recalibrate from texture-detectors to pure edge-detectors. While *Style-Transfer Augmentation* improved target accuracy by 9.4% completely unsupervised, the extreme sensory gap between photos and line drawings makes supervised fine-tuning mathematically superior when a small target-label budget is available.

---

## Reproducibility and Execution Guide

### 1. Environment Setup
To guarantee strict determinism, identical preprocessing was applied across domains, and all pseudo-random number generators (PyTorch, NumPy, Python) were fixed to `seed=42`. 

Ensure your environment has a GPU available. Install the exact dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download real.zip and sketch.zip from DomainNet and extract them into the raw_data/ directory. Then, execute the data preparation script to strictly isolate the 50-shot training splits and avoid data leakage:

```bash
python prepare_data.py
```

Expected output: A cleanly structured challenge_data/ folder.

### 3. Training the Part A Baseline

Execute the baseline classifier which sequentially performs Feature Extraction and Fine-Tuning:

```bash
python main.py
```

Expected output: Training logs and the serialized weights file `resnet50_phaseA.pt`.

### 4. Running Neural Style Transfer (Part B)

Generate the 180 synthetic cross-domain training tokens (30 per class):

```bash
python generate_synthetic.py
```

Expected output: The directory `data/synthetic_target/` populated with stylized images.

### 5. Evaluating Domain Adaptation (Part C)

Run the adaptation pipeline to evaluate the three strategies (Baseline, Target Fine-Tuning, and Style-Transfer Augmentation):

```bash
python main_phaseC.py
```

Expected output: Final comparative metric table printed to the console and updated `.pt` weights.

### 6. Generating Explainability Figures

Extract t-SNE projections and Grad-CAM spatial attention maps:

```bash
python visualizations.py
```

Expected output: `tsne_faseA.png` and `gradcam_sample.png` saved in the root directory.

## Video

[Link to video](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAbb2uELBWVGTqaxLPugW1zY?e=5dTRpc&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

### Timestamps
[0:00](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAbb2uELBWVGTqaxLPugW1zY?e=gAuRck&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7fX0%3D) - Presentation

[0:50](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=Zhq9xO&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NTAuMjN9fQ%3D%3D) - The Domain Shift Problem

[1:39](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=b115no&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6OTkuMX19) - Phase A: Few-Shot Baseline

[2:33](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=4RMXWX&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTUzLjUxfX0%3D) - Baseline Results & Feature Segregation (t-SNE)

[3:12](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=da5ZxO&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTkyLjQzfX0%3D) - Phase B: Neural Style Transfer (NST)

[4:18](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=La5KkR&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjU4LjZ9fQ%3D%3D) - Qualitative Results: Style-Transfer Synthesis

[5:07](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=5ZNqh7&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzA3LjI1fX0%3D) - Phase C: Domain Adaptation Strategies

[6:09](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=HcQnG3&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzY5LjYzfX0%3D) - Quantitative Performance Matrix

[7:10](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=V4tYCb&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D) - Explainability: Grad-CAM Attention Shift

[8:00](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQDZ8sLgrOWtTLTmrqzYGLZtAfGwNrQx3oEVb5bpeLNBp-4?e=XL6x7u&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDgwLjQ2fX0%3D) - Conclusions
