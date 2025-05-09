# melanoma-fair-xai
 Fair and Explainable Melanoma Diagnosis using ResNet Classifiers, Grad-CAM, and Diffusion Models.
Understood. Here's your professional, emoji-free `README.md` for the root of your GitHub repository:

---

```markdown
# Fair and Explainable Melanoma Diagnosis  
Integrating ResNet Classifiers, Grad-CAM, and Diffusion-Based Image Generation

This repository presents a deep learning pipeline for melanoma classification, focused on fairness and explainability. It includes two components:
- A classification module with integrated Grad-CAM for visual explainability
- A diffusion-based image generator for producing synthetic lesion data conditioned on metadata

---

---

## Overview

Melanoma is a highly aggressive skin cancer, and early diagnosis is critical. This project introduces a diagnostic system that addresses:
- Bias in skin-tone representation through targeted fine-tuning on Fitzpatrick17k
- Model interpretability through Grad-CAM visualizations
- Synthetic data augmentation using diffusion-based generative modeling

### Datasets
- **HAM10000**: Dermoscopic images of skin lesions
- **Fitzpatrick17k**: Clinical images labeled by Fitzpatrick skin type

---

## Classification Module (XAI)

**Notebooks:**
- `binary classification train.ipynb`: Trains ResNet models (18, 34, 50, 101) using HAM10000
- `testing XAI.ipynb`: Applies Grad-CAM to visualize model decisions and analyze fidelity
- `fine tuning.ipynb`: Fine-tunes models on Fitzpatrick17k and evaluates fairness

**Features:**
- Modular implementation using PyTorch and Lightning
- Reproducible results with fixed random seed
- Grad-CAM for visual attributions and explainability
- Performance evaluation stratified by skin tone

---

## Diffusion-Based Image Generator

**Notebooks:**
- `DiffusionM_training_fromZero.ipynb`: Trains a denoising diffusion model from scratch
- `DiffusionM_training_fromLastCheckpoint.ipynb`: Resumes training from a saved checkpoint
- `dermatological_samples_generator.ipynb`: Generates synthetic lesion images based on metadata

**Outputs:**
- 256×256 RGB dermoscopic images
- Associated metadata in structured CSV format

**Directory Structure Required:**
```

## Installation and Requirements

This project runs in Google Colab using GPU. Install the following packages:

```bash
pip install torch torchvision pytorch-lightning
pip install albumentations scikit-learn captum
pip install matplotlib seaborn
````


---

## Sample Results

| Model    | ROC AUC (Fitz 4) | Sensitivity | Balanced Accuracy |
| -------- | ---------------- | ----------- | ----------------- |
| ResNet18 | 0.9923           | 1.0000      | 0.9500            |
| ResNet50 | 0.9485           | 0.9231      | 0.9115            |

---



## Authors

* Anna Berdichevskaia ([annab4@mail.tau.ac.il](mailto:annab4@mail.tau.ac.il))
* Waseem Bsharat ([wassemb@mail.tau.ac.il](mailto:wassemb@mail.tau.ac.il))


```
