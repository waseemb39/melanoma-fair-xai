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

## Repository Structure

```

melanoma-fair-xai/
├── XAI-Classification/               # Classification and explainability
│   ├── binary classification train.ipynb
│   ├── testing XAI.ipynb
│   └── fine tuning.ipynb
│
├── Diffusion-Based-Image-Generator/ # DDPM training and image sampling
│   ├── DiffusionM\_training\_fromZero.ipynb
│   ├── DiffusionM\_training\_fromLastCheckpoint.ipynb
│   └── dermatological\_samples\_generator.ipynb
│
├── DATABASE/                        # Metadata and scripts (datasets not included)
│   ├── ham10000\_metadata.csv
│   └── helper scripts/
│
├── figures/                         # Grad-CAM visualizations, fidelity graphs, generated images
├── requirements.txt
└── README.md

```

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

/MyDrive/DATABase/
├── HAM10000\_images/
├── ham10000\_metadata.csv
├── trained\_diffusion\_model/
└── Synthetic\_Melanocytic\_Control/

````

---

## Installation and Requirements

This project runs in Google Colab using GPU. Install the following packages:

```bash
pip install torch torchvision pytorch-lightning
pip install albumentations scikit-learn captum
pip install matplotlib seaborn
````

---

## Open in Google Colab

| Notebook                      | Link                                                                                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification Training       | [Open in Colab](https://colab.research.google.com/github/yourusername/melanoma-fair-xai/blob/main/XAI-Classification/binary%20classification%20train.ipynb)               |
| Grad-CAM and Fidelity Testing | [Open in Colab](https://colab.research.google.com/github/yourusername/melanoma-fair-xai/blob/main/XAI-Classification/testing%20XAI.ipynb)                                 |
| Diffusion Model Training      | [Open in Colab](https://colab.research.google.com/github/yourusername/melanoma-fair-xai/blob/main/Diffusion-Based-Image-Generator/DiffusionM_training_fromZero.ipynb)     |
| Synthetic Image Generation    | [Open in Colab](https://colab.research.google.com/github/yourusername/melanoma-fair-xai/blob/main/Diffusion-Based-Image-Generator/dermatological_samples_generator.ipynb) |

> Replace `yourusername` with your actual GitHub username

---

## Sample Results

| Model    | ROC AUC (Fitz 4) | Sensitivity | Balanced Accuracy |
| -------- | ---------------- | ----------- | ----------------- |
| ResNet18 | 0.9923           | 1.0000      | 0.9500            |
| ResNet50 | 0.9485           | 0.9231      | 0.9115            |

---

## Google Drive Resources

All model weights, outputs, and additional materials are available at:
[Google Drive Folder](https://drive.google.com/drive/folders/16gfyKjb4kzp3QcRdgFzr5S6uuCD7CMaK?usp=sharing)

---

## Authors

* Anna Berdichevskaia ([annab4@mail.tau.ac.il](mailto:annab4@mail.tau.ac.il))
* Waseem Bsharat ([wassemb@mail.tau.ac.il](mailto:wassemb@mail.tau.ac.il))

---

## License

This project is licensed under the MIT License.

```

---

Let me know your GitHub username so I can insert working Colab links. Would you also like a `requirements.txt` file generated from your code?
```
