# Dogs vs Cats Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project comparing a custom CNN architecture against transfer learning with VGG16 for binary image classification on the Dogs vs Cats dataset.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Project Overview

This project explores deep learning for computer vision, specifically examining:

1. **Building a custom CNN from scratch** — Designing and training a convolutional neural network without pre-trained weights
2. **Implementing transfer learning** — Adapting a pre-trained VGG16 model to the binary classification task
3. **Comprehensive model evaluation** — Assessing performance across multiple metrics and visualization techniques
4. **Error analysis** — Investigating model failures and their underlying causes

### Learning Objectives

- Understand the principles behind CNN architecture design
- Apply transfer learning effectively on limited training data
- Use data augmentation to improve generalization
- Evaluate and compare models using industry-standard metrics
- Identify systematic failure patterns through error analysis

## 📊 Dataset

**Source:** [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

**Subset Used:** 5,000 images
- **Training:** 4,000 images (2,000 dogs + 2,000 cats)
- **Validation:** 500 images (250 dogs + 250 cats)
- **Test:** 500 images (250 dogs + 250 cats)

### Data Characteristics
- **Image dimensions:** Variable (200–500 pixels typical)
- **Standardized input size:** 150×150×3 (RGB)
- **Format:** JPEG
- **Classes:** Binary (0 = Cats, 1 = Dogs)

### Data Preprocessing
- **Rescaling:** Pixel values normalized to [0, 1]
- **Augmentation (training only):**
  - Rotation: ±40°
  - Width/Height shift: ±20%
  - Shear transformation: 20%
  - Zoom: ±20%
  - Horizontal flip
  - Fill mode: Nearest

## 🔧 Installation

### Prerequisites

```bash
Python 3.8+
pip
Git
```

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/thuan20132000/CSCN8010_dogs-vs-cats-classification.git
cd CSCN8010_dogs-vs-cats-classification
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
   - Extract to the `data/train/` directory
   - The notebook will automatically split the data into train/validation/test sets

### Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.0.0
jupyter>=1.0.0
```

## 📁 Project Structure

```
dogs-vs-cats-classification/
│
├── data/
│   ├── train/                    # Original training images
│   └── organized/                # Organized into train/val/test splits
│       ├── train/
│       │   ├── dogs/
│       │   └── cats/
│       ├── validation/
│       │   ├── dogs/
│       │   └── cats/
│       └── test/
│           ├── dogs/
│           └── cats/
│
├── dogs_vs_cats_classification.ipynb  # Main Jupyter notebook
├── Dogs_vs_Cats_Assignment_Cover.pdf  # Assignment cover page
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── models/                           # Saved model weights (after training)
│   ├── best_custom_cnn.keras
│   └── best_vgg16_transfer.keras
│
└── figures/                          # Generated visualizations (optional)
    ├── eda/
    ├── training_history/
    └── evaluation/
```

## 🧠 Models

### 1. Custom CNN Architecture

**Design Philosophy:**
- Progressive feature extraction with increasing filter depth
- Batch normalization for training stability
- Dropout regularization to prevent overfitting
- L2 regularization on dense layers

**Architecture:**
```
Input (150×150×3)
    ↓
Block 1: Conv2D(32)×2 → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv2D(64)×2 → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128)×2 → MaxPool → Dropout(0.25)
    ↓
Block 4: Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → Dense(256) → Dense(1, sigmoid)
```

**Total Parameters:** ~3–5 million (all trainable)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary crossentropy
- Batch size: 32
- Early stopping: Patience 10
- Learning rate reduction: Factor 0.5, patience 5

### 2. VGG16 Transfer Learning

**Transfer Learning Strategy:**
- Pre-trained VGG16 base (ImageNet weights) — **frozen**
- Custom classification head — **trainable**
- Global Average Pooling instead of Flatten

**Architecture:**
```
Input (150×150×3)
    ↓
VGG16 Base (14.7M frozen parameters)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

**Total Parameters:** ~15.8 million (1.1M trainable, 14.7M frozen)

**Training Configuration:**
- Optimizer: Adam (lr=0.0001)
- Loss: Binary crossentropy
- Batch size: 32
- Early stopping: Patience 10

## 📈 Results

### Actual Model Performance (Test Set — 500 Samples, 5 Epochs, CPU Only)

| Metric | Custom CNN | VGG16 Transfer | Winner |
|--------|-----------|----------------|--------|
| **Test Accuracy** | 0.508 (254/500) | **0.838 (419/500)** | VGG16 ✓ |
| **Precision (Dogs)** | 0.5068 | **0.8073** | VGG16 ✓ |
| **Recall (Dogs)** | 0.5920 | **0.8880** | VGG16 ✓ |
| **F1-Score (Dogs)** | 0.5461 | **0.8457** | VGG16 ✓ |
| **Precision (Cats)** | 0.5096 | **0.8756** | VGG16 ✓ |
| **Recall (Cats)** | 0.4240 | **0.7880** | VGG16 ✓ |
| **F1-Score (Cats)** | 0.4629 | **0.8295** | VGG16 ✓ |
| **ROC AUC** | 0.5202 | **0.9212** | VGG16 ✓ |
| **PR AUC** | 0.5409 | **0.9271** | VGG16 ✓ |
| **Val Accuracy (Ep. 5)** | 0.536 | **0.880** | VGG16 ✓ |
| **Model Size** | ~15–20 MB | ~60–65 MB | Custom ✓ |
| **Inference Speed** | Faster | Slower | Custom ✓ |

> Training was performed on CPU (~12 min/epoch), limited to 5 epochs per model.

### Key Visualizations

The notebook includes:
- **Training curves** (accuracy and loss over epochs)
- **Confusion matrices** for both models
- **Precision-Recall curves**
- **ROC curves**
- **Sample predictions** with confidence scores
- **Error analysis** with misclassified examples
- **Confidence distribution** analysis

## 🚀 Usage

### Running the Complete Notebook

1. **Start Jupyter Notebook:**
```bash
jupyter notebook practical_lab3.ipynb
```

2. **Execute cells sequentially:**
   - The notebook is designed to run from top to bottom
   - The first execution will organize the dataset
   - Training both models takes approximately 2 hours on CPU, ~1 hour on GPU

### Running Individual Sections

You can run specific sections independently:

```python
# Exploratory Data Analysis only
# Run cells in Section 3

# Train only Custom CNN
# Run cells in Section 4.1

# Train only VGG16
# Run cells in Section 4.2

# Evaluation only (requires saved models)
# Run Section 5 with pre-trained model files
```

### Using Pre-trained Models

If you have the saved model files:

```python
from tensorflow import keras

# Load models
custom_cnn = keras.models.load_model('models/best_custom_cnn.keras')
vgg16_model = keras.models.load_model('models/best_vgg16_transfer.keras')

# Make predictions
predictions = custom_cnn.predict(test_generator)
```

## 🔍 Key Findings

### 1. Transfer Learning is Essential on Limited Data
VGG16 outperformed the custom CNN by 33.0 percentage points (83.8% vs 50.8%). The custom CNN's ROC AUC of 0.52 — barely above random — confirms it failed to learn meaningful features within the constraints of 5 CPU epochs.

### 2. VGG16 Converged Rapidly
VGG16 reached 70.8% validation accuracy after just the first epoch and climbed to 88.0% by epoch 3, with a final ROC AUC of 0.921 and PR AUC of 0.927. These scores indicate strong, reliable discrimination across all confidence thresholds.

### 3. Custom CNN Class Bias
The custom CNN showed slightly higher recall for dogs (0.592) than cats (0.424), suggesting a mild default bias toward one class — a sign the model had not meaningfully converged and was not making informed predictions.

### 4. CPU Training Was the Primary Bottleneck
Each epoch took approximately 12–15 minutes on CPU. This limited both models to 5 epochs, which is insufficient for the custom CNN to converge from scratch. With GPU access and 20–30 epochs, the custom CNN would likely reach 75–85% accuracy.

### 5. Shared Error Patterns
Both models struggled with the same difficult images: unusual camera angles, partially occluded subjects, poor lighting, extreme close-ups or distant shots, and frames with multiple animals — issues rooted in dataset ambiguity rather than model architecture.

### 6. Practical Guidance
- **For production use:** VGG16 is the only viable option in its current form
- **Do not deploy the custom CNN** at 50.8% accuracy — it offers no predictive value over a coin flip
- **To close the gap:** GPU training for 20–30 epochs would allow the custom CNN to demonstrate its true potential
- **Next step for VGG16:** Unfreeze the top convolutional blocks for fine-tuning to push accuracy toward 90–95%

## 👥 Contributors

- ANTHONY IZEVBOKUN

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset:** Kaggle Dogs vs Cats competition
- **Original dataset:** Microsoft Research & Petfinder.com (Asirra project)
- **Pre-trained model:** VGG16 from ImageNet (Simonyan & Zisserman, 2014)
- **Course materials:** CSCN8010 class notebooks and resources
- **Deep learning framework:** TensorFlow/Keras team

## 📚 References

1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
2. Elson, J., Douceur, J., Howell, J., & Saul, J. (2007). Asirra: A CAPTCHA that exploits interest-aligned manual image categorization. ACM CCS.
3. Keras Documentation: https://keras.io/
4. TensorFlow Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning

---

*Last updated: April 2026*
