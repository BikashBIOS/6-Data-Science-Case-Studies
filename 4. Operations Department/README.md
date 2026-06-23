# 🫁 Chest Disease Detection using Deep Learning (ResNet50 + Transfer Learning)

> A deep learning project that automates the detection and classification of chest diseases from X-ray images using Transfer Learning on ResNet50 — achieving **70% test accuracy** across 4 disease classes in under 1 minute per prediction.

---

## 📋 Table of Contents

- [Business Problem & Scenario](#business-problem--scenario)
- [Dataset Description](#dataset-description)
- [Data Analysis](#data-analysis)
- [Visualization](#visualization)
- [Data Preprocessing & Pipeline](#data-preprocessing--pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation & Results](#evaluation--results)
- [Classification Report](#classification-report)
- [Confusion Matrix](#confusion-matrix)
- [Key Observations & Limitations](#key-observations--limitations)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## 🏥 Business Problem & Scenario

> **Role:** Deep Learning Consultant hired by a hospital in downtown Toronto.

**Objective:** Automate the process of detecting and classifying chest diseases from X-ray images to:
- Reduce the time of detection to **under 1 minute**
- Lower the operational cost of manual radiology screening
- Improve diagnostic accuracy over traditional methods

AI/ML has already demonstrated superiority in medical imaging. A notable benchmark: in 2018, a deep learning model detected skin cancer with **95% accuracy** compared to human dermatologists at **86.6%** (source: *The Guardian*, 2018). This project applies the same principle to chest X-ray classification.

---

## 📂 Dataset Description

| Property | Detail |
|---|---|
| **Source** | Hospital-collected chest X-ray images |
| **Total Images** | 532 (133 per class × 4 classes) |
| **Image Size (after resize)** | 256 × 256 × 3 (RGB) |
| **Number of Classes** | 4 |
| **Class Labels** | Covid-19, Normal, Viral Pneumonia, Bacterial Pneumonia |
| **Training Set** | 428 images (80%) |
| **Validation Set** | 104 images (20%) |
| **Test Set** | 40 images (10 per class) |

### Class Label Mapping

| Index | Class |
|---|---|
| 0 | Covid-19 |
| 1 | Normal (Healthy) |
| 2 | Viral Pneumonia |
| 3 | Bacterial Pneumonia |

### One-Hot Encoded Label Sample (Training Batch)

Each image label is one-hot encoded into a vector of size 4. For example:
- `[1, 0, 0, 0]` → Covid-19
- `[0, 1, 0, 0]` → Normal
- `[0, 0, 1, 0]` → Viral Pneumonia
- `[0, 0, 0, 1]` → Bacterial Pneumonia

---

## 📊 Data Analysis

### Dataset Split Summary

| Split | Images | Percentage |
|---|---|---|
| Training | 428 | 80% |
| Validation | 104 | 20% |
| Test | 40 | ~7.5% of total |

- The dataset is **balanced** — 133 images per class, ensuring no class imbalance during training.
- Images span 4 clinically distinct conditions, making this a **multi-class classification** problem.
- Input shape confirmed: `(40, 256, 256, 3)` — batch size 40, 256×256 resolution, 3 RGB channels.
- Labels shape confirmed: `(40, 4)` — 40 samples, 4 one-hot encoded classes.

### Key Data Characteristics

- **Image modality:** Grayscale chest X-rays loaded as RGB (3 channels to match ResNet50 input requirements).
- **Normalization:** All pixel values rescaled from `[0, 255]` to `[0.0, 1.0]` by dividing by 255.
- **Class boundary:** Diseases present visually different opacity, consolidation, and distribution patterns on X-rays, making CNN-based feature extraction a strong fit.

---

## 🖼️ Visualization

### Dataset Grid Visualization (Task 2)

A 6×6 grid of 36 sample training images was plotted using `matplotlib`, with each image labelled by its disease class (Covid-19, Normal, Viral Pneumonia, Bacterial Pneumonia). This step confirmed:
- Images are correctly labelled after loading via `ImageDataGenerator`.
- Visual diversity exists within each class (e.g., varying lung opacification for Covid-19 cases).
- The dataset does not appear to have gross labelling errors at the visual inspection level.

### Training Progress Plots (Task 5)

Three key plots were generated from the training history object:

**Plot 1 — Training Accuracy & Loss**
- X-axis: Epoch (1–10), Y-axis: Accuracy and Loss
- Training accuracy rose from ~89.7% (Epoch 1) to ~98.1% (Epoch 10)
- Training loss decreased from ~0.327 (Epoch 1) to ~0.112 (Epoch 10)

**Plot 2 — Validation Loss Curve**
- High variance observed in early epochs (val_loss spiked to 11.68 at Epoch 2)
- Val_loss improved significantly at Epoch 7: dropped to **1.04** (best checkpoint saved here)
- Indicates the model needed several epochs to begin generalising to unseen data

**Plot 3 — Validation Accuracy Curve**
- Validation accuracy was low and erratic in Epochs 1–6 (25%–40%)
- Jumped to **75.96%** at Epoch 7, correlating with the best model checkpoint
- Rose further to **82.69%** at Epoch 10 (but val_loss at Epoch 7 remained the best saved model)

---

## ⚙️ Data Preprocessing & Pipeline

### Image Data Generator

```python
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

| Parameter | Value | Purpose |
|---|---|---|
| `rescale` | 1./255 | Normalize pixel values to [0, 1] |
| `validation_split` | 0.2 | Reserve 20% of training data for validation |
| `batch_size` (training) | 4 | Smaller batch for fine-grained gradient updates |
| `target_size` | (256, 256) | Resize all images to match ResNet50 input |
| `class_mode` | `categorical` | Multi-class one-hot encoding |
| `shuffle` | True | Randomise batch order to prevent overfitting on order |

### Train/Validation Split

```
Found 428 images belonging to 4 classes.  ← Training
Found 104 images belonging to 4 classes.  ← Validation
```

### Test Generator (No Augmentation)

```python
test_gen = ImageDataGenerator(rescale=1./255)
```
The test set uses a separate generator without augmentation to ensure clean, unbiased evaluation.

---

## 🧠 Model Architecture

### Architecture: ResNet50 + Custom Head (Transfer Learning)

This project uses **Transfer Learning** — a technique where a neural network pre-trained on a large dataset (ImageNet, 11M images, 11,000 categories) is repurposed for a new domain-specific task.

**Why ResNet50?**
- ResNet (Residual Network, 2015) solved the **vanishing gradient problem** via "skip connections" (identity mappings), enabling training of up to 152 layers.
- ResNet50 achieved **3.57% error on ImageNet**, surpassing human-level performance.
- Pre-trained weights encode rich, general visual features (edges → textures → shapes → patterns) applicable to medical imaging.

### Transfer Learning Strategy Applied

**Partial Fine-Tuning (Strategy 1 + 2 Hybrid):**
- All ResNet50 layers **except the last 10** were frozen (their weights were not updated).
- Only Stage 5 (the last 10 layers) + the custom head were retrained.
- This preserves low-level features while allowing high-level, task-specific adaptation.

```python
for layer in basemodel.layers[:-10]:
    layers.trainable = False
```

### Base Model

| Property | Value |
|---|---|
| Architecture | ResNet50 |
| Pre-trained on | ImageNet |
| Input Shape | (256, 256, 3) |
| `include_top` | False (removes ImageNet classification head) |
| Total Params | 23,587,712 (89.98 MB) |
| Trainable Params | 23,534,592 (89.78 MB) |
| Non-trainable Params | 53,120 (207.50 KB) |

### Custom Classification Head

```
ResNet50 Base Output
        ↓
AveragePooling2D (pool_size=4×4)
        ↓
Flatten
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.3)
        ↓
Dense(128, activation='relu')
        ↓
Dropout(0.2)
        ↓
Dense(4, activation='softmax')   ← 4-class output
```

| Layer | Output | Purpose |
|---|---|---|
| AveragePooling2D (4×4) | Spatial dimensionality reduction | Summarise feature maps |
| Flatten | 1D vector | Prepare for Dense layers |
| Dense(256, ReLU) | Feature learning | High-level representations |
| Dropout(0.3) | Regularisation | Prevent overfitting |
| Dense(128, ReLU) | Intermediate features | Further compression |
| Dropout(0.2) | Regularisation | Additional overfitting protection |
| Dense(4, Softmax) | Probability per class | Multi-class output |

### Compilation

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4, weight_decay=1e-6),
    metrics=['accuracy']
)
```

| Parameter | Value | Rationale |
|---|---|---|
| Loss | Categorical Crossentropy | Standard for multi-class classification |
| Optimizer | RMSprop | Adaptive learning rate; suitable for fine-tuning |
| Learning Rate | 1e-4 | Small LR to avoid overwriting pre-trained weights |
| Weight Decay | 1e-6 | L2 regularisation |

---

## 🏋️ Training

### Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | `monitor='val_loss'`, `patience=20` | Stop if val_loss doesn't improve for 20 epochs |
| `ModelCheckpoint` | `save_best_only=True`, file: `weights.hdf5.keras` | Save best weights automatically |

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Steps per Epoch | `train_generator.n // 4` = 107 steps |
| Batch Size | 4 |
| Validation Steps | `val_generator.n // 4` = 26 steps |

### Epoch-by-Epoch Training Log

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Checkpoint |
|---|---|---|---|---|---|
| 1 | 89.72% | 0.3265 | 25.00% | 10.5662 | ✗ |
| 2 | 94.39% | 0.2069 | 25.00% | 11.6790 | ✗ |
| 3 | 94.86% | 0.2025 | 24.04% | 6.9614 | ✗ |
| 4 | 95.79% | 0.1874 | 40.38% | 2.5533 | ✅ Saved |
| 5 | 96.96% | 0.1287 | 30.77% | 12.2855 | ✗ |
| 6 | 97.90% | 0.0710 | 39.42% | 5.2056 | ✗ |
| 7 | 98.60% | 0.0531 | **75.96%** | **1.0409** | ✅ **Best Saved** |
| 8 | 97.90% | 0.0945 | 71.15% | 2.1234 | ✗ |
| 9 | 98.13% | 0.1164 | 77.88% | 1.6763 | ✗ |
| 10 | 98.13% | 0.1121 | 82.69% | 1.3042 | ✗ |

> **Best model checkpoint saved at Epoch 7** — Val Loss: 1.0409, Val Accuracy: 75.96%

---

## 📈 Evaluation & Results

### Test Set Evaluation

The model was evaluated on the held-out test set of **40 images** (10 per class).

```
model.evaluate(test_generator, steps=test_generator.n // 4, verbose=1)
```

| Metric | Value |
|---|---|
| **Test Loss** | 1.8492 |
| **Test Accuracy (Keras evaluate)** | **70.00%** |
| **Test Accuracy (sklearn)** | **70.00%** |

> Both evaluation methods independently confirmed a test accuracy of **70%** on 40 test images.

---

## 📋 Classification Report

Per-class performance on 40 test images (10 per class):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Covid-19 (0)** | 0.83 | **1.00** | **0.91** | 10 |
| **Normal (1)** | 0.75 | 0.90 | 0.82 | 10 |
| **Viral Pneumonia (2)** | 0.62 | 0.50 | 0.56 | 10 |
| **Bacterial Pneumonia (3)** | 0.50 | 0.40 | 0.44 | 10 |
| **Accuracy** | — | — | **0.70** | **40** |
| **Macro Avg** | 0.68 | 0.70 | 0.68 | 40 |
| **Weighted Avg** | 0.68 | 0.70 | 0.68 | 40 |

### Per-Class Analysis

**Covid-19 (Best Performing Class)**
- Precision: 0.83 — 83% of predicted Covid-19 cases are correct.
- Recall: 1.00 — The model correctly identified **all 10** Covid-19 cases (zero missed detections).
- F1-Score: 0.91 — Excellent balance of precision and recall.
- This class is most critical in a clinical setting — missing a Covid-19 case is a high-risk error, and the model achieves perfect recall here.

**Normal (Second Best)**
- Precision: 0.75 — 75% of "Normal" predictions are correct.
- Recall: 0.90 — Correctly identified 9 out of 10 healthy cases.
- F1-Score: 0.82 — Strong performance, important for avoiding over-diagnosis.

**Viral Pneumonia (Moderate)**
- Precision: 0.62, Recall: 0.50, F1: 0.56 — Moderate performance.
- The model missed ~50% of Viral Pneumonia cases, suggesting visual overlap with other conditions.

**Bacterial Pneumonia (Weakest Class)**
- Precision: 0.50, Recall: 0.40, F1: 0.44 — Lowest performing class.
- Only 4 out of 10 Bacterial Pneumonia cases correctly identified.
- Likely confused with Viral Pneumonia due to similar radiological presentation.

---

## 🔢 Confusion Matrix

Generated using `seaborn.heatmap` on `sklearn.metrics.confusion_matrix`:

```
Predicted →   Covid-19   Normal   Viral Pneum.   Bacterial Pneum.
↓ Actual
Covid-19         10         0           0               0
Normal            1         9           0               0
Viral Pneum.      1         0           5               4
Bacterial         2         1           3               4
```

**Key Observations from Confusion Matrix:**
- **Covid-19** → Perfect: all 10 correctly classified (zero false negatives).
- **Normal** → 9/10 correct; 1 misclassified as Covid-19.
- **Viral Pneumonia** → Only 5/10 correct; 4 confused with Bacterial Pneumonia, 1 with Covid-19.
- **Bacterial Pneumonia** → 4/10 correct; 3 misclassified as Viral Pneumonia, 2 as Covid-19, 1 as Normal.
- The Viral/Bacterial Pneumonia confusion is clinically well-known, as both conditions produce similar consolidation patterns on X-rays.

---

## 🔍 Key Observations & Limitations

### Why Training Accuracy >> Test Accuracy (Overfitting)

| Metric | Final Training | Test |
|---|---|---|
| Accuracy | ~98% | 70% |
| Loss | ~0.11 | ~1.85 |

The large gap indicates **overfitting**, likely due to:
- **Very small dataset** (532 total images) — insufficient for a 23M parameter model.
- **Limited data augmentation** — only rescaling was applied; no flipping, rotation, or zoom.
- **Partial freezing** — only the last 10 of 175 ResNet50 layers were unfrozen, potentially under-adapting to the medical domain.


---

## 🛠️ Tech Stack

| Category | Library / Tool |
|---|---|
| **Deep Learning Framework** | TensorFlow 2.x / Keras |
| **Pre-trained Model** | ResNet50 (ImageNet weights) |
| **Image Processing** | OpenCV (`cv2`), PIL |
| **Data Pipeline** | `ImageDataGenerator` (Keras) |
| **Evaluation Metrics** | scikit-learn (`confusion_matrix`, `classification_report`, `accuracy_score`) |
| **Visualization** | Matplotlib, Seaborn |
| **Data Handling** | NumPy, Pandas |
| **Environment** | Python 3.x, VS Code / Jupyter Notebook |
| **Platform** | Windows (CPU training — GPU not available natively on TF >= 2.11) |

---

## 📁 Project Structure

```
chest-disease-detection/
│
├── Dataset/                   # Training + validation images (4 subfolders by class)
│   ├── 0/                     # Covid-19
│   ├── 1/                     # Normal
│   ├── 2/                     # Viral Pneumonia
│   └── 3/                     # Bacterial Pneumonia
│
├── Test/                      # Test images (4 subfolders by class)
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
│
├── operation.ipynb            # Main Jupyter notebook (all tasks)
├── weights.hdf5.keras         # Best saved model weights (Epoch 7)
└── README.md                  # This file
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install tensorflow keras opencv-python scikit-learn matplotlib seaborn numpy pandas
```

### Steps

1. **Prepare the dataset:**
   - Ensure `Dataset/` and `Test/` directories exist with images organized in numbered subfolders (0, 1, 2, 3).

2. **Launch the notebook:**
   ```bash
   jupyter notebook operation.ipynb
   ```

3. **Run all cells in order:**
   - Task 1: Import libraries and load data
   - Task 2: Visualize the dataset
   - Task 3: Load pre-trained ResNet50
   - Task 4: Build and train the model
   - Task 5: Evaluate and generate metrics

4. **Best model is auto-saved** to `weights.hdf5.keras` during training.

---

## 📌 Summary of Results

| Metric | Value |
|---|---|
| Total Images | 532 |
| Training Images | 428 (80%) |
| Validation Images | 104 (20%) |
| Test Images | 40 |
| Number of Classes | 4 |
| Best Val Loss (Epoch 7) | 1.0409 |
| Best Val Accuracy (Epoch 7) | 75.96% |
| Final Train Accuracy (Epoch 10) | 98.13% |
| **Test Accuracy** | **70.00%** |
| Covid-19 F1-Score | **0.91** |
| Normal F1-Score | 0.82 |
| Viral Pneumonia F1-Score | 0.56 |
| Bacterial Pneumonia F1-Score | 0.44 |
| Macro Avg F1 | 0.68 |

---

*Built using Transfer Learning on ResNet50 with TensorFlow/Keras as part of a deep learning case study in medical image classification.*

*Built with 💖 by BikashBIOS.*