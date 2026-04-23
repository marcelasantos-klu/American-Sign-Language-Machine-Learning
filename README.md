# American Sign Language Image Classification

This project applies machine learning and deep learning techniques to classify hand gesture images representing letters in American Sign Language (ASL). Using the Sign Language MNIST dataset, convolutional neural networks (CNNs) are trained to recognise static hand signs.

The dataset includes **24 classes (A–Y, excluding J and Z)**, as these two letters require motion and cannot be captured in static images.

---

## Environment Setup

```bash
source ~/asl_env/bin/activate
```

---

## Repository Structure

| Component | Description |
|---|---|
| `sign_language_compare_models_scheduler_test.py` | Main script comparing multiple models and scheduler configurations |
| `sign_language_customcnn_only.py` | Clean script for training a single Custom CNN model |
| `data/` | Contains training and test CSV datasets |
| `outputs_compare/` | Outputs for model comparison (plots, metrics, reports) |
| `outputs_customcnn_only/` | Outputs for single model run |
| `README.md` | Project documentation |

---

## Dataset

**Sign Language MNIST** — 28×28 grayscale images of ASL hand signs stored as CSV files.

| File | Samples |
|---|---|
| `data/sign_mnist_train.csv` | 27,455 |
| `data/sign_mnist_test.csv` | 7,172 |

The training CSV is split internally at runtime: **85% train / 15% validation**. The test CSV is held out separately for final evaluation only.

---

## Project Workflow

| Step | Description |
|---|---|
| 01 | Data loading and preprocessing |
| 02 | Dataset construction and augmentation |
| 03 | Model design (LeNet-5 and Custom CNN) |
| 04 | Training pipeline implementation |
| 05 | Validation and early stopping |
| 06 | Learning rate scheduling experiments |
| 07 | Model evaluation and comparison |
| 08 | Visualisation (EDA, confusion matrices, learning curves) |
| 09 | Final model selection |
| 10 | Result analysis and reporting |

---

## Methodology

### Data Preprocessing
- Conversion from CSV format to image tensors (28×28 grayscale)
- Normalisation of pixel values to `[0, 1]`
- Label remapping to contiguous indices

### Data Splitting
- Training set: 85%
- Validation set: 15%
- Test set: separate held-out dataset

### Data Augmentation
Applied to training data only, to improve generalisation while preserving hand structure:
- Small brightness variations
- Gaussian noise

---

## Models

### LeNet-5
A classical CNN architecture (LeCun, 1998) adapted for 28×28 grayscale input. Used as a lightweight baseline.

- 2 convolutional layers with average pooling
- 3 fully connected layers
- **~45,600 parameters**

### Custom CNN
A deeper, more robust architecture built for this task.

- 3 convolutional blocks, each with: 2 × Conv2d → BatchNorm → ReLU → MaxPool → Dropout
- Filter depth grows: 32 → 64 → 128
- Fully connected classifier: 256 units with 50% dropout
- **~588,600 parameters**

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Learning Rate | 1e-3 |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping Patience | 5 epochs |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Patience | 3 epochs |
| Random Seed | 42 |

### Learning Rate Scheduling
`ReduceLROnPlateau` reduces the learning rate when validation loss plateaus. Two decay factors were tested on the Custom CNN — **0.5** and **0.1** — to study the effect of learning rate decay aggressiveness on final performance.

### Early Stopping
Training halts after 5 epochs with no improvement in validation accuracy, restoring the best-performing weights. This prevents overfitting and avoids unnecessary computation.

---

## Evaluation

Model performance is evaluated using:
- Accuracy (training, validation, and test)
- Classification report (per-class precision, recall, F1-score)
- Confusion matrix (normalised)
- Learning curves (loss and accuracy over epochs)

---

## Results

### Model Comparison (`outputs_compare/`)

| Model | Scheduler Factor | Parameters | Best Val Accuracy | Test Accuracy |
|---|---|---|---|---|
| LeNet-5 | 0.5 | 45,616 | 100.0% | 84.68% |
| Custom CNN | **0.5** | 588,664 | 100.0% | **98.12%** |
| Custom CNN | **0.1** | 588,664 | 100.0% | 98.02% |

Both scheduler factors (0.5 and 0.1) were confirmed tested on the Custom CNN. The difference in test accuracy is minimal (0.10%), indicating the model is robust to this hyperparameter within the tested range.

### Standalone Custom CNN Run (`outputs_customcnn_only/`)

| Model | Test Accuracy |
|---|---|
| Custom CNN | **98.65%** |

### Key Observations
- The Custom CNN significantly outperforms LeNet-5 due to deeper feature extraction and modern regularisation techniques
- Learning rate scheduling improves convergence stability across both models
- Some visually similar letters (e.g. G/H, M/N) are more difficult to distinguish and show lower per-class F1-scores

---

## Outputs

Each output folder contains:

| File | Description |
|---|---|
| `1_eda.png` | EDA: class distribution, mean images per class, pixel histogram, sample grid |
| `2_*_training_curves.png` | Loss and accuracy curves over epochs |
| `3_*_confusion_matrix.png` | Normalised confusion matrix |
| `*_classification_report.txt` | Per-class precision, recall, and F1-score |
| `*_best.pth` | Saved model weights (best validation checkpoint) |
| `*_summary.csv` / `*_summary.json` | Summary of results in CSV and JSON format |

---

## How to Run

### Install dependencies

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
```

### Run single model

```bash
python sign_language_customcnn_only.py
```

### Run model comparison

```bash
python sign_language_compare_models_scheduler_test.py
```

Output files will be saved automatically to `outputs_customcnn_only/` and `outputs_compare/` respectively.
