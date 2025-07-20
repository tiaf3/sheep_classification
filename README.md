
# Sheep Breed Classification - DAAL 2025 Challenge

This project was developed for the **Eid Al-Adha 2025: Sheep Classification Challenge**, hosted on Kaggle by **DAAL**.  
The goal was to classify sheep images into **7 distinct breeds** using deep learning.

---

## Highlights

- **Model**: EfficientNetV2-S (pretrained on ImageNet)
- **Cross-Validation**: 5-Fold Stratified
- **Loss Function**: Class-Weighted CrossEntropy
- **Inference**: Softmax Ensembling weighted by F1 score
- **Validation Accuracy**: `96.0%`
- **Final Validation Macro F1**: `0.9605`
- **Expected Calibration Error (ECE)**: `0.0271`
- **Private Leaderboard Score**: `0.96516`

---

## Technical Overview

| Component        | Description                                                  |
|------------------|--------------------------------------------------------------|
| **Architecture** | EfficientNetV2-S with fine-tuning                            |
| **Augmentation** | Resize, RandomCrop, HorizontalFlip, ColorJitter, Erasing     |
| **Regularization** | EarlyStopping, ReduceLROnPlateau                          |
| **Evaluation**   | F1 Score (macro), Confusion Matrix, Classification Report    |
| **Calibration**  | Expected Calibration Error (ECE)                             |
| **Ensembling**   | Weighted softmax across folds                                |

---

## Target Classes

- Naeimi
- Goat
- Sawakni
- Roman
- Najdi
- Harri
- Barbari

---

## Visualizations

### 🔹 Sample Images per Class  
<img src="images/Sample_Sheep.png" width="600"/>

### 🔹 Class Distribution  
<img src="images/Class_Distribution.png" width="600"/>

### 🔹 Validation F1 Score per Fold  
<img src="images/f1_per_fold.png" width="500"/>

### 🔹 Confusion Matrix  
<img src="images/confusion_matrix.png" width="500"/>

---

## Acknowledgments

- Special thanks to **DAAL** for organizing the 2025 Eid challenge.
- Pretrained models from `torchvision.models`.
- Implemented using **PyTorch**, **NumPy**, and **Seaborn**.

---

> This project demonstrates a robust classification workflow with strong generalization, calibration, and performance tuning.
