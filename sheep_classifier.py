# sheep_classification.py

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ========= Setup ========= #

SEED = 42
NUM_CLASSES = 7
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 300
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {"Naeimi": 0, "Goat": 1, "Sawakni": 2, "Roman": 3, "Najdi": 4, "Harri": 5, "Barbari": 6}
inv_label_map = {v: k for k, v in label_map.items()}

DATA_DIR = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

df = pd.read_csv(LABELS_CSV)

# ========= Reproducibility ========= #

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# ========= Transforms ========= #

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========= Dataset Classes ========= #

class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = label_map[self.df.iloc[idx]['label']]
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# ========= Model ========= #

def get_model():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model

# ========= Compute ECE ========= #

def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(mask):
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(probs)) * np.abs(bin_confidence - bin_accuracy)
    return ece

# ========= Training ========= #

all_labels = df['label'].map(label_map).values
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- Fold {fold+1} ---")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(SheepDataset(train_df, TRAIN_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SheepDataset(val_df, TRAIN_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1, early_stop = 0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_preds, train_labels = [], []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_preds.extend(out.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.numpy())
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"model_fold{fold}.pth")
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= PATIENCE:
            print("Early stopping.")
            break

# ========= Evaluation ========= #

print("\n--- Validation Performance Across Folds ---")
all_probs, all_preds, all_labels = [], [], []
fold_scores = []

for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_loader = DataLoader(SheepDataset(val_df, TRAIN_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(f"model_fold{fold}.pth"))
    model.eval()

    fold_probs, fold_preds, fold_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            fold_probs.extend(probs)
            fold_preds.extend(np.argmax(probs, axis=1))
            fold_labels.extend(y.numpy())

    all_probs.extend(fold_probs)
    all_preds.extend(fold_preds)
    all_labels.extend(fold_labels)

    f1 = f1_score(fold_labels, fold_preds, average='macro')
    print(f"Fold {fold+1} F1 Score: {f1:.4f}")
    fold_scores.append(f1)

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="BuGn",
            xticklabels=label_map.keys(), yticklabels=label_map.keys(), cbar=False)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted\n", fontsize=12)
plt.ylabel("True", fontsize=12)

# ECE Below Confusion Matrix
ece_score = compute_ece(np.array(all_probs), np.array(all_labels))
plt.figtext(0.5, -0.03, f"Expected Calibration Error (ECE): {ece_score:.4f}",
            ha="center", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ========= Save F1 per Fold Plot ========= #

x_labels = np.array([f'Fold {i+1}' for i in range(NUM_FOLDS)])
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=x_labels, y=fold_scores, palette='viridis')
mean_score = np.mean(fold_scores)
plt.axhline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {mean_score:.4f}")
for i, score in enumerate(fold_scores):
    plt.text(i, score + 0.002, f"{score:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
plt.title("Validation F1 Score per Fold", fontsize=16, fontweight='bold')
plt.xlabel("\nFold", fontsize=12)
plt.ylabel("F1 Score\n", fontsize=12)
plt.ylim(0.85, 1.0)
barplot.yaxis.grid(True, linestyle='--', alpha=0.7)
sns.despine()
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("f1_per_fold.png", dpi=300)
plt.close()
