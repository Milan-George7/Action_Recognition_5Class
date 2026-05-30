# ============================================================
# train.py — Full training pipeline
# Run: python src/train.py
# ============================================================

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SEED, EPOCHS, LR, LR_STEP, LR_GAMMA, WEIGHT_DECAY,
    CHECKPOINT_BEST, CLASS_MAP_PATH, RESULTS_DIR
)
from dataset import get_dataloaders
from model import build_model, print_model_summary, verify_model


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'All random seeds fixed to {seed} for reproducibility.')


# ── Training & Validation ────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for videos, labels in tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Train]'):
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        total         += labels.size(0)
        correct       += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Val]  '):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss    = criterion(outputs, labels)
            total_loss    += loss.item()
            _, predicted   = torch.max(outputs, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


# ── Plot Training Curves ─────────────────────────────────────
def plot_curves(history, save_dir=RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'],   label='Val Loss',   marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'],   label='Val Acc',   marker='o')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f'Training curves saved → {out_path}')


# ── Main ─────────────────────────────────────────────────────
def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    train_loader, val_loader, test_loader, full_dataset, test_dataset, test_idx = get_dataloaders()

    # Save class mapping
    os.makedirs(os.path.dirname(CLASS_MAP_PATH), exist_ok=True)
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)
    print(f'Class mapping saved → {CLASS_MAP_PATH}')

    # Model
    model = build_model(pretrained=True)
    model = model.to(device)
    print_model_summary(model)
    verify_model(model, device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(CHECKPOINT_BEST), exist_ok=True)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device, epoch, EPOCHS)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f'Epoch {epoch:02d} | Train Loss: {t_loss:.4f}  Acc: {t_acc:.2f}% | '
              f'Val Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%')

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CHECKPOINT_BEST)
            print(f'  ✅ Best model saved (val acc: {best_val_acc:.2f}%)')

    print(f'\nTraining complete. Best val accuracy: {best_val_acc:.2f}%')
    plot_curves(history)

    # Return for use by evaluate.py
    return model, test_loader, full_dataset, test_dataset, test_idx, device


if __name__ == '__main__':
    main()
