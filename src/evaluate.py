# ============================================================
# evaluate.py — Full evaluation: metrics, confusion matrix,
#               prediction visualization, inference benchmark
# Run: python src/evaluate.py
# ============================================================

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CHECKPOINT_BEST, CLASS_MAP_PATH, RESULTS_DIR,
    NUM_FRAMES, NUM_CLASSES, SEED
)
from dataset import get_dataloaders, VideoDataset, build_transforms
from model import build_model


os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load Model ───────────────────────────────────────────────
def load_best_model(device):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_BEST, map_location=device))
    model = model.to(device)
    model.eval()
    print(f'Loaded best model from {CHECKPOINT_BEST}')
    return model


# ── Run Evaluation ───────────────────────────────────────────
def run_evaluation(model, test_loader, device):
    all_preds, all_labels, all_outputs = [], [], []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            videos  = videos.to(device)
            outputs = model(videos)
            all_outputs.append(outputs.cpu())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_outputs = torch.cat(all_outputs, dim=0)
    return all_preds, all_labels, all_outputs


# ── Metrics ──────────────────────────────────────────────────
def print_metrics(all_preds, all_labels, all_outputs, class_names):
    # Top-1
    top1_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    top1_acc     = 100.0 * top1_correct / len(all_labels)
    print(f'\nTop-1 Accuracy : {top1_acc:.2f}%')

    # Top-5 (k = NUM_CLASSES = 5)
    labels_tensor = torch.tensor(all_labels)
    top5_correct  = 0
    for i, output in enumerate(all_outputs):
        top5_preds = torch.topk(output, k=NUM_CLASSES).indices
        if labels_tensor[i] in top5_preds:
            top5_correct += 1
    top5_acc = 100.0 * top5_correct / len(all_labels)
    print(f'Top-5 Accuracy : {top5_acc:.2f}%  (k={NUM_CLASSES})')

    # Per-class report
    print('\nPer-class Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    print(f'Macro Avg — Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

    return top1_acc, top5_acc, precision, recall, f1


# ── Confusion Matrix ─────────────────────────────────────────
def plot_confusion_matrix(all_preds, all_labels, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — Test Set', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f'Confusion matrix saved → {out_path}')


# ── Prediction Visualization ─────────────────────────────────
def visualize_prediction(video_path, dataset_ref, model, device,
                         num_frames_to_show=8, save_name='prediction_visualization.png'):
    """Display frame grid with predicted label and top-3 class probabilities."""
    model.eval()
    frames = dataset_ref.extract_frames(video_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frames)
        probs  = torch.softmax(output, dim=1)[0]
    pred_idx   = torch.argmax(probs).item()
    pred_class = dataset_ref.idx_to_class[pred_idx]
    confidence = probs[pred_idx].item() * 100

    # Extract display frames
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    show_indices = set(np.linspace(0, total_frames - 1, num_frames_to_show).astype(int))
    display_frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in show_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frames.append(cv2.resize(frame, (112, 112)))
        idx += 1
    cap.release()

    n   = len(display_frames)
    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, n, height_ratios=[2, 1], hspace=0.4, wspace=0.1)

    for i, frm in enumerate(display_frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frm)
        ax.axis('off')
        ax.set_title(f'Frame {i+1}', fontsize=8, pad=3)

    ax_bar    = fig.add_subplot(gs[1, :])
    classes   = dataset_ref.classes
    prob_vals = [probs[dataset_ref.class_to_idx[c]].item() * 100 for c in classes]
    colors    = ['#2ecc71' if c == pred_class else '#3498db' for c in classes]

    bars = ax_bar.barh(classes, prob_vals, color=colors, edgecolor='white', height=0.5)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Confidence (%)', fontsize=10)
    ax_bar.set_title('Class Probabilities', fontsize=10)
    ax_bar.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, prob_vals):
        ax_bar.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)

    green_patch = mpatches.Patch(color='#2ecc71', label='Predicted class')
    blue_patch  = mpatches.Patch(color='#3498db', label='Other classes')
    ax_bar.legend(handles=[green_patch, blue_patch], fontsize=8, loc='lower right')

    fig.suptitle(
        f'Predicted: {pred_class}  |  Confidence: {confidence:.1f}%',
        fontsize=13, fontweight='bold', y=1.01
    )

    out_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Prediction visualization saved → {out_path}')
    print(f'Predicted Action : {pred_class}  ({confidence:.1f}%)')


# ── Inference Time Benchmark ─────────────────────────────────
def benchmark_inference(model, test_dataset, device, num_runs=20):
    model.eval()

    # Warm-up
    dummy = torch.zeros(1, 3, NUM_FRAMES, 112, 112).to(device)
    with torch.no_grad():
        _ = model(dummy)

    times = []
    for video_path, _ in test_dataset.samples[:num_runs]:
        frames = test_dataset.extract_frames(video_path).unsqueeze(0).to(device)
        start  = time.perf_counter()
        with torch.no_grad():
            _ = model(frames)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times)  * 1000
    fps    = 1.0 / np.mean(times)
    print(f'\nInference time per clip — Avg: {avg_ms:.1f} ms ± {std_ms:.1f} ms')
    print(f'Throughput: {fps:.1f} clips/sec')
    return avg_ms


# ── Main ─────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data & model
    _, _, test_loader, full_dataset, test_dataset, test_idx = get_dataloaders()
    model = load_best_model(device)

    class_names = full_dataset.classes

    # Evaluate
    all_preds, all_labels, all_outputs = run_evaluation(model, test_loader, device)
    print_metrics(all_preds, all_labels, all_outputs, class_names)
    plot_confusion_matrix(all_preds, all_labels, class_names)

    # Visualize a sample prediction
    sample_path, sample_label = test_dataset.samples[test_idx[0]]
    true_class = full_dataset.idx_to_class[sample_label]
    print(f'\nSample video true label: {true_class}')
    visualize_prediction(sample_path, test_dataset, model, device)

    # Inference benchmark
    benchmark_inference(model, test_dataset, device)


if __name__ == '__main__':
    main()
