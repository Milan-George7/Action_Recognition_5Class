# ============================================================
# predict.py — Run inference on any video file
# Usage: python src/predict.py --video path/to/video.avi
# ============================================================

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from config import CHECKPOINT_BEST, NUM_FRAMES, RESULTS_DIR
from dataset import get_dataloaders
from model import build_model


def predict_video(video_path, model, dataset_ref, device):
    """Run inference on a single video. Returns (predicted_class, probs_tensor)."""
    model.eval()
    frames = dataset_ref.extract_frames(video_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frames)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = torch.argmax(probs).item()
    return dataset_ref.idx_to_class[pred], probs


def show_prediction(video_path, model, dataset_ref, device, save_path=None):
    pred_class, probs = predict_video(video_path, model, dataset_ref, device)
    confidence        = probs[dataset_ref.class_to_idx[pred_class]].item() * 100

    # Extract display frames
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    show_indices = set(np.linspace(0, total_frames - 1, 8).astype(int))
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

    print(f'\nVideo       : {os.path.basename(video_path)}')
    print(f'Frames      : {total_frames}  ({fps_video:.1f} fps)')
    print(f'Predicted   : {pred_class}')
    print(f'Confidence  : {confidence:.1f}%')
    print('\nAll class probabilities:')
    for cls in dataset_ref.classes:
        prob = probs[dataset_ref.class_to_idx[cls]].item() * 100
        bar  = '█' * int(prob / 5)
        print(f'  {cls:<20} {prob:5.1f}%  {bar}')

    # Plot
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
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nResult saved → {save_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Action Recognition — Single Video Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to video file (.avi/.mp4)')
    parser.add_argument('--save',  type=str, default=None,  help='Path to save result image (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f'ERROR: Video file not found: {args.video}')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data reference (for class names & frame extractor)
    _, _, _, _, test_dataset, _ = get_dataloaders()

    # Load model
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_BEST, map_location=device))
    model = model.to(device)

    save_path = args.save or os.path.join(RESULTS_DIR, 'prediction_result.png')
    show_prediction(args.video, model, test_dataset, device, save_path=save_path)


if __name__ == '__main__':
    main()
