# ============================================================
# dataset.py — Video preprocessing pipeline & Dataset class
# ============================================================

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    TARGET_DIR, NUM_FRAMES, FRAME_SIZE, BATCH_SIZE,
    NUM_WORKERS, SEED, MEAN, STD
)


# ── Temporal Jitter ──────────────────────────────────────────
def temporal_jitter(total_frames, num_frames, jitter=4):
    """Randomly shift uniformly-sampled frame indices by up to `jitter` frames."""
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    offsets = np.random.randint(-jitter, jitter + 1, size=len(indices))
    return np.clip(indices + offsets, 0, total_frames - 1)


# ── Per-Frame Transforms ─────────────────────────────────────
def build_transforms(split='train', size=FRAME_SIZE):
    """
    Train : RandomResizedCrop + RandomHorizontalFlip (augmentation).
    Val/Test: CenterCrop only (no augmentation).
    Normalization: Kinetics-400 mean/std — matches R3D-18 pretraining.
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    if split == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            normalize,
        ])


# ── Dataset ──────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Loads video clips from directory structure:
        root_dir/class_name/video_file.avi

    Args:
        root_dir   (str)  : Dataset root path.
        num_frames (int)  : Frames to sample per clip.
        split      (str)  : 'train', 'val', or 'test'.
        jitter     (bool) : Apply temporal jitter (train only).
    """

    def __init__(self, root_dir, num_frames=NUM_FRAMES, split='train', jitter=True):
        self.root_dir   = root_dir
        self.num_frames = num_frames
        self.split      = split
        self.jitter     = jitter and (split == 'train')
        self.transform  = build_transforms(split)
        self.samples    = []

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, video), self.class_to_idx[cls]))

    def extract_frames(self, video_path):
        """Extract num_frames from video. Returns tensor of shape (C, T, H, W)."""
        placeholder = torch.zeros(3, self.num_frames, FRAME_SIZE, FRAME_SIZE)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'ERROR: Cannot open {video_path}')
            return placeholder

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return placeholder

        if self.jitter:
            frame_indices = set(temporal_jitter(total, self.num_frames))
        else:
            frame_indices = set(np.linspace(0, total - 1, self.num_frames).astype(int))

        frames, idx = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                t = self.transform(frame)
                if torch.isnan(t).any() or torch.isinf(t).any():
                    cap.release()
                    return placeholder
                frames.append(t)
            idx += 1
        cap.release()

        if not frames:
            return placeholder

        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]

        return torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        return self.extract_frames(video_path), label


# ── Build Dataloaders ────────────────────────────────────────
def get_dataloaders(data_dir=TARGET_DIR):
    """
    Returns train_loader, val_loader, test_loader, full_dataset.
    Stratified 70 / 15 / 15 split.
    """
    # Full dataset to collect labels for stratified split
    full_dataset = VideoDataset(data_dir, num_frames=NUM_FRAMES, split='train', jitter=False)
    labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]

    # Stratified 70 / 15 / 15
    train_idx, temp_idx, _, temp_labels = train_test_split(
        range(len(full_dataset)), labels,
        test_size=0.30, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=temp_labels, random_state=SEED
    )

    train_dataset = VideoDataset(data_dir, num_frames=NUM_FRAMES, split='train', jitter=True)
    val_dataset   = VideoDataset(data_dir, num_frames=NUM_FRAMES, split='val',   jitter=False)
    test_dataset  = VideoDataset(data_dir, num_frames=NUM_FRAMES, split='test',  jitter=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(val_dataset,   val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(Subset(test_dataset,  test_idx),  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f'Classes ({len(full_dataset.classes)}): {full_dataset.classes}')
    print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} samples')

    return train_loader, val_loader, test_loader, full_dataset, test_dataset, test_idx
