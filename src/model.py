# ============================================================
# model.py — R3D-18 model architecture
# ============================================================

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from config import NUM_CLASSES, FRAME_SIZE, NUM_FRAMES


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    R3D-18 pretrained on Kinetics-400 (torchvision).
    Final FC replaced with Dropout(0.3) + Linear(512 -> num_classes).

    Reference:
        Tran et al., 2018. A Closer Look at Spatiotemporal Convolutions
        for Action Recognition. CVPR.
        Weights: Kinetics-400 (publicly available via torchvision).
    """
    model = r3d_18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def print_model_summary(model):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params    : {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Output classes  : {NUM_CLASSES}')


def verify_model(model, device):
    """Quick forward-pass sanity check."""
    dummy = torch.zeros(1, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f'Model output shape: {out.shape}  ✅')
