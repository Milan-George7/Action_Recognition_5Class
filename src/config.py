# ============================================================
# config.py — Central configuration for all scripts
# ============================================================

import os

# ── Dataset ─────────────────────────────────────────────────
# After downloading UCF101, point SOURCE_DIR to the extracted folder.
# Default assumes you extracted UCF101.rar in the project root.
SOURCE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'UCF-101')
TARGET_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_dataset')

SELECTED_CLASSES = [
    'JumpingJack',
    'PushUps',
    'HorseRiding',
    'Swimming',
    'WalkingWithDog',
]

# ── Model ────────────────────────────────────────────────────
NUM_FRAMES  = 16
FRAME_SIZE  = 112
NUM_CLASSES = len(SELECTED_CLASSES)

# ── Training ─────────────────────────────────────────────────
SEED        = 42
EPOCHS      = 10
BATCH_SIZE  = 4
NUM_WORKERS = 2
LR          = 1e-4
LR_STEP     = 5
LR_GAMMA    = 0.1
WEIGHT_DECAY = 1e-4

# ── Paths ────────────────────────────────────────────────────
CHECKPOINT_BEST  = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_action_model.pth')
CHECKPOINT_FINAL = os.path.join(os.path.dirname(__file__), '..', 'model', 'action_recognition_final.pth')
CLASS_MAP_PATH   = os.path.join(os.path.dirname(__file__), '..', 'class_mapping.json')
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'sample_results')

# Normalization: Kinetics-400 mean/std (matches R3D-18 pretraining)
MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.221459, 0.216321]
