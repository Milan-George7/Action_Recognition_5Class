# ============================================================
# prepare_dataset.py — Download & prepare UCF101 5-class subset
#
# USAGE (choose one method):
#
# Method 1 — Kaggle (recommended, most reliable):
#   1. Place your kaggle.json in the project root
#   2. Run: python src/prepare_dataset.py --method kaggle
#
# Method 2 — Manual:
#   1. Download UCF101.rar from https://www.crcv.ucf.edu/data/UCF101.php
#   2. Extract it so you have data/UCF-101/ folder
#   3. Run: python src/prepare_dataset.py --method manual
# ============================================================

import os
import sys
import shutil
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from config import SOURCE_DIR, TARGET_DIR, SELECTED_CLASSES


def setup_kaggle():
    """Configure Kaggle credentials from kaggle.json in project root."""
    project_root  = os.path.join(os.path.dirname(__file__), '..')
    kaggle_json   = os.path.join(project_root, 'kaggle.json')
    kaggle_dir    = os.path.expanduser('~/.kaggle')

    if not os.path.exists(kaggle_json):
        print('ERROR: kaggle.json not found in project root.')
        print('Get it from: kaggle.com → Profile → Settings → API → Create New Token')
        sys.exit(1)

    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_json, os.path.join(kaggle_dir, 'kaggle.json'))
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    print('Kaggle credentials configured.')


def download_via_kaggle():
    """Download UCF101 via Kaggle API."""
    setup_kaggle()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    print('Downloading UCF101 from Kaggle...')
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'pevogam/ucf101',
        '-p', data_dir
    ], check=True)

    print('Extracting...')
    zip_path = os.path.join(data_dir, 'ucf101.zip')
    subprocess.run(['unzip', '-q', zip_path, '-d', data_dir], check=True)
    os.remove(zip_path)
    print(f'UCF101 extracted to {data_dir}')


def prepare_5class_subset():
    """Copy only the 5 selected classes from UCF-101 to filtered_dataset."""
    if not os.path.exists(SOURCE_DIR):
        print(f'ERROR: UCF-101 folder not found at {SOURCE_DIR}')
        print('Please download and extract the dataset first.')
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f'\nPreparing 5-class subset → {TARGET_DIR}')
    for cls in SELECTED_CLASSES:
        src = os.path.join(SOURCE_DIR, cls)
        dst = os.path.join(TARGET_DIR, cls)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            count = len(os.listdir(dst))
            print(f'  ✅ {cls}: {count} videos')
        else:
            print(f'  ❌ WARNING: {cls} not found in {SOURCE_DIR}')

    print(f'\nDataset ready: {len(SELECTED_CLASSES)} classes → {TARGET_DIR}')
    print(f'Classes: {SELECTED_CLASSES}')


def main():
    parser = argparse.ArgumentParser(description='Prepare UCF101 dataset')
    parser.add_argument('--method', choices=['kaggle', 'manual'], default='manual',
                        help='kaggle = auto-download via Kaggle API | manual = already downloaded')
    args = parser.parse_args()

    if args.method == 'kaggle':
        download_via_kaggle()
    else:
        print('Manual mode — assuming UCF-101 is already extracted.')
        print(f'Expected path: {SOURCE_DIR}')

    prepare_5class_subset()
    print('\nDone! You can now run: python src/train.py')


if __name__ == '__main__':
    main()
