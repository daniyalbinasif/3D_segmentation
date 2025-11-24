import os
from pathlib import Path

ROOT_DIR = Path("/content/dataset/heart_dataset") 
DATA_DIR = ROOT_DIR 
TARGET_SHAPE = (224, 224, 96)
BATCH_SIZE = 2
EPOCHS = 150
SEED = 42

TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
LOG_DIR = ROOT_DIR / 'logs'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
RESULTS_DIR = ROOT_DIR / 'results'

ROOT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)