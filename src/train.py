import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
from pathlib import Path
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, SEED, LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR
from utils import find_volumes, show_training_curves, show_augmentation_example
from data_loader import build_dataset, preprocess_pair
from model import build_unet_3d, bce_dice_loss, dice_coef
from augmentation import augment_intensity, elastic_deformation
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def train_experiment(train_pairs, val_pairs, experiment_name='baseline', augment_fn=None):
    print(f"\n Experiment: {experiment_name}")
  
    train_ds = build_dataset(train_pairs, batch_size=BATCH_SIZE, shuffle=True, augment_fn=augment_fn)
    val_ds = build_dataset(val_pairs, batch_size=BATCH_SIZE, shuffle=False, augment_fn=None)

    model = build_unet_3d()
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    ckpt_path = CHECKPOINT_DIR / experiment_name / 'best_model.h5'
    ckpt_path.parent.mkdir(exist_ok=True)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        str(ckpt_path), 
        monitor='val_dice_coef', 
        mode='max', 
        save_best_only=True, 
        verbose=1
    )
    
    early = keras.callbacks.EarlyStopping(
        monitor='val_dice_coef', 
        mode='max', 
        patience=20, 
        restore_best_weights=True
    )
    
    tb = keras.callbacks.TensorBoard(log_dir=LOG_DIR / experiment_name)

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=[checkpoint, early, tb]
    )

    final_model_path = CHECKPOINT_DIR / experiment_name / 'final_model.h5'
    model.save(str(final_model_path))
    model.load_weights(str(ckpt_path))
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Run a specific 3D Segmentation experiment.")
    parser.add_argument('experiment_mode', type=str, 
                        choices=['baseline', 'intensity', 'elastic'],
                        help="Specify which experiment to run: baseline (no aug), intensity, elastic")
    args = parser.parse_args()
    
    all_train_pairs = find_volumes(TRAIN_DIR)
    random.shuffle(all_train_pairs)
    n_val = max(1, int(0.15 * len(all_train_pairs)))
    val_pairs = all_train_pairs[:n_val]
    train_pairs = all_train_pairs[n_val:]
    print(f"Found {len(train_pairs)} train, {len(val_pairs)} val pairs.")

    if not all_train_pairs:
        print("ERROR: No training data found.")
        return

    experiments_map = {
        'baseline': (None, 'Baseline'),
        'intensity': (augment_intensity, 'Intensity_Augmentation'),
        'elastic': (elastic_deformation, 'Elastic_Deformation_Augmentation')
    }
    
    aug_fn, exp_name = experiments_map[args.experiment_mode]
    title = exp_name.replace('_', ' ')

    model, history = train_experiment(train_pairs, val_pairs, experiment_name=exp_name, augment_fn=aug_fn)
    show_training_curves(history, exp_name, RESULTS_DIR)
    if aug_fn is not None:
        img_orig, msk_orig = preprocess_pair(train_pairs[0][0], train_pairs[0][1])
        for i in range(5):
             img_aug, msk_aug = aug_fn(img_orig, msk_orig)
             show_augmentation_example(img_orig, msk_orig, img_aug, msk_aug, f'{title}_Sample_{i+1}', RESULTS_DIR)
    print(f"\nExperiment '{exp_name}' complete. Model saved to {CHECKPOINT_DIR / exp_name}.")
    
if __name__ == '__main__':
    main()