import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm
import os
from pathlib import Path
from scipy.spatial.distance import cdist
    
def find_volumes(base_dir):
    pairs = []
    img_dir = os.path.join(base_dir, 'images')
    msk_dir = os.path.join(base_dir, 'masks')
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        train_img = os.path.join(base_dir, 'train', 'images')
        if os.path.isdir(train_img):
            img_dir = train_img
            msk_dir = os.path.join(base_dir, 'train', 'masks')
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        print(f"Warning: expected folders not found under {base_dir}: {img_dir} or {msk_dir}")
        return pairs

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(msk_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    for img_name in image_files:
        if img_name in mask_files:
            img_path = os.path.join(img_dir, img_name)
            msk_path = os.path.join(msk_dir, img_name)
            pairs.append((img_path, msk_path))
        else:
            stem = Path(img_name).stem
            candidates = [stem + suffix for suffix in ['_mask.nii.gz','_mask.nii','_seg.nii.gz','_seg.nii','_label.nii.gz','_label.nii']]
            found = False
            for c in candidates:
                if c in mask_files:
                    pairs.append((os.path.join(img_dir, img_name), os.path.join(msk_dir, c)))
                    found = True
                    break
            if not found:
                print(f"No mask found for image: {img_name}")
    return pairs

def dice_coef_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    return float(2 * intersection / (np.sum(pred_bin) + np.sum(gt_bin) + eps))

def sensitivity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    tp = np.sum(pred_bin & gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    return float(tp / (tp + fn + eps))

def precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    return float(tp / (tp + fp + eps))

def hd95(pred_b, true_b, spacing=(1,1,1)):
    pred_b = pred_b.astype(bool)
    true_b = true_b.astype(bool)
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_b ^ binary_erosion(pred_b, structure=struct)
    true_surface = true_b ^ binary_erosion(true_b, structure=struct)
    pred_pts = np.array(np.nonzero(pred_surface)).T
    true_pts = np.array(np.nonzero(true_surface)).T

    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.nan

    pred_pts = pred_pts * spacing
    true_pts = true_pts * spacing
    D = cdist(pred_pts, true_pts)
    dist_pred_to_true = D.min(axis=1)
    dist_true_to_pred = D.min(axis=0)
    all_dists = np.concatenate([dist_pred_to_true, dist_true_to_pred])
    return np.percentile(all_dists, 95)

def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    y_predb = (y_pred >= 0.5).astype(np.uint8)
    y_trueb = (y_true >= 0.5).astype(np.uint8)
    if y_predb.ndim == 4 and y_predb.shape[-1] == 1:
        y_predb = y_predb.squeeze(-1)
        y_trueb = y_trueb.squeeze(-1)
    dice = dice_coef_np(y_predb, y_trueb)
    prec = precision(y_predb, y_trueb)
    sens = sensitivity(y_predb, y_trueb)
    h = hd95(y_predb, y_trueb, spacing)

    return {
        'Dice': dice,
        'Precision': prec,
        'Sensitivity': sens,
        'HD95': h
    }

def show_training_curves(history, experiment_name, results_dir: Path):
    plt.figure(figsize=(10, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{experiment_name} - Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()
    
    # Dice
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], label='train_dice')
    plt.plot(history.history['val_dice_coef'], label='val_dice')
    plt.title(f'{experiment_name} - Dice'); plt.xlabel('Epoch'); plt.ylabel('Dice Coefficient')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{experiment_name}_metrics.png')
    plt.close()


def show_slice_comparison(img, msk, pred, title, results_dir: Path):
    slice_idx = img.shape[2] // 2
    pred_bin = (pred > 0.5).astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Image
    axes[0].imshow(img[:, :, slice_idx], cmap='gray') # Removed the ', 0'
    axes[0].set_title('Image (Slice {})'.format(slice_idx))
    axes[0].axis('off')

    # Ground Truth Mask
    axes[1].imshow(msk[:, :, slice_idx], cmap='gray') # Removed the ', 0'
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Prediction Mask
    axes[2].imshow(pred_bin[:, :, slice_idx], cmap='gray') # Removed the ', 0'
    axes[2].set_title('Prediction Mask')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_dir / f'{title.replace(" ", "_")}.png')
    plt.show() 
    plt.close()

def show_augmentation_example(original_img, original_msk, aug_img, aug_msk, title, results_dir: Path):
    slice_idx = original_img.shape[2] // 2
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(original_img[:, :, slice_idx, 0], cmap='gray')
    axes[0, 0].imshow(original_msk[:, :, slice_idx, 0], alpha=0.5, cmap='Reds')
    axes[0, 0].set_title('Original (Image + GT)')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(aug_img[:, :, slice_idx, 0], cmap='gray')
    axes[0, 1].imshow(aug_msk[:, :, slice_idx, 0], alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Augmented (Image + GT)')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(original_msk[:, :, slice_idx, 0], cmap='gray')
    axes[1, 0].set_title('Original GT Mask')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(aug_msk[:, :, slice_idx, 0], cmap='gray')
    axes[1, 1].set_title('Augmented GT Mask')
    axes[1, 1].axis('off')

    fig.suptitle(f'Augmentation: {title}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_dir / f'Augmentation_Example_{title.replace(" ", "_")}.png')
    plt.close()