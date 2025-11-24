import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import random
from config import TARGET_SHAPE, BATCH_SIZE, SEED
from augmentation import augment_intensity, elastic_deformation


def load_nifti(path):
    img = sitk.ReadImage(path)
    return img

def resample_sitk(img: sitk.Image, target_shape=TARGET_SHAPE, is_mask=False):
    original_size = np.array(img.GetSize())
    original_spacing = np.array(img.GetSpacing())
    new_size = target_shape
    new_spacing = original_spacing * (original_size / new_size)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(img)
    arr = sitk.GetArrayFromImage(resampled)
    arr = np.transpose(arr, (1, 2, 0))
    return arr

def normalize_volume(vol):
    v = vol.copy()
    mask = v != 0
    if np.any(mask):
        m = v[mask]
        mu = m.mean()
        sd = m.std() if m.std() > 0 else 1.0
        v[mask] = (v[mask] - mu) / sd
    else:
        v = (v - v.mean()) / (v.std() + 1e-8)
    return np.clip(v, -6, 6)

def preprocess_pair(img_path, msk_path, target_shape=TARGET_SHAPE):
    img_sitk = load_nifti(img_path)
    msk_sitk = load_nifti(msk_path)
    img = resample_sitk(img_sitk, target_shape, is_mask=False)
    msk = resample_sitk(msk_sitk, target_shape, is_mask=True)
    img = normalize_volume(img)
    msk = (msk > 0.5).astype(np.float32)
    img = img[..., np.newaxis]
    msk = msk[..., np.newaxis]
    return img.astype(np.float32), msk.astype(np.float32)

def tf_preprocess_wrapper(img_path, msk_path):
    img, msk = tf.numpy_function(
        lambda a,b: preprocess_pair(a.decode(), b.decode()), 
        [img_path, msk_path], 
        [tf.float32, tf.float32]
    )
    img.set_shape(TARGET_SHAPE + (1,))
    msk.set_shape(TARGET_SHAPE + (1,))
    return img, msk

def build_dataset(pairs, batch_size=BATCH_SIZE, shuffle=True, augment_fn=None):
    if not pairs:
        print("Warning: Dataset list is empty.")
        return tf.data.Dataset.from_tensors((tf.zeros(TARGET_SHAPE + (1,), dtype=tf.float32), tf.zeros(TARGET_SHAPE + (1,), dtype=tf.float32))).batch(batch_size).cache()

    img_paths = [p for p, m in pairs]
    msk_paths = [m for p, m in pairs]
    
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    
    if shuffle:
        ds = ds.shuffle(len(img_paths), seed=SEED)
        
    ds = ds.map(tf_preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_fn is not None:
        ds = ds.map(
            lambda x, y: tf.numpy_function(
                lambda a, b: augment_fn(a, b), [x, y], [tf.float32, tf.float32]
            ), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(
            lambda x, y: (tf.reshape(x, TARGET_SHAPE + (1,)), tf.reshape(y, TARGET_SHAPE + (1,)))
        )
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds