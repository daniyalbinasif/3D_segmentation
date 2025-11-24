import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter


def augment_intensity(img, msk, max_rot=10, max_shift=10, flip_prob=0.5, intensity_jitter=0.1):
    """
    Applies random flip, 3D affine transformation, and intensity jitter.
    """
    img = img.copy()[..., 0]
    msk = msk.copy()[..., 0]
    msk = (msk > 0.5).astype(np.float32)

    # Random Flip
    if random.random() < flip_prob:
        axis = random.choice([0, 1, 2])
        img = np.flip(img, axis=axis)
        msk = np.flip(msk, axis=axis)

    # Rotation and Shift
    angles = np.deg2rad(np.random.uniform(-max_rot, max_rot, size=3))
    Rx = np.array([[1,0,0],[0,np.cos(angles[0]), -np.sin(angles[0])],[0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],[0,1,0],[-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]),0],[np.sin(angles[2]), np.cos(angles[2]),0],[0,0,1]])
    R = Rx @ Ry @ Rz
    center = np.array(img.shape) / 2.0
    shift = np.random.uniform(-max_shift, max_shift, size=3)
    matrix = R
    offset = center - matrix @ center + shift

    img_t = ndimage.affine_transform(img, matrix, offset=offset, order=1, mode='constant', cval=0.0) # Image transformation
    msk_t = ndimage.affine_transform(msk, matrix, offset=offset, order=0, mode='constant', cval=0.0) # Mask transformation

    # Intensity Jitter
    img_t = img_t + np.random.normal(0, intensity_jitter, size=img_t.shape)

    # Clip
    img_t = np.clip(img_t, -6, 6)

    return img_t[..., np.newaxis].astype(np.float32), (msk_t > 0.5).astype(np.float32)[..., np.newaxis]


def elastic_deformation(img: np.ndarray, msk: np.ndarray, alpha=15, sigma=3):
    img = img[..., 0]
    msk = msk[..., 0]
    shape = img.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.array([x + dx, y + dy, z + dz])
    img_aug = map_coordinates(img, indices, order=1, mode='nearest')
    msk_aug = map_coordinates(msk, indices, order=0, mode='nearest')
    return img_aug[..., np.newaxis], (msk_aug > 0.5)[..., np.newaxis].astype(np.float32)