import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import TARGET_SHAPE

def conv_block(x, filters, kernel_size=3):
    x = layers.Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    return x

def build_unet_3d(input_shape=TARGET_SHAPE + (1,), base_filters=16):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPool3D((2,2,2))(c1)
    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPool3D((2,2,2))(c2)
    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPool3D((2,2,2))(c3)
    c4 = conv_block(p3, base_filters*8)
    p4 = layers.MaxPool3D((2,2,2))(c4)
    
    # Bottleneck
    bn = conv_block(p4, base_filters*16)
    
    # Decoder
    u4 = layers.Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(bn)
    u4 = layers.concatenate([u4, c4])
    c5 = conv_block(u4, base_filters*8)
    u3 = layers.Conv3DTranspose(base_filters*4, 2, strides=2, padding='same')(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = conv_block(u3, base_filters*4)
    u2 = layers.Conv3DTranspose(base_filters*2, 2, strides=2, padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = conv_block(u2, base_filters*2)
    u1 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = conv_block(u1, base_filters)
    
    # Output layer
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(c8)
    return keras.Model(inputs, outputs, name="3D_Unet")

# Loss Functions

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1.0 - dice_coef(y_true, y_pred))