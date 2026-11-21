import numpy as np
import pywt
from pyts.image import GramianAngularField
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from Evaluation import evaluation


def compute_wpd(signal, wavelet='db4', maxlevel=3):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    coeffs = [wp[node].data for node in nodes]
    return np.stack(coeffs, axis=0)

def compute_gaf(signal, image_size=64, method='summation'):
    signal = signal.reshape(1, -1)
    gaf = GramianAngularField(image_size=image_size, method=method)
    return gaf.fit_transform(signal)[0]

def preprocess_signal(signal, image_size=64):
    subbands = compute_wpd(signal)
    gaf_images = [compute_gaf(sb, image_size=image_size) for sb in subbands]
    stacked = np.stack(gaf_images, axis=-1)
    return stacked.astype(np.float32)

def cross_attention_block(x):
    in_channels = x.shape[-1]

    q = layers.Conv2D(in_channels, 1)(x)
    k = layers.Conv2D(in_channels, 1)(x)
    v = layers.Conv2D(in_channels, 1)(x)
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    N = H * W

    q_flat = tf.reshape(q, (-1, N, in_channels))
    k_flat = tf.reshape(k, (-1, N, in_channels))
    v_flat = tf.reshape(v, (-1, N, in_channels))

    attn = tf.nn.softmax(tf.matmul(q_flat, k_flat, transpose_b=True), axis=-1)
    out = tf.matmul(attn, v_flat)

    out = tf.reshape(out, (-1, H, W, in_channels))
    out = layers.Conv2D(in_channels, 1)(out)
    return layers.Add()([x, out])

def residual_attention_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = cross_attention_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def Model_GAFWPDARCAN(x_train, y_train, x_test, y_test, steps_per_epoch, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    x_train_gaf = np.array([preprocess_signal(sig, image_size=64) for sig in x_train])
    x_test_gaf  = np.array([preprocess_signal(sig, image_size=64) for sig in x_test])

    if x_train_gaf.shape[-1] != 3:
        x_train_gaf = x_train_gaf[..., :3]
        x_test_gaf  = x_test_gaf[..., :3]
    inputs = layers.Input(shape=x_train_gaf.shape[1:])

    x = layers.Conv2D(32, 7, padding='same', activation='relu')(inputs)
    x = residual_attention_block(x, 32)
    x = residual_attention_block(x, 64)
    x = residual_attention_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)

    hidden_units = int(sol[0])
    x = layers.Dense(hidden_units, activation='relu')(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train_gaf, y_train,
        batch_size=64,
        epochs=int(sol[2]),
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test_gaf, y_test)
    )
    predict = model.predict(x_test_gaf)
    Eval = evaluation(y_test, predict)

    return Eval, predict

