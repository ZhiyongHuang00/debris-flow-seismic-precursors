# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/17 15:40
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : Model_AE.py
@Function: {}:
'''
# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/17 13:27
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : make_paz.py
@Function: CAE with Layer Normalization
'''

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LayerNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import set_random_seed

def set_global_determinism(seed=42):
    import tensorflow as tf
    import numpy as np
    import random
    import os

    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.config.experimental.enable_op_determinism()

def CAE(input_shape=(64, 512, 1), filters=[8, 16, 32, 64, 128], encoding_filters=32, set_seed=[False, 87], summary=True):
    if set_seed[0] is True:
        set_global_determinism(seed=set_seed[1])
        set_random_seed(set_seed[1])

    inp = Input(shape=input_shape, name="original_img")

    # Encoder
    x = Conv2D(filters=filters[0], kernel_size=3, strides=2, padding="same", name="Conv_1")(inp)
    x = LayerNormalization(name="LN_1")(x)
    x = Activation("relu", name="Act_1")(x)

    x = Conv2D(filters=filters[1], kernel_size=3, strides=2, padding="same", name="Conv_2")(x)
    x = LayerNormalization(name="LN_2")(x)
    x = Activation("relu", name="Act_2")(x)

    # 注释的层如果需要可以打开
    x = Conv2D(filters=filters[2], kernel_size=3, strides=2, padding="same", name="Conv_3")(x)
    x = LayerNormalization(name="LN_3")(x)
    x = Activation("relu", name="Act_3")(x)

    x = Conv2D(filters=filters[3], kernel_size=3, strides=2, padding="same", name="Conv_4")(x)
    x = LayerNormalization(name="LN_4")(x)
    x = Activation("relu", name="Act_4")(x)

    x = Conv2D(filters=encoding_filters, kernel_size=3, strides=2, padding="same", name="Conv_5")(x)
    x = LayerNormalization(name="LN_5")(x)
    x = Activation("relu", name="Act_5")(x)

    shape_before_flattening = K.int_shape(x)
    x = Flatten(name="Flatten")(x)
    encoder_output = Dense(encoding_filters, activation="linear", name="encoded")(x)

    encoder = Model(inp, encoder_output, name="encoder")

    # Decoder
    decoder_input = encoder_output
    x = Dense(shape_before_flattening[1] * shape_before_flattening[2] * shape_before_flattening[3], activation="relu")(decoder_input)
    x = Reshape((shape_before_flattening[1], shape_before_flattening[2], shape_before_flattening[3]))(x)

    x = Conv2DTranspose(filters=encoding_filters, kernel_size=3, strides=2, padding="same")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=filters[3], kernel_size=3, strides=2, padding="same")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=filters[2], kernel_size=3, strides=2, padding="same")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=filters[1], kernel_size=3, strides=2, padding="same")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=filters[0], kernel_size=3, strides=2, padding="same")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    decoder_output = Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=1, padding="same", activation="linear", name="decoded")(x)

    autoencoder = Model(inputs=inp, outputs=decoder_output, name="autoencoder")
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

    if summary:
        autoencoder.summary()

    return encoder, decoder, autoencoder
# encoder, decoder, autoencoder = CAE()
