from keras.layers import *
from keras.models import *
from keras import layers
from keras import backend as K
import sys
import numpy as np
import random as python_random
np.random.seed(42)
python_random.seed(42)


def three_ppm(input):
    x = input
    shapex = K.int_shape(x)[3]
    p1 = layers.AveragePooling2D((32, 32), strides=(32, 32), name='tp1')(x)
    p2 = layers.AveragePooling2D((16, 16), strides=(16, 16), name='tp2')(x)
    p3 = layers.AveragePooling2D((8, 8), strides=(8, 8), name='tp3')(x)
    p4 = layers.AveragePooling2D((4, 4), strides=(4, 4), name='tp4')(x)

    c1 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tp5')(p1)
    c2 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tp6')(p2)
    c3 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tp7')(p3)
    c4 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tp8')(p4)

    u1 = layers.UpSampling2D((32, 32), name='tp9')(c1)
    u2 = layers.UpSampling2D((16, 16), name='tp10')(c2)
    u3 = layers.UpSampling2D((8, 8), name='tp11')(c3)
    u4 = layers.UpSampling2D((4, 4), name='tp12')(c4)

    j = layers.concatenate([input, u1, u2, u3, u4], axis=3, name='tp13')
    out = layers.Conv2D(shapex, (1, 1), strides=(1, 1), padding='same', name='tp14')(j)
    return out


def three_ppm_4v(input):
    x = input
    shapex = K.int_shape(x)[3]
    p1 = layers.AveragePooling2D((32, 32), strides=(32, 32), name='tpv1')(x)
    p2 = layers.AveragePooling2D((16, 16), strides=(16, 16), name='tpv2')(x)
    p3 = layers.AveragePooling2D((8, 8), strides=(8, 8), name='tpv3')(x)
    p4 = layers.AveragePooling2D((4, 4), strides=(4, 4), name='tpv4')(x)

    c1 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tpv5')(p1)
    c2 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tpv6')(p2)
    c3 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tpv7')(p3)
    c4 = layers.Conv2D(shapex // 4, (1, 1), activation='relu', padding='same', name='tpv8')(p4)

    u1 = layers.UpSampling2D((32, 32), name='tpv9')(c1)
    u2 = layers.UpSampling2D((16, 16), name='tpv10')(c2)
    u3 = layers.UpSampling2D((8, 8), name='tpv11')(c3)
    u4 = layers.UpSampling2D((4, 4), name='tpv12')(c4)

    j = layers.concatenate([input, u1, u2, u3, u4], axis=3, name='tpv13')
    out = layers.Conv2D(shapex, (1, 1), strides=(1, 1), padding='same', name='tpv14')(j)
    return out


def encoder_three_ppm(img_input):
    channels = [64, 128, 256, 512]

    x = layers.Conv2D(channels[0], (3, 3), activation='relu', padding='same', name='etp1')(img_input)
    x = layers.Conv2D(channels[0], (3, 3), activation='relu', padding='same', name='etp2')(x)
    feat1 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etp3')(x)

    x = layers.Conv2D(channels[1], (3, 3), activation='relu', padding='same', name='etp4')(x)
    x = layers.Conv2D(channels[1], (3, 3), activation='relu', padding='same', name='etp5')(x)
    feat2 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etp6')(x)

    x = layers.Conv2D(channels[2], (3, 3), activation='relu', padding='same', name='etp7')(x)
    x = layers.Conv2D(channels[2], (3, 3), activation='relu', padding='same', name='etp8')(x)
    feat3 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etp9')(x)

    x = layers.Conv2D(channels[3], (3, 3), activation='relu', padding='same', name='etp10')(x)
    x = layers.Conv2D(channels[3], (3, 3), activation='relu', padding='same', name='etp11')(x)
    # feat4 = x

    x = three_ppm(x)
    feat4 = x

    return feat1, feat2, feat3, feat4


def encoder_mlp(img_input):
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 5), strides=(1, 1, 1), padding='same', activation='relu', name='emu1')(img_input)
    feat1_mlp = x
    x = layers.MaxPooling3D((2, 2, 2), name='emu2')(x)

    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 5), padding='same', activation='relu', name='emu3')(x)
    feat2_mlp = x
    x = layers.MaxPooling3D((2, 2, 2), name='emu4')(x)

    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 5), padding='same', activation='relu', name='emu5')(x)
    feat3_mlp = x
    # x = layers.MaxPooling3D((2, 2, 2), name='emu6')(x)

    return feat1_mlp, feat2_mlp, feat3_mlp


def att_eunet_addcon(input_shape, nClasses=1):
    inputs = Input(input_shape)
    inputs_3D = Reshape((256, 256, 12, 1))(inputs)  # Instantiating a Keras tensor，inputs_3D=(256, 256, 12, 1)

    feat1, feat2, feat3, feat4 = encoder_three_ppm(inputs)
    feat1_mlp, feat2_mlp, feat3_mlp = encoder_mlp(inputs_3D)

    channels = [64, 128, 256, 512]

    P5_up = Conv2DTranspose(channels[2], kernel_size=2, strides=2, activation='relu', name='e1')(feat4)
    feat3_mlp = Reshape((64, 64, 3 * 128), name='e2')(feat3_mlp)
    feat3_mlp = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='e3')(feat3_mlp)
    x = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e4')(P5_up)
    g = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='e5')(feat3)
    skp = Concatenate(axis=3, name='e6')([g, feat3_mlp])
    psi = Activation('relu', name='e7')(add([skp, x]))
    psi = Conv2D(channels[2], kernel_size=1, strides=1, padding='same', name='e8')(psi)
    psi = Activation('sigmoid', name='e9')(psi)
    out = multiply([P5_up, psi], name='e10')

    P4 = Concatenate(axis=3, name='e11')([out, P5_up])
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e12')(P4)
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e13')(P4)

    P4_up = Conv2DTranspose(channels[1], kernel_size=2, strides=2, activation='relu', name='e14')(P4)
    feat2_mlp = Reshape((128, 128, 6 * 64), name='e15')(feat2_mlp)
    feat2_mlp = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='e16')(feat2_mlp)
    x = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e17')(P4_up)
    g = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='e18')(feat2)
    skp = Concatenate(axis=3, name='e19')([g, feat2_mlp])
    psi = Activation('relu', name='e20')(add([skp, x]))
    psi = Conv2D(channels[1], kernel_size=1, strides=1, padding='same', name='e21')(psi)
    psi = Activation('sigmoid', name='e22')(psi)
    out = multiply([P4_up, psi], name='e23')

    P3 = Concatenate(axis=3, name='e24')([out, P4_up])
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e25')(P3)
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e26')(P3)

    P3_up = Conv2DTranspose(channels[0], kernel_size=2, strides=2, activation='relu', name='e27')(P3)
    feat1_mlp = Reshape((256, 256, 12 * 32), name='e28')(feat1_mlp)
    feat1_mlp = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='e29')(feat1_mlp)
    x = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e30')(P3_up)
    g = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='e31')(feat1)
    skp = Concatenate(axis=3, name='e32')([g, feat1_mlp])
    psi = Activation('relu', name='e33')(add([skp, x]))
    psi = Conv2D(channels[0], kernel_size=1, strides=1, padding='same', name='e34')(psi)
    psi = Activation('sigmoid', name='e35')(psi)
    out = multiply([P3_up, psi], name='e36')

    P2 = Concatenate(axis=3, name='e37')([out, P3_up])
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e38')(P2)
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e39')(P2)  # 74

    o = Conv2D(nClasses, 1, padding='same', name='e40')(P2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]
    out = (Reshape((outputHeight * outputWidth, nClasses), name='e41'))(o)
    out = Activation('sigmoid', name='e42')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    model.summary()
    return model


def clone_atteunet(inputs, inputs_3D, nClasses=1):
    # inputs = Input(input_shape)
    # inputs_3D = Reshape((256, 256, 12, 1))(inputs)

    feat1, feat2, feat3, feat4 = encoder_three_ppm(inputs)
    feat1_mlp, feat2_mlp, feat3_mlp = encoder_mlp(inputs_3D)

    channels = [64, 128, 256, 512]

    P5_up = Conv2DTranspose(channels[2], kernel_size=2, strides=2, activation='relu', name='e1')(feat4)
    feat3_mlp = Reshape((64, 64, 3 * 128), name='e2')(feat3_mlp)
    feat3_mlp = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e3')(feat3_mlp)
    x = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e4')(P5_up)
    g = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e5')(feat3)
    psi = Activation('relu', name='e6')(add([g, feat3_mlp]))
    psi = Activation('relu', name='e7')(add([psi, x]))
    psi = Conv2D(channels[2], kernel_size=1, strides=1, padding='same', name='e8')(psi)
    psi = Activation('sigmoid', name='e9')(psi)
    out = multiply([P5_up, psi], name='e10')

    P4 = Concatenate(axis=3, name='e11')([out, P5_up])
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e12')(P4)
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e13')(P4)

    P4_up = Conv2DTranspose(channels[1], kernel_size=2, strides=2, activation='relu', name='e14')(P4)
    feat2_mlp = Reshape((128, 128, 6 * 64), name='e15')(feat2_mlp)
    feat2_mlp = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e16')(feat2_mlp)
    x = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e17')(P4_up)
    g = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e18')(feat2)
    psi = Activation('relu', name='e19')(add([g, feat2_mlp]))
    psi = Activation('relu', name='e20')(add([psi, x]))
    psi = Conv2D(channels[1], kernel_size=1, strides=1, padding='same', name='e21')(psi)
    psi = Activation('sigmoid', name='e22')(psi)
    out = multiply([P4_up, psi], name='e23')

    P3 = Concatenate(axis=3, name='e24')([out, P4_up])
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e25')(P3)
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e26')(P3)

    P3_up = Conv2DTranspose(channels[0], kernel_size=2, strides=2, activation='relu', name='e27')(P3)
    feat1_mlp = Reshape((256, 256, 12 * 32), name='e28')(feat1_mlp)
    feat1_mlp = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e29')(feat1_mlp)
    x = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e30')(P3_up)
    g = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e31')(feat1)
    psi = Activation('relu', name='e32')(add([g, feat1_mlp]))
    psi = Activation('relu', name='e33')(add([psi, x]))
    psi = Conv2D(channels[0], kernel_size=1, strides=1, padding='same', name='e34')(psi)
    psi = Activation('sigmoid', name='e35')(psi)
    out = multiply([P3_up, psi], name='e36')

    P2 = Concatenate(axis=3, name='e37')([out, P3_up])
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e38')(P2)
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e39')(P2)

    o = Conv2D(nClasses, 1, padding='same', name='e40')(P2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]
    out = (Reshape((outputHeight * outputWidth, nClasses), name='e41'))(o)
    out = Activation('sigmoid', name='output_1')(out)

    # model = Model(input=inputs, output=out)
    # model.outputHeight = outputHeight
    # model.outputWidth = outputWidth

    # model.summary()
    return out, outputHeight, outputWidth


def clone_atteunet4pred(input_shape, nClasses=1):
    inputs = Input(input_shape)
    inputs_3D = Reshape((256, 256, 12, 1))(inputs)  # Instantiating a Keras tensor，inputs_3D=(256, 256, 12, 1)

    feat1, feat2, feat3, feat4 = encoder_three_ppm(inputs)
    feat1_mlp, feat2_mlp, feat3_mlp = encoder_mlp(inputs_3D)

    channels = [64, 128, 256, 512]

    P5_up = Conv2DTranspose(channels[2], kernel_size=2, strides=2, activation='relu', name='e1')(feat4)
    feat3_mlp = Reshape((64, 64, 3 * 128), name='e2')(feat3_mlp)
    feat3_mlp = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e3')(feat3_mlp)
    x = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e4')(P5_up)
    g = Conv2D(channels[2], (1, 1), strides=(1, 1), padding='same', name='e5')(feat3)
    psi = Activation('relu', name='e6')(add([g, feat3_mlp]))
    psi = Activation('relu', name='e7')(add([psi, x]))
    psi = Conv2D(channels[2], kernel_size=1, strides=1, padding='same', name='e8')(psi)
    psi = Activation('sigmoid', name='e9')(psi)
    out = multiply([P5_up, psi], name='e10')

    P4 = Concatenate(axis=3, name='e11')([out, P5_up])
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e12')(P4)
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='e13')(P4)

    P4_up = Conv2DTranspose(channels[1], kernel_size=2, strides=2, activation='relu', name='e14')(P4)
    feat2_mlp = Reshape((128, 128, 6 * 64), name='e15')(feat2_mlp)
    feat2_mlp = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e16')(feat2_mlp)
    x = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e17')(P4_up)
    g = Conv2D(channels[1], (1, 1), strides=(1, 1), padding='same', name='e18')(feat2)
    psi = Activation('relu', name='e19')(add([g, feat2_mlp]))
    psi = Activation('relu', name='e20')(add([psi, x]))
    psi = Conv2D(channels[1], kernel_size=1, strides=1, padding='same', name='e21')(psi)
    psi = Activation('sigmoid', name='e22')(psi)
    out = multiply([P4_up, psi], name='e23')

    P3 = Concatenate(axis=3, name='e24')([out, P4_up])
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e25')(P3)
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='e26')(P3)

    P3_up = Conv2DTranspose(channels[0], kernel_size=2, strides=2, activation='relu', name='e27')(P3)
    feat1_mlp = Reshape((256, 256, 12 * 32), name='e28')(feat1_mlp)
    feat1_mlp = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e29')(feat1_mlp)
    x = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e30')(P3_up)
    g = Conv2D(channels[0], (1, 1), strides=(1, 1), padding='same', name='e31')(feat1)
    psi = Activation('relu', name='e32')(add([g, feat1_mlp]))
    psi = Activation('relu', name='e33')(add([psi, x]))
    psi = Conv2D(channels[0], kernel_size=1, strides=1, padding='same', name='e34')(psi)
    psi = Activation('sigmoid', name='e35')(psi)
    out = multiply([P3_up, psi], name='e36')

    P2 = Concatenate(axis=3, name='e37')([out, P3_up])
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e38')(P2)
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='e39')(P2)

    o = Conv2D(nClasses, 1, padding='same', name='e40')(P2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]
    out = (Reshape((outputHeight * outputWidth, nClasses), name='e41'))(o)
    out = Activation('sigmoid', name='output_1')(out)

    model = Model(inputs, out)
    # model.outputHeight = outputHeight
    # model.outputWidth = outputWidth

    # model.summary()
    return model


def encoder_three_ppm_4v2c(img_input):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='etpv2c1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='etpv2c2')(x)
    feat1 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etpv2c3')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='etpv2c4')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='etpv2c5')(x)
    feat2 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etpv2c6')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='etpv2c7')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='etpv2c8')(x)
    feat3 = x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='etpv2c9')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='etpv2c10')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='etpv2c11')(x)
    # feat4 = x

    x = three_ppm_4v(x)
    feat4 = x

    return feat1, feat2, feat3, feat4


def v2c_sANDv(input_shape, nClasses=1):
    inputs = Input(input_shape)
    inputs_3D = Reshape((256, 256, 12, 1))(inputs)  # Instantiating a Keras tensor，inputs_3D=(256, 256, 12, 1)

    P1, outputHeight, outputWidth = clone_atteunet(inputs, inputs_3D)
    P2 = Reshape((outputHeight, outputWidth, nClasses), name='v2c_sANDv1')(P1)

    feat1, feat2, feat3, feat4 = encoder_three_ppm_4v2c(P2)

    channels = [64, 128, 256, 512]  # v2c

    P5_up = Conv2DTranspose(channels[2], kernel_size=2, strides=2, activation='relu', name='v2c1')(feat4)
    P4 = Concatenate(axis=3, name='v2c2')([feat3, P5_up])
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='v2c3')(P4)
    P4 = Conv2D(channels[2], 3, activation='relu', padding='same', name='v2c4')(P4)

    P4_up = Conv2DTranspose(channels[1], kernel_size=2, strides=2, activation='relu', name='v2c5')(P4)
    P3 = Concatenate(axis=3, name='v2c6')([feat2, P4_up])
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='v2c7')(P3)
    P3 = Conv2D(channels[1], 3, activation='relu', padding='same', name='v2c8')(P3)

    P3_up = Conv2DTranspose(channels[0], kernel_size=2, strides=2, activation='relu', name='v2c9')(P3)
    P2 = Concatenate(axis=3, name='v2c10')([feat1, P3_up])
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='v2c11')(P2)
    P2 = Conv2D(channels[0], 3, activation='relu', padding='same', name='v2c12')(P2)

    o = Conv2D(nClasses, 1, padding='same', name='v2c13')(P2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]
    out = (Reshape((outputHeight * outputWidth, nClasses), name='v2c14'))(o)
    out = Activation('sigmoid', name='output_2')(out)

    model = Model(inputs, out)

    # model.summary()
    return model
