# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam, Nadam, SGD
from keras.callbacks import TensorBoard, Callback
from data_gen import data_flow
from keras.layers import Dense, Input, Dropout, Activation,GlobalAveragePooling2D,LeakyReLU,BatchNormalization
from keras.layers import concatenate,Concatenate,multiply, LocallyConnected2D, Lambda,Conv2D,GlobalMaxPooling2D,Flatten
from keras.layers.core import Reshape
from keras.layers import multiply
import keras as ks
from keras.models import load_model
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from models.resnet50 import ResNet50
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from keras.utils import multi_gpu_model
from Groupnormalization import GroupNormalization
from keras_efficientnets import EfficientNetB5
from keras_efficientnets import EfficientNetB4
import efficientnet.keras as efn
# ResNet50
def model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained resnet50 model
    """
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    base_model = multi_gpu_model(base_model,4)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model

# SE-ResNet50
def model_fn(FLAGS, objective, optimizer, metrics):
    inputs_dim = Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))
    x = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=max,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)(inputs_dim)

    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=2048 // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=2048)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 2048))(excitation)


    scale = multiply([x, excitation])

    x = GlobalAveragePooling2D()(scale)
    # x = Dropout(0.3)(x)
    fc2 = Dense(FLAGS.num_classes)(x)
    fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
    model = Model(inputs=inputs_dim, outputs=fc2)
    # model.load_weights('/home/work/user-job-dir/src/SE-Xception.h5',by_name=True)
    # model = load_model('/home/work/user-job-dir/src/SE-Xception.h5')
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model

# EfficientNet
def model_fn(FLAGS, objective, optimizer, metrics):
    model = efn.EfficientNetB3(weights=None,
                               include_top=False,
                               input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                               classes=FLAGS.num_classes)
    model.load_weights('/home/work/user-job-dir/src/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')
    for i, layer in enumerate(model.layers):
        if "batch_normalization" in layer.name:
            model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)  # activation="linear",activation='softmax'
    model = Model(input=model.input, output=predictions)
    model = multi_gpu_model(model, 4)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model
# Xception
def model_fn(FLAGS, objective, optimizer, metrics):
    inputs_dim = Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))
    Xception_notop = Xception(include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                pooling=max)

    Xception_notop.load_weights('/home/work/user-job-dir/src/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    output = Xception_notop.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(FLAGS.num_classes, activation='softmax')(output)
    Xception_model = Model(inputs=Xception_notop.input, outputs=output)
    # Xception_model = multi_gpu_model(Xception_model, 4)
    Xception_model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return Xception_model

#######################################################################SE-Xception
    # Xception_notop = Xception_notop(inputs_dim)
    # squeeze =  GlobalAveragePooling2D()(Xception_notop)
    # excitation = Dense(units=2048 // 16)(squeeze)
    # excitation = Activation('relu')(excitation)
    # excitation = Dense(units=2048)(excitation)
    # excitation = Activation('sigmoid')(excitation)
    # excitation = Reshape((1, 1, 2048))(excitation)
    #
    # scale = multiply([Xception_notop, excitation])
    # x = GlobalAveragePooling2D()(scale)
    # x = Dropout(0.3)(x)
    # fc2 = Dense(FLAGS.num_classes)(x)
    # fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
    # model = Model(inputs=inputs_dim, outputs=fc2)
    # # model = multi_gpu_model(model, 4)
    # model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    # return model
