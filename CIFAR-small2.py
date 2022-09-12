#!/usr/bin/env python
import numpy as np
import os
import sirepo.numpy
import sirepo.sim_data.ml

import h5py


from sklearn.model_selection import train_test_split
infile = 'CIFAR-4.h5'
with h5py.File(infile, 'r') as f:
    x_values = f['images']
    y_values = f['metadata/image_types']
    print(x_values.shape, y_values.shape)
    trainx, tvx, trainy, tvy = train_test_split(x_values, y_values, test_size=0.25, random_state=42, shuffle=False)
    trainx, tvx = trainx / 255.0, tvx / 255.0
    testx, valx, testy, valy = train_test_split(tvx, tvy, test_size=0.5, random_state=42, shuffle=False)

    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
    #input_args = Input(shape=(8,))
    input_args = Input(shape=(32, 32, 3))

    x = Conv2D(32,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(input_args)
    x = BatchNormalization(momentum=0.99)(x)
    x = Conv2D(32,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         padding="same")(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = Conv2D(64,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         padding="valid")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = Conv2D(128,
                   activation="relu",
                   kernel_size=(3, 3),
                   strides=1,
                   padding="same"
                   )(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         padding="valid")(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation="softmax")(x)

    #x = Dense(1, activation="linear")(x)
    model = Model(input_args, x)

    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        trainx, trainy,
        batch_size=64,
        epochs=10,
        validation_data=(valx, valy))
