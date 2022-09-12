#!/usr/bin/env python
import numpy as np
import os
import sirepo.numpy
import sirepo.sim_data.ml

import h5py

# from sklearn.preprocessing import RobustScaler

# def read_data(data_reader, data_path, **kwargs):
#     return sirepo.numpy.ndarray_from_ctx(
#         data_reader.data_context_manager(data_path),
#         1,
#         **kwargs,
#     )


# def read_data_and_encode_output_column(data_reader, data_path, column_types):
#     from sklearn.preprocessing import LabelEncoder
#     _DATA_TYPE = np.float

#     def save_encoding_file(encoder):
#         from pykern import pkjson
#         from pykern.pkcollections import PKDict
#         pkjson.dump_pretty(
#             PKDict(
#                 zip(
#                     encoder.transform(encoder.classes_).astype(_DATA_TYPE).tolist(),
#                     encoder.classes_,
#                 ),
#             ),
#             filename='classification-output-col-encoding.json',
#         )

#     v = read_data(data_reader, data_path, dtype=None, encoding='utf=8')
#     if len(v.dtype.descr) > 1:
#         descr = v.dtype.descr
#         output_encoding = None
#         for idx in range(len(descr)):
#             ft = descr[idx]
#             if np.dtype(ft[1]).kind == 'U':
#                 if column_types[idx] == 'none':
#                     v[ft[0]] = 0
#                     descr[idx] = (ft[0], np.float)
#                 else:
#                     encoder = LabelEncoder().fit(v[ft[0]])
#                     v[ft[0]] = encoder.transform(v[ft[0]])
#                     descr[idx] = (ft[0], _DATA_TYPE)
#                     if output_encoding is None and column_types[idx] == 'output':
#                         output_encoding = encoder
#         v = np.array(v.astype(descr).tolist())
#         if output_encoding:
#             save_encoding_file(output_encoding)
#     return v

# def scale_columns(values, column_types, col_type, scaler):
#     columns = list(filter(lambda idx: column_types[idx] == col_type, range(len(column_types))))
#     if scaler and len(columns):
#         values[:, columns] = scaler().fit_transform(values[:, columns])
#     return columns


# def scale_file(data_reader, data_path, column_types, inputs_scaler, outputs_scaler):
#     v = read_data_and_encode_output_column(data_reader, data_path, column_types)
#     in_idx = scale_columns(v, column_types, 'input', inputs_scaler)
#     out_idx = scale_columns(v, column_types, 'output', outputs_scaler)
#     os.remove(data_reader.path)
#     np.save('scaled.npy', v)
#     return v, in_idx, out_idx


# scaled, in_idx, out_idx = scale_file(
#     sirepo.sim_data.ml.DataReader('dataFile-file.2019Happiness.data.csv'),
#     'None',
#     ['output','input','input','input','input','input','input','input','input'],
#     RobustScaler,
#     RobustScaler,
# )


from sklearn.model_selection import train_test_split
infile = 'CIFAR-4.h5'
with h5py.File(infile, 'r') as f:
    # idx = 4
    # img = PIL.Image.fromarray(f['images'][idx])
    # print('it is a {}: {}'.format(
    #     f['metadata/image_types'][idx],
    #     labels[f['metadata/image_types'][idx]],
    # ))
    # img.save('test2.png')
    x_values = f['images']
    y_values = f['metadata/image_types']
    print(x_values.shape, y_values.shape)

    # train, tv = train_test_split(x_values, test_size=0.75, random_state=42, shuffle=False)
    # test, validate = train_test_split(tv, test_size=0.5, random_state=42, shuffle=False)

    trainx, tvx, trainy, tvy = train_test_split(x_values, y_values, test_size=0.25, random_state=42, shuffle=False)
    trainx, tvx = trainx / 255.0, tvx / 255.0
    testx, valx, testy, valy = train_test_split(tvx, tvy, test_size=0.5, random_state=42, shuffle=False)



    # from sklearn.model_selection import train_test_split

    # def train_test_validate(values, train_size, test_size):
    #     size = (100 - train_size) / 100.
    #     train, tv = train_test_split(values, test_size=size, random_state=42, shuffle=True)
    #     size = test_size / (test_size + (100 - train_size - test_size))
    #     test, validate = train_test_split(tv, test_size=size, random_state=42, shuffle=True)
    #     return (train, test, validate)

    # train, test, validate = train_test_validate(
    #     scaled,
    #     75,
    #     12.5,
    # )


    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
    #input_args = Input(shape=(8,))
    input_args = Input(shape=(32, 32, 3))

    x = Conv2D(32,
        activation="relu",
        kernel_size=(3, 3),
        strides=1,
        padding="valid"
        )(input_args)
    x = MaxPooling2D(pool_size=(2, 2),
        strides=None,
        padding="valid")(x)
    x = Conv2D(64,
        activation="relu",
        kernel_size=(3, 3),
        strides=1,
        padding="valid"
        )(x)
    x = MaxPooling2D(pool_size=(2, 2),
        strides=None,
        padding="valid")(x)
    x = Conv2D(64,
        activation="relu",
        kernel_size=(3, 3),
        strides=1,
        padding="valid"
        )(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)

    #x = Dense(1, activation="linear")(x)
    #x = Dense(4, activation="relu")(x)
    x = Dense(4, activation="linear")(x)

    model = Model(input_args, x)

    model.summary()

    # from keras.callbacks import CSVLogger

    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # model.fit(
    #     x=train[:, in_idx],
    #     y=train[:, out_idx],
    #     validation_data=(validate[:, in_idx], validate[:, out_idx]),
    #     batch_size=50,
    #     shuffle=True,
    #     epochs=500,
    #     verbose=False,
    #     callbacks=[CSVLogger('fit.csv')],
    # )
    # np.save('test.npy', test[:, out_idx])
    # np.save('predict.npy', model.predict(x=test[:, in_idx]))


    import keras.losses

    model.compile(
        optimizer='adam',
        #loss='sparse_categorical_crossentropy',
        #loss='mean_squared_error',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        trainx, trainy, epochs=10,
        validation_data=(valx, valy))


# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 32, 32, 3)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 30, 30, 32)        896
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
# _________________________________________________________________
# flatten (Flatten)            (None, 1024)              0
# _________________________________________________________________
# dense (Dense)                (None, 64)                65600
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 260
# =================================================================
# Total params: 122,180
# Trainable params: 122,180
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/10
# 563/563 [==============================] - 5s 9ms/step - loss: 1.0858 - accuracy: 0.5384 - val_loss: 1.0689 - val_accuracy: 0.5373
# Epoch 2/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.9108 - accuracy: 0.6668 - val_loss: 0.8373 - val_accuracy: 0.7087
# Epoch 3/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.7053 - accuracy: 0.7271 - val_loss: 0.6679 - val_accuracy: 0.7343
# Epoch 4/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.5827 - accuracy: 0.7716 - val_loss: 0.5869 - val_accuracy: 0.7737
# Epoch 5/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.5222 - accuracy: 0.7979 - val_loss: 0.5685 - val_accuracy: 0.7843
# Epoch 6/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.4757 - accuracy: 0.8187 - val_loss: 0.5308 - val_accuracy: 0.7980
# Epoch 7/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.4346 - accuracy: 0.8352 - val_loss: 0.5576 - val_accuracy: 0.7810
# Epoch 8/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.3902 - accuracy: 0.8523 - val_loss: 0.5391 - val_accuracy: 0.8003
# Epoch 9/10
# 563/563 [==============================] - 5s 8ms/step - loss: 0.3570 - accuracy: 0.8649 - val_loss: 0.5451 - val_accuracy: 0.8020
# Epoch 10/10
# 563/563 [==============================] - 5s 9ms/step - loss: 0.3241 - accuracy: 0.8772 - val_loss: 0.5374 - val_accuracy: 0.8090
