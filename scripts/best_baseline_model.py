
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt


def build_model(X_train, y_train, X_val=None, y_val=None):
    dim_X = X_train[0].shape
    conv_1_size = 116
    conv_2_size = 150
    kernel_size = int(dim_X[0] * 0.05)
    strides = int(kernel_size/2)
    layer_1_size = 44
    dropout = 0.2388
    layer_2_size = 68
    optimizer = 'adam'
    learning_rate = 0.0004297
    weight_initializers = "random_normal"
    epochs = 145
    batch_size = 8

    model = Sequential()
    model.add(Conv1D(conv_1_size, kernel_size=kernel_size, strides=strides, padding='same',
                     input_shape=dim_X, kernel_initializer=weight_initializers, data_format='channels_last'))

    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers, strides=strides,
                     padding='same', activation='relu'))

    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(layer_1_size,
                    kernel_initializer=weight_initializers))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Dense(layer_2_size,
                    kernel_initializer=weight_initializers, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, kernel_initializer=weight_initializers, activation='softmax'))

    if(optimizer == "adam"):

        opt = Adam(learning_rate=learning_rate)

    else:
        opt = SGD(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    # summarize loss & val_loss
    plt.plot(history.history['loss'])
    try:
        plt.plot(history.history['val_loss'])
    except:
        print("No validation set provided")

    plt.title('loss and accuracy model')
    plt.ylabel('Loss & val_loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

    # summarize acc & val_acc
    plt.plot(history.history['accuracy'])
    try:
        plt.plot(history.history['val_accuracy'])
    except:
        print("No validation set provided")
    plt.title('accuracy and val_accuracy model')
    plt.ylabel('accuracy & val_accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()
    return model
