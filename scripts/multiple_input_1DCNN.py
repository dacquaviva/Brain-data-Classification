
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from keras.models import Model


def build_model(X_train, y_train, X_val=None, y_val=None):

    n_input = 4
    dim_X = (X_train[0].shape[0], int(X_train[0].shape[1]/n_input))

    # conv configiration
    conv_1_size = 116
    conv_2_size = 50

    # First head configuration
    kernel_size = int(dim_X[0] * 0.05)
    strides = int(kernel_size/2)

    # dense layers configuration
    layer_1_size = 100
    dropout = 0.2388
    layer_2_size = 30
    optimizer = 'adam'
    learning_rate = 0.0004297
    weight_initializers = "random_normal"
    epochs = 145
    batch_size = 32

    # input 1 -----------------------------------------------
    inputs1 = Input(shape=(dim_X))
    x1 = Conv1D(conv_1_size, kernel_size=kernel_size, strides=strides, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers, strides=strides,
                padding='same', activation='relu')(x1)
    x1 = Dropout(dropout)(x1)
    flat1 = Flatten()(x1)  # ---> flat input 1
    # --------------------------------------------------------------

    # input 2----------------------------------------------------------
    inputs2 = Input(shape=(dim_X))
    x2 = Conv1D(conv_1_size, kernel_size=kernel_size, strides=strides, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs2)
    x2 = Dropout(dropout)(x2)
    x2 = Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers, strides=strides,
                padding='same', activation='relu')(x2)

    x2 = Dropout(dropout)(x2)
    flat2 = Flatten()(x2)  # ---> flat input 2
    # --------------------------------------------------------------------

    # input 3-------------------------------------------------------------
    inputs3 = Input(shape=(dim_X))
    x3 = Conv1D(conv_1_size, kernel_size=kernel_size, strides=strides, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs3)
    x3 = Dropout(dropout)(x3)
    x3 = Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers, strides=strides,
                padding='same', activation='relu')(x3)

    x3 = Dropout(dropout)(x3)
    flat3 = Flatten()(x3)  # ---> flat input 3
    # --------------------------------------------------------------------

    # input 4------------------------------------------------------------
    inputs4 = Input(shape=(dim_X))
    x4 = Conv1D(conv_1_size, kernel_size=kernel_size, strides=strides, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs4)
    x4 = Dropout(dropout)(x4)
    x4 = Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers, strides=strides,
                padding='same', activation='relu')(x4)

    x4 = Dropout(dropout)(x4)
    flat4 = Flatten()(x4)  # ---> flat input 4
    # --------------------------------------------------------------------

    # merge
    merged = concatenate([flat1, flat2, flat3, flat4])

    d = Dense(layer_1_size, kernel_initializer=weight_initializers,
              activation='relu')(merged)
    d = Dropout(dropout)(d)
    d = Dense(layer_2_size,
              kernel_initializer=weight_initializers, activation='relu')(d)
    d = Dropout(dropout)(d)
    outputs = Dense(4, kernel_initializer=weight_initializers,
                    activation='softmax')(d)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    dataset_path = "gdrive/My Drive/colab/DEEP LEARNING/data/"
    plot_model(model, show_shapes=True,
               to_file=dataset_path + 'Multiple_Input.png')
    if(optimizer == "adam"):

        opt = Adam(learning_rate=learning_rate)

    else:
        opt = SGD(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=["accuracy"])

    history = model.fit([X_train, X_train], y_train, validation_data=(
        [X_val, X_val], y_val), epochs=epochs, batch_size=batch_size, verbose=1)
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
