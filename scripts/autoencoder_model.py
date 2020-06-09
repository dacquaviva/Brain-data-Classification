
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Reshape
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.layers import Input, Add
from keras.optimizers import SGD, Adam
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt


def build_model(X_train, X_val=None):
    epochs = 145
    batch_size = 8
    dim_X = X_train[0].shape
    input_data = Input(shape=(dim_X))
    # Encoder-------------------------------------------
    x = Conv1D(64, kernel_size=5, activation='relu',
               padding='same')(input_data)
    x = MaxPooling1D(5)(x)
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(1, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(19, activation='relu')(x)
    # -------------------------------------------------------------------

    # latent space--------------------------------------
    encoded = Dense(10, activation='relu')(x)
    encoder = Model(input_data, encoded)
    # -----------------------------------------------------

    # Decoder-------------------------------------------------------
    x = Dense(19, activation='relu')(encoded)
    x = Reshape((19, 1))(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(5)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(5)(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = UpSampling1D(5)(x)
    decoded = Conv1D(248, 3, activation='sigmoid', padding='same')(x)
    # ---------------------------------------------------------------

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    history = autoencoder.fit(
        X_train, X_train, epochs=10, batch_size=batch_size, verbose=1)  # Autoencoder

    # summarize loss & val_loss
    plt.plot(history.history['loss'])
    try:
        plt.plot(history.history['val_loss'])
    except:
        print("No validation set provided")

    plt.title('loss')
    plt.ylabel('Loss & val_loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()
    return encoder
