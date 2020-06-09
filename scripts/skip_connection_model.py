
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.layers import Input, Add
from keras.optimizers import SGD, Adam
import numpy as np
from keras.models import Model
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
    input_data = Input(dim_X)
    x = Conv1D(conv_1_size, kernel_size=kernel_size, padding='same',
               strides=strides, kernel_initializer=weight_initializers)(input_data)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(conv_2_size, kernel_size=kernel_size, padding='same',
               strides=strides, kernel_initializer=weight_initializers)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    # ---------------------------------------------
    x = Dense(layer_1_size, kernel_initializer=weight_initializers)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    X_shortcut = x
    # ---------------------------------------------

    # ---------------------------------------------
    x = Dense(layer_2_size, kernel_initializer=weight_initializers)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    # ---------------------------------------------------------
    # # ---------------------------------------------------------
    x = Dense(layer_1_size, kernel_initializer=weight_initializers)(x)
    x = Add()([x, X_shortcut])  # ------------------> Skip connection
    x = Activation('relu')(x)
    # ---------------------------------------------------------

    # ---------------------------------------------
    x = Dense(layer_2_size, kernel_initializer=weight_initializers)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    X_shortcut = x
    # ---------------------------------------------------------
    # # ---------------------------------------------------------
    x = Dense(50, kernel_initializer=weight_initializers)(x)
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    # ---------------------------------------------------------
    x = Dense(layer_2_size, kernel_initializer=weight_initializers)(x)

    x = Add()([x, X_shortcut])  # ------------------> Skip connection
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    output = Dense(4, kernel_initializer=weight_initializers,
                   activation='softmax')(x)
    model = Model(input_data, output)

    if(optimizer == "adam"):

        opt = Adam(learning_rate=learning_rate)

    else:
        opt = SGD(lr=learning_rate, nesterov=True)

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
