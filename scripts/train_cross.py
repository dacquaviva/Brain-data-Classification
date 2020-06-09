import argparse
import os
import preprocess
import json
import numpy as np
import math
from datagenerator import DataGenerator
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD, Adam
import wandb
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
# default config/hyperparameter values
# you can modify these below or via command line
PROJECT_NAME = "Deep Learning Assignment"
MODEL_NAME = "multiple_input_1DCNN"

C1_SIZE = 116
C2_SIZE = 50
L1_SIZE = 100
DROPOUT_CONV = 0.2388
DROPOUT_DENSE = 0.2388
L2_SIZE = 30

BATCH_SIZE = 32
EPOCHS = 145

OPTIMIZER = "adam"
LEARNING_RATE = 0.0004297
WEIGHTS_INITIALIZER = "random_normal"
SAMPLING_RATE = 100


def build_model(X_train, y_train, X_val, y_val, conv_1_size, conv_2_size, kernel_size_head_1,  strides_head_1, kernel_size_head_2, strides_head_2, layer_1_size, dropout_conv, dropout_dense, layer_2_size, optimizer, learning_rate, weight_initializers):
    """ model """
    dim_X = X_train[0].shape

    # head 1
    inputs1 = Input(shape=(dim_X))
    x1 = Conv1D(conv_1_size, kernel_size=kernel_size_head_1, strides=strides_head_1, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs1)
    x1 = Dropout(dropout_conv)(x1)
    x1 = Conv1D(conv_2_size, kernel_size=kernel_size_head_1, kernel_initializer=weight_initializers, strides=strides_head_1,
                padding='same', activation='relu')(x1)
    x1 = Dropout(dropout_conv)(x1)
    flat1 = Flatten()(x1)  # ---> flat head 1

    # head 2
    inputs2 = Input(shape=(dim_X))
    x2 = Conv1D(conv_1_size, kernel_size=kernel_size_head_2, strides=strides_head_2, padding='same', input_shape=dim_X,
                kernel_initializer=weight_initializers, activation="relu", data_format='channels_last')(inputs2)
    x2 = Dropout(dropout_conv)(x2)
    x2 = Conv1D(conv_2_size, kernel_size=kernel_size_head_2, kernel_initializer=weight_initializers, strides=strides_head_2,
                padding='same', activation='relu')(x2)

    x2 = Dropout(dropout_conv)(x2)
    flat2 = Flatten()(x2)  # ---> flat head 2

    # merge
    merged = concatenate([flat1, flat2])

    d = Dense(layer_1_size, kernel_initializer=weight_initializers,
              activation='relu')(merged)
    d = Dropout(dropout_dense)(d)
    d = Dense(layer_2_size,
              kernel_initializer=weight_initializers, activation='relu')(d)
    d = Dropout(dropout_dense)(d)
    outputs = Dense(4, kernel_initializer=weight_initializers,
                    activation='softmax')(d)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    if(optimizer == "adam"):

        opt = Adam(learning_rate=learning_rate)

    else:
        opt = SGD(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=["accuracy"])
    return model


def run_experiment(args):
    """run training and save to wandb"""
    wandb.init(project=args.project_name)

    sample_rate = args.sampling_rate
    X, y_labels = preprocess.start(
        "Cross", sample_rate, test_path_folder=None, generator=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_labels, test_size=0.06, random_state=42, stratify=y_labels)
    y_train = preprocess.convert_y_labels_to_hot_vectors(y_train)
    y_val = preprocess.convert_y_labels_to_hot_vectors(y_val)
    X_train = preprocess.outliers_removal(X_train)
    # First head configuration
    kernel_size_head_1 = int(X_train[1] * 0.05)
    strides_head_1 = int(kernel_size_head_1/2)

    # Second head configuration
    kernel_size_head_2 = int(X_train[1] * 0.25)
    strides_head_2 = int(kernel_size_head_2/2)

    wandb.config.update(args)
    indices_train = np.arange(X_train.shape[0])
    indices_val = np.arange(X_val.shape[0])
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_val)

    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    X_val = X_train[indices_val]
    y_val = y_train[indices_val]
    model = build_model(X_train, y_train, X_val, y_val, conv_1_size=args.conv_1_size, conv_2_size=args.conv_2_size, kernel_size_head_1=kernel_size_head_1,  strides_head_1=strides_head_1, kernel_size_head_2=kernel_size_head_2,
                        strides_head_2=strides_head_2, layer_1_size=args.layer_1_size, dropout_conv=args.dropout_conv, dropout_dense=args.dropout_dense, layer_2_size=args.layer_2_size, optimizer=args.optimizer, learning_rate=args.learning_rate, weight_initializers=args.weight_initializers)
    # log all values of interest to wandb
    model.fit([X_train, X_train], y_train, validation_data=([X_val, X_val], y_val), epochs=args.epochs,
              batch_size=args.batch_size, callbacks=[WandbCallback(monitor="val_loss", mode="min")], verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Project info args
    # ----------------------------
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        default=PROJECT_NAME,
        help="Final project for the deep learning course")

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Conv1D")

    # Model architecture args
    # -------------------------------
    parser.add_argument(
        "-c1",
        "--conv_1_size",
        type=int,
        default=C1_SIZE,
        help="Conv layer 1 filters size")

    parser.add_argument(
        "-c2",
        "--conv_2_size",
        type=int,
        default=C2_SIZE,
        help="Conv layer 2 filters size")

    parser.add_argument(
        "-l1",
        "--layer_1_size",
        type=int,
        default=L1_SIZE,
        help="layer 1 size")

    parser.add_argument(
        "-dc",
        "--dropoutconv",
        type=float,
        default=DROPOUT_CONV,
        help="dropout before dense layers")
    parser.add_argument(
        "-dl",
        "--dropoutdense",
        type=float,
        default=DROPOUT_DENSE,
        help="dropout before dense layers")

    parser.add_argument(
        "-l2",
        "--layer_2_size",
        type=int,
        default=L2_SIZE,
        help="layer 2 size")

    # -------------------------------

    # Training info args
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs")

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size")
    # -------------------------------------

    # Optimazier args
    # ---------------------------------
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default=OPTIMIZER,
        help="Learning optimizer")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="learning rate")

    parser.add_argument(
        "-wi",
        "--weight_initializers",
        type=str,
        default=WEIGHTS_INITIALIZER,
        help="weight initializers")
    # -------------------------------------

    # data args
    # ---------------------------------
    parser.add_argument(
        "-sr",
        "--sampling_rate",
        type=int,
        default=SAMPLING_RATE,
        help="sampling rate")

    args = parser.parse_args()

    run_experiment(args)
