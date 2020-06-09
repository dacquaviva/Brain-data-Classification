import argparse
import os
import preprocess
import json
import numpy as np
import math
from datagenerator import DataGenerator
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD, Adam
import wandb

# default config/hyperparameter values
# you can modify these below or via command line
PROJECT_NAME = "Deep Learning Assignment"
MODEL_NAME = "1D cnn"

C1_SIZE = 32
BATCH_NORM = "True"
C2_SIZE = 64
KERNEL_SIZE = 3
L1_SIZE = 16
DROPOUT = 0.2
L2_SIZE = 32

BATCH_SIZE = 32
EPOCHS = 25

OPTIMIZER = "adam"
LEARNING_RATE = 0.001
WEIGHTS_INITIALIZER = "ones"
SAMPLING_RATE = 1


def build_model(X_train_dim, conv_1_size, batch_norm, conv_2_size, kernel_size, layer_1_size, dropout, layer_2_size, optimizer, learning_rate, weight_initializers):
    """ model """

    model = Sequential()
    model.add(Conv1D(conv_1_size, kernel_size=kernel_size, padding='same',
                     activation='relu', input_shape=X_train_dim, kernel_initializer=weight_initializers, data_format='channels_last'))

    if(batch_norm == "True"):
        model.add(BatchNormalization())
        print(batch_norm)
    model.add(Conv1D(conv_2_size, kernel_size=kernel_size, kernel_initializer=weight_initializers,
                     padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(layer_1_size,
                    kernel_initializer=weight_initializers, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(layer_2_size,
                    kernel_initializer=weight_initializers, activation='relu'))
    model.add(Dense(4, kernel_initializer=weight_initializers, activation='softmax'))

    if(optimizer == "adam"):

        opt = Adam(learning_rate=learning_rate)

    else:
        opt = SGD(lr=learning_rate, nesterov=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=["accuracy"])
    return model






def run_experiment(args):
    """run training and save to wandb"""
    wandb.init(project=args.project_name)
    sample_rate = args.sampling_rate
    X, y_labels = preprocess.start(
        "Intra", sample_rate, test_path_folder=None, generator=False)

    model = build_model(X[0].shape, conv_1_size=args.conv_1_size, batch_norm=args.batch_norm, conv_2_size=args.conv_2_size, kernel_size=args.kernel_size, layer_1_size=args.layer_1_size,
                        dropout=args.dropout, layer_2_size=args.layer_2_size, optimizer=args.optimizer, learning_rate=args.learning_rate, weight_initializers=args.weight_initializers)
    # log all values of interest to wandb
    wandb.config.update(args)


    dictionary_metrics = {out: [] for i, out in enumerate(model.metrics_names)}
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, y_labels):
        y = preprocess.convert_y_labels_to_hot_vectors(y_labels)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, epochs=args.epochs,
                  batch_size=args.batch_size, verbose=1)
        metrics = model.evaluate(X_test, y_test)
        for i, metrics_names in enumerate(model.metrics_names):
            dictionary_metrics[metrics_names].append(metrics[i])

    for key in dictionary_metrics:
        dictionary_metrics[key] = np.array(dictionary_metrics[key]).mean()

    wandb.log(dictionary_metrics)


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
        "-bn",
        "--batch_norm",
        type=str,
        default=BATCH_NORM,
        help="batch norm")

    parser.add_argument(
        "-c2",
        "--conv_2_size",
        type=int,
        default=C2_SIZE,
        help="Conv layer 2 filters size")

    parser.add_argument(
        "-k",
        "--kernel_size",
        type=int,
        default=KERNEL_SIZE,
        help="Kernel size")

    parser.add_argument(
        "-l1",
        "--layer_1_size",
        type=int,
        default=L1_SIZE,
        help="layer 1 size")

    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=DROPOUT,
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

