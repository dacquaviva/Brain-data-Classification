import h5py
import numpy as np
from os.path import join, isfile
from os import listdir, remove, path
import json
import keras


def start(type_data, sample_rate, test_path_folder=None, generator=False):
    dataset_path = "gdrive/My Drive/colab/DEEP LEARNING/data/"
    classes = {'rest': 0, 'task_motor': 1,
               'task_story_math': 2, 'task_working_memory': 3}
    channels = 248
    time_step = 35624
    if type_data == "Intra":

        if test_path_folder:
            number_of_samples = 8
            path_h5_data = dataset_path + 'Intra/' + test_path_folder
        else:
            number_of_samples = 32
            path_h5_data = dataset_path + 'Intra/train/'
        path_numpy_data = dataset_path + 'Intra/numpy_train/'
    elif type_data == "Cross":

        if test_path_folder:
            number_of_samples = 16
            path_h5_data = dataset_path + 'Cross/' + test_path_folder
        else:
            number_of_samples = 64
            path_h5_data = dataset_path + 'Cross/train/'
        path_numpy_data = dataset_path + 'Cross/numpy_train/'
    else:
        raise Exception(
            "No Intra nor Cross has been selected, make sure to select one of them!")

    data, y_label, labels, order_sample = create_numpy_from_file(
        number_of_samples, channels, time_step, path_h5_data, path_numpy_data, classes)
    data = downsample(data, sample_rate)
    scaled_data, minmax = scale_data(data, test_path_folder, path_numpy_data)
    if(generator):
        save_numpy_data_in_chunks(data, path_numpy_data, order_sample)
        with open(path_numpy_data + 'labels.json', 'w') as fp:
            json.dump(labels, fp)
    with open(path_numpy_data + 'minmax.json', 'w') as fp:
        json.dump(minmax, fp)

    return data, y_label


def outliers_removal(X):
    for sample in range(X.shape[0]):  # loop over sample
        for channel in range(X.shape[2]):  # loop over channel
            # calculate column mean and standard deviation
            data_mean, data_std = np.mean(
                X[sample, :, channel]), np.std(X[sample, :, channel])
            # define outlier bounds
            cut_off = data_std * 4
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # remove too small
            n_outliers = 0
            for time_step in range(X.shape[2]):  # loop over time step channel
                if X[sample, time_step, channel] < lower:
                    n_outliers = n_outliers + 1
                if X[sample, time_step, channel] > upper:
                    n_outliers = n_outliers + 1
            if n_outliers > 0:
                X[sample, :, channel] = 0
    return X


def convert_y_labels_to_hot_vectors(y_label):
    n_classes = 4
    return keras.utils.to_categorical(y_label, num_classes=n_classes)


def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def create_numpy_from_file(number_of_samples, channels, time_step, path_h5_data, path_numpy_data, classes):
    filelist = [f for f in listdir(path_numpy_data) if f.endswith(".npy")]
    for f in filelist:
        remove(path.join(path_numpy_data, f))
    data = np.zeros((number_of_samples, time_step, channels))
    labels = {}
    order_sample = []
    y_label = []
    filedirectory = [f for f in listdir(
        path_h5_data) if isfile(join(path_h5_data, f))]
    for idx, sample in enumerate(filedirectory):
        filename_path = path_h5_data + sample
        sep = '.'
        rest = sample.split(sep, 1)[0]
        with h5py.File(filename_path, 'r') as f:
            for className in classes:
                if className in sample:
                    labels[rest] = classes[className]
                    y_label.append(classes[className])
            dataset_name = get_dataset_name(sample)
            matrix = f.get(dataset_name)[()]
            order_sample.append(rest)
            # transponse to have channel last (Since Conv1D do not support anymore channel_first)
            data[idx] = matrix.T

    return data, y_label, labels, order_sample


def downsample(data, sample_rate):
    data = data[:, ::sample_rate, :]
    return data


def scale_data(data, test_path_folder, path_numpy_data):
    if test_path_folder:
        try:
            with open(path_numpy_data + 'minmax.json', 'r') as fp:
                minmax = json.load(fp)
        except:
            raise Exception(
                "No minmax training file found, make sure to preprocess training data first!")
    else:
        minValues = np.amin(data, axis=(0, 1))
        maxValues = np.amax(data, axis=(0, 1))
        minmax = {
            "min": minValues.tolist(),
            "max": maxValues.tolist()
        }
    for channel in range(data.shape[2]):
        data[:, :, channel] = (
            data[:, :, channel] - minmax["min"][channel])/(minmax["max"][channel] - minmax["min"][channel])

    return data, minmax


def save_numpy_data_in_chunks(data, path_numpy_data, order_sample):
    for i in range(data.shape[0]):
        np.save(path_numpy_data + order_sample[i] + '.npy', data[i, :, :])
