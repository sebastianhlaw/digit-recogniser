# Author:   Sebastian Law
# Date:     07-Jul-2016
# Revised:  25-Nov-2016

# TODO: convert classification to one-hot encoding (?) See sklearn.preprocessing.LabelBinarizer

from sklearn import datasets, preprocessing
import os
from sys import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_path():
    """Provides the local path to the data file, depending on platform."""
    if platform == "darwin":
        return os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle', 'Digit Recognizer')
    elif platform == "win32":
        return os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets', 'Kaggle', 'Digit Recognizer')
    else:
        return


def load_sklearn(scaled=True):
    """Loads sklearn toy dataset."""
    digits = datasets.load_digits()
    n = len(digits.images)
    if scaled:
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(digits.images.reshape((n, -1)).astype(float))
    else:
        data = digits.images.reshape((n, -1)).astype(int)
    labels = digits.target.astype(int)
    print(n)
    return data, labels


def load_kaggle_public(scaled=True):
    """Loads kaggle training data from file."""
    df = pd.read_csv(os.path.join(data_path(), 'train.csv'), header=0)
    labels = df.ix[:, 0].as_matrix().astype(int)
    if scaled:
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(df.ix[:, 1:].as_matrix().astype(float))
    else:
        data = df.ix[:, 1:].as_matrix().astype(int)
    print(len(data))
    return data, labels


def load_kaggle_private(scaled=True):
    """Loads kaggle test data from file."""
    df = pd.read_csv(os.path.join(data_path(), 'test.csv'), header=0)
    if scaled:
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data = df.as_matrix().astype(float))
    else:
        data = df.as_matrix().astype(int)
    print(len(data))
    return data


def display(image, label=None):
    assert type(image) is np.ndarray
    if image.ndim == 1:  # unflatten if flat
        side = int(np.sqrt(image.shape[0]))
        image = image.reshape(side, -1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    if label is not None:
        plt.title(str(label))
