# Author:   Sebastian Law
# Date:     07-Jul-2016
# Revised:  08-Jul-2016

# TODO: convert classification to one-hot encoding

from sklearn import datasets
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_sklearn():
    """Loads sklearn toy dataset."""
    digits = datasets.load_digits()
    n = len(digits.images)
    data = digits.images.reshape((n, -1))
    labels = digits.target
    print(n)
    return data, labels


def load_kaggle_train():
    """Loads kaggle training data from file."""
    path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle', 'Digit Recognizer')
    df = pd.read_csv(os.path.join(path, 'train.csv'), header=0)
    labels = df.ix[:, 0].as_matrix()
    data = df.ix[:, 1:].as_matrix()
    print(len(data))
    return data, labels


def load_kaggle_test():
    """Loads kaggle test data from file."""
    path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle', 'Digit Recognizer')
    df = pd.read_csv(os.path.join(path, 'test.csv'), header=0)
    data = df.as_matrix()
    print(len(data))
    return data


def display(image):
    assert type(image) is np.ndarray
    if image.ndim == 1:  # unflatten if flat
        side = int(np.sqrt(image.shape[0]))
        image = image.reshape(side, -1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
