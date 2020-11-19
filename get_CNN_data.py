import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet_v2 import preprocess_input

data_params = {

    'paths': {'X_train': 'data/MI/CNN/X_train.mat',
              'X_test': 'data/MI/CNN/X_test.mat',
              'y_train': 'data/MI/CNN/y_train.mat',
              'y_test': 'data/MI/CNN/y_test.mat', },
}


def get_data(paths):
    """
    This function load the mat files and convert them from 3d matrix to list of ndarray
    :return: 4 lists for X and y train and test.
    """

    # Load the mat files
    # Note: I saved the mat files with var name of X & y, if your var names are different you
    # need to change it.
    X_test = sio.loadmat(paths['X_test'])['X']
    X_train = sio.loadmat(paths['X_train'])['X']
    y_test = sio.loadmat(paths['y_test'])['y']
    y_train = sio.loadmat(paths['y_train'])['y']

    # Change the X files to be list instead of 3d arrays
    X_train = [X_train[i] for i in range(np.shape(X_train)[0])]
    X_test = [X_test[i] for i in range(np.shape(X_test)[0])]

    return X_train, X_test, y_train, y_test


def scale_data(X):
    """
    This function normalize each column to be between 0 to 1.

    :param X: the data to scale
    :return: scaled X
    """

    pass


def get_CNN_data():
    """

    :return:
    """

    X_train, X_test, y_train, y_test = get_data(data_params['paths'])

    # Pre-process the X data
    X_train = [preprocess_input(x) for x in X_train]
    X_test = [preprocess_input(x) for x in X_test]

    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = get_CNN_data()
