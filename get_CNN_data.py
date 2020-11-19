import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet_v2 import preprocess_input
import cv2

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
    y_test = sio.loadmat(paths['y_test'])['y'].tolist()[0]
    y_train = sio.loadmat(paths['y_train'])['y'].tolist()[0]

    # Change the X files to be list instead of 3d arrays
    X_train = [X_train[i] for i in range(np.shape(X_train)[0])]
    X_test = [X_test[i] for i in range(np.shape(X_test)[0])]

    return X_train, X_test, y_train, y_test


def preprocess_X(X):
    """
    This function preprocess X data

    :param X: the data to scale
    :return: scaled X
    """

    # Scale using resnet pre-process function
    # X = [preprocess_input(x) for x in X]

    # Resize X
    X = [cv2.resize(x, (224, 224)) for x in X]

    # Change 1 channel to 3 channel
    X = [cv2.merge((x, x, x)) for x in X]

    return X


def get_CNN_data():
    """

    :return:
    """

    X_train, X_test, y_train, y_test = get_data(data_params['paths'])

    # Pre-process the X data
    X_train = preprocess_X(X_train)
    X_test = preprocess_X(X_test)

    return X_train, X_test, y_train, y_test

