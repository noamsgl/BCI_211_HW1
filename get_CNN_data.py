import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pickle

data_params = {

    'paths': {'X': 'data/MI/CNN/X.pickle',
              'y': 'data/MI/CNN/y.pickle'},
    'image_size': (32, 32),
    'train_ratio': 0.85,
    'random_state': 23,
}


def get_data(params):

    """
    This function load the mat files and convert them from 3d matrix to list of ndarray
    :return: 4 lists for X and y train and test.
    """

    # Params
    paths = params['paths']
    train_ratio = params['train_ratio']
    random_state = params['random_state']

    # Load the pickle file with the data
    X = pickle.load(open(paths['X'], 'rb'))
    y = pickle.load(open(paths['y'], 'rb'))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_state)

    return X_train, X_test, y_train, y_test


def preprocess_X(X, image_size):
    """
    This function preprocess X data

    :param image_size: image size to resize
    :param X: the data to scale
    :return: scaled X
    """

    # Scale using resnet pre-process function
    # X = [preprocess_input(x) for x in X]

    # Resize X
    X = [cv2.resize(x, image_size, interpolation=cv2.INTER_AREA) for x in X]

    # Debug - show X after resize
    # plt.imshow(X[0])
    # plt.show()

    # Change 1 channel to 3 channel
    X = [cv2.merge((x, x, x)) for x in X]

    return X


def get_CNN_data():
    """

    :return:
    """

    X_train, X_test, y_train, y_test = get_data(data_params)

    # Pre-process the X data
    X_train = preprocess_X(X_train, data_params['image_size'])
    X_test = preprocess_X(X_test, data_params['image_size'])

    return X_train, X_test, y_train, y_test

