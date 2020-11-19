from get_CNN_data import get_CNN_data
import numpy as np
from keras.applications import resnet_v2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

model_params = {
    'input_shape': None,  # will be completed by the code
    'batch_size': 1,
    'epochs': 20,
}


def train_net(X_train, y_train, params):
    """

    :param X_train:
    :param y_train:
    :param params:
    :return:
    """
    # Params
    input_shape = params['input_shape']
    epochs = params['epochs']
    batch_size = params['batch_size']

    # Init the resnet
    resnet = resnet_v2.ResNet50V2(include_top=True, weights='imagenet',
                                  pooling='avg', input_shape=input_shape)

    return resnet


def train_model(X_train, y_train):

    clf = RandomForestClassifier(n_estimators=200)
    return clf.fit(X_train, y_train)


def main():
    # Get data and update input shape
    X_train, X_test, y_train, y_test = get_CNN_data()

    # Get ResNet
    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                  pooling='max', input_shape=X_train[0].shape)

    # Train model
    x_train_net = resnet.predict(np.asarray(X_train))
    model = train_model(x_train_net, y_train)

    # Test
    x_test_net = resnet.predict(np.asarray(X_test))
    print('Predictions: {}'.format(model.predict(x_test_net)))
    print('True Labels: {}'.format(np.asarray(y_test)))
    print('Score: {}'.format(model.score(x_test_net, y_test)))


if __name__ == '__main__':
    main()
