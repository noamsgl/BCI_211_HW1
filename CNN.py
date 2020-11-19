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
from sklearn.model_selection import GridSearchCV

cv_params = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.1, 0.001, 0.0001, 0.00001],
}


def train_model(X_train, y_train):

    clf = SVC(C=100, gamma=0.00001)
    return clf.fit(X_train, y_train)


def main():
    # Get data and update input shape
    X_train, X_test, y_train, y_test = get_CNN_data()

    # Get ResNet
    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                  pooling='avg', input_shape=X_train[0].shape)

    # Feature extraction using ResNet
    x_train_net = resnet.predict(np.asarray(X_train))
    x_test_net = resnet.predict(np.asarray(X_test))

    # Train model
    model = train_model(x_train_net, y_train)

    # Cross-validate model
    # svm = SVC()
    # clf = GridSearchCV(svm, cv_params)

    # Test
    # print('Predictions: {}'.format(model.predict(x_test_net)))
    # print('True Labels: {}'.format(np.asarray(y_test)))
    # print('Score: {}'.format(model.score(x_test_net, y_test)))


if __name__ == '__main__':
    main()
