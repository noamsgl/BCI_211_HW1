from keras_preprocessing.image import ImageDataGenerator

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
    'batch_size': 8,
    'epochs': 20,
    'steps': 15,
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
    steps = params['steps']

    # Init the ResNet
    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                  pooling='avg', input_shape=input_shape)

    # Make all layers un-trainable
    for layer in resnet.layers:
        layer.trainable = False

    # Add ResNet to your net with some more layers
    net = Sequential()
    net.add(resnet)
    net.add(Dropout(0.5))
    net.add(Dense(1, activation='sigmoid'))

    # Compile net
    net.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['accuracy'])

    # Print the net summary
    net.summary()

    # Fit the net
    # net.fit(np.asarray(X_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs)

    # For using data augmentation
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                       horizontal_flip=False, fill_mode="nearest", vertical_flip=True)

    train_generator = train_datagen.flow(np.asarray(X_train), np.asarray(y_train), batch_size=batch_size)

    net.fit_generator(train_generator, epochs=epochs,
                      steps_per_epoch=steps)

    return net


def train_model(X_train, y_train):
    clf = SVC()
    return clf.fit(X_train, y_train)


def main():

    # Get data and update input shape
    X_train, X_test, y_train, y_test = get_CNN_data()
    model_params['input_shape'] = X_train[0].shape

    # Train net
    net = train_net(X_train, y_train, model_params)

    # Test
    # net.score(np.asarray(X_test), np.asarray(y_test))


if __name__ == '__main__':
    main()
