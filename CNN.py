from get_CNN_data import get_CNN_data

from keras.applications import resnet_v2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

model_params = {
    'input_shape': (786, 2, 1),
    'batch_size': 32,
    'epochs': 50,
}


def train_model(X_train, y_train, params):
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
    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                  pooling='avg', input_shape=input_shape)

    # Make all layers un-trainable
    for layer in resnet.layers:
        layer.trainable = False

    # Add resnet to your net with some more layers
    net = Sequential()
    net.add(resnet)
    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(1, activation='sigmoid'))

    # Compile net
    net.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['accuracy'])

    # Fit the net
    net.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Print the net summary
    net.summary()

    return net


def main():
    # Get data and update input shape
    X_train, X_test, y_train, y_test = get_CNN_data()
    model_params['input_shape'] = X_train[0].shape

    # Train model
    model = train_model(X_train, y_train, model_params)


if __name__ == '__main__':
    main()
