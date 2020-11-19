import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import math
import tensorflow as tf

from keras.applications import resnet_v2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# Don't show TensorFlow warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Data path and test image indices
data_path = 'C:\\Users\\noamga\\Downloads\\FlowerData-20200115\\FlowerData'
test_images_indices = list(range(301, 473))


# ----------------------------------Functions----------------------------


def getDefaultParameters():
    """
    get the default parameters for running the model
    :return: the default parameters for every stage of creating and running the model
    """
    data_params = {'size': [224, 224, 3], 'labels_file_name': 'FlowerDataLabels.mat'}

    train_params = {'input_shape': (data_params['size'][0], data_params['size'][1], data_params['size'][2]),
                    'epochs': 26, 'batch_size': 5}

    tune_params = {'k': 2, 'epochs_range': [26], 'batch_range': list(range(5, 76, 5))}

    test_params = {'batch_size': 32}

    return {'data': data_params, 'train': train_params,
            'tune': tune_params, 'test': test_params}


def getData(params_data):
    """
    The function get the data from the data_path
    :param params_data: parameters for getting the data
    :return: a dictionary contains the data (dict as 'name: image') and the labels (list of 1/0)
    """
    print('\nStart getting the data')
    image_size = params_data['size']  # The desirable size of the images
    images_names = os.listdir(data_path)  # Get all the names of the images
    labels_file_name = params_data['labels_file_name']  # The name of the file containing the labels

    labels = sio.loadmat(data_path + '\\' + labels_file_name)['Labels'][0].tolist()  # Load the labels
    data = {}

    for name in images_names:

        # Skip on the mat file
        if '.mat' in name:
            continue

        img_path = data_path + '\\' + name  # Get the path of the current image
        name_key = name.split('.')[0]  # Get only the name of the image without the .jpeg

        image = cv2.imread(img_path)  # Get the current image
        image = cv2.resize(image, (image_size[0], image_size[1]))  # Resize the image
        # image = cv2.normalize(image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Norm image
        image = resnet_v2.preprocess_input(image)

        data[name_key] = image

    print('End getting the data')
    return {'data': data, 'labels': labels}


def trainTestSplit(data, labels):
    """
    split the data into train and test, according to the 'test_images_indices' variable
    :param data: the images (dict - name: image)
    :param labels: the labels of the images (list - 1/0)
    :param split_params: parameters for the split stage
    :return: dictionary contains train, train_labels, test, test_labels. The train/test are ndarray,
    and the train/test labels are lists
    """
    # The train images indices
    train_images_indices = np.setdiff1d(range(1, len(labels) + 1), test_images_indices)

    # Set the train and test labels
    # The indices mean the name of the image, hence image name '1' is in labels[0]
    train_labels = [labels[i - 1] for i in train_images_indices]
    test_labels = [labels[i - 1] for i in test_images_indices]

    # Set the train and test data
    train_size = (len(train_labels),) + data['1'].shape
    test_size = (len(test_labels),) + data['1'].shape

    train = np.zeros(train_size)
    test = np.zeros(test_size)

    for index, img_name in enumerate(train_images_indices):
        train[index, :] = data[str(img_name)]

    for index, img_name in enumerate(test_images_indices):
        test[index, :] = data[str(img_name)]

    # Debug prints
    print('\nThe data was split into train & test:')
    print('Train data size:', train.shape)
    print('Train labels size:', len(train_labels))
    print('Test data size:', test.shape)
    print('Test labels size:', len(test_labels))

    return {'train': train, 'train_labels': train_labels, 'test': test,
            'test_labels': test_labels}


def trainModel(train, train_labels, train_model_params):
    """
        creates net and trains it.
        :param train: numpy array of the train images
        :param train_labels: list of the labels of the train images
        :param train_model_params: parameters for the split stage
        :return: a trained network on the relevant data
        """
    print('\nStart creating the CNN model')
    input_shape = train_model_params['input_shape']
    epochs = train_model_params['epochs']
    batch_size = train_model_params['batch_size']

    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape)
    for layer in resnet.layers:
        layer.trainable = False

    network = Sequential()
    network.add(resnet)
    network.add(Dense(512, activation='relu', input_dim=input_shape))
    network.add(Dropout(0.3))
    network.add(Dense(512, activation='relu'))
    network.add(Dropout(0.3))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=2e-5),
                    metrics=['accuracy'])
    # Print the summary of the model
    network.summary()

    # For using data augmentation
    train_datagen = ImageDataGenerator(rotation_range=50,
                                       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                       horizontal_flip=True, fill_mode="nearest")

    train_generator = train_datagen.flow(train, train_labels, batch_size=batch_size)

    network.fit_generator(train_generator, epochs=epochs,
                          steps_per_epoch=math.ceil(len(train_labels) / batch_size))

    return network


def testModel(network, test, params):
    """
        predicts the calsses and score for each test image
        :param network: fitted resnet network
        :param test: numpy of the test images to classify
        :param params: parameters for the test stage
        :return: dictionary with predicted classes and scores
        """

    batch_size = params['batch_size']
    pred_classes = network.predict_classes(x=test, batch_size=batch_size, verbose=1)
    scores = network.predict(x=test, batch_size=batch_size, verbose=1)
    return {'classes': pred_classes, 'scores': scores}


def getError(result_labels, test_labels):
    """
        calculates the error of the model
        :param result_labels: numpy array of the classes predicted
        :param test_labels: list of the true labels of the test images
        :return: returns the error rate of the model
        """

    error = 0

    for index in range(0, len(result_labels)):
        if result_labels[index] != test_labels[index]:
            error = error + 1

    return error / len(result_labels)


def evaluate(results, test_labels, test_data):
    """
        displays the 5 largest errors and the recall precision curve, prints to console the error
        :param results: dictionary including predicted classes and scored
        :param test_labels: list of the true labels of the test images
        :param test_data: numpy of the test images that were tested
        """

    display_5max_errors(results, test_labels, test_data)
    display_recall_precision_curve(results, test_labels)
    error = getError(results['classes'], test_labels)
    print("The test error result is: " + str(error))


def display_recall_precision_curve(results, test_labels):
    """
           displays the the recall precision curve,
           :param results: dictionary including predicted classes and scored
           :param test_labels: list of the true labels of the test images
           """
    scores = results['scores']
    real_labels = np.array(test_labels)

    precision, recall, _ = precision_recall_curve(real_labels.ravel(), scores.ravel())

    fig = plt.figure()
    plt.plot(recall, precision, alpha=0.2, color='g', )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall Precision Curve')
    plt.show()


def display_5max_errors(results, test_labels, test_data):
    """
           finds the 5 largest errors
           :param results: dictionary including predicted classes and scored
           :param test_labels: list of the true labels of the test images
           :param test_data: numpy of the test images that were tested
           """
    # Typei_errors = {'index': '[score, distance from 0.5]'}
    Type1_errors = {}
    Type2_errors = {}
    pred_classes = results['classes']
    scores = results['scores']

    # Find all the errors
    for i in range(len(scores)):

        if pred_classes[i] != test_labels[i]:
            if pred_classes[i] == 1:
                Type2_errors[i] = scores[i][0]
            else:
                Type1_errors[i] = scores[i][0]

    Type1_sorted_errors = sorted(Type1_errors.items(), key=lambda x: x[1])  # tuple
    Type2_sorted_errors = sorted(Type2_errors.items(), reverse=True, key=lambda x: x[1])  # tuple
    Type1_max5_errors = Type1_sorted_errors[0:5]
    Type2_max5_errors = Type2_sorted_errors[0:5]

    # For display the images run these two lines
    # report_Figures(Type1_max5_errors, test_data, 'Type 1')
    # report_Figures(Type2_max5_errors, test_data, 'Type 2')


def report_Figures(max5_errors, test_data, Type):
    """
           displays the 5 largest errors
           :param max5_errors: list of tuples of items in error dictionary  -{'index': '[score, distance from 0.5, Type]'}
           :param test_data: numpy of the test images that were tested
           """
    for i in range(len(max5_errors)):
        error = max5_errors[i]
        fig = plt.figure()
        Title = "Error Type: " + Type + ", Error Index: " + str(i + 1) + ", Classification Score: " + str(error[1])
        fig.suptitle(Title, fontsize=16)
        # plt.imshow(cv2.cvtColor(test_data[error[0]], cv2.COLOR_BGR2RGB))
        plt.imshow(test_data[error[0]])
        plt.axis('off')
        plt.show()


def trainWithTuning(train, train_labels, params):
    """
        Tunes the model using kfold for epoch and batchsize parameters
        :param train: numpy array of the train images
        :param train_labels: list of the labels of the train images
        :param params: parameters for the tuning
        """

    # Define the K parameter
    k = params['k']

    # Get all the ranges to check
    epochs_range = params['epochs_range']
    batch_range = params['batch_range']
    input_shape = (224, 224, 3)

    results = {'Type': '(Epoch, Batch)'}

    for epoch in epochs_range:
        for batch in batch_range:

            # Run K-Folds algorithm
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=7)
            errors = []

            for train_index, test_index in kfold.split(train, train_labels):

                # Initiate the current cross-validation train and test datasets
                cv_train = np.zeros((len(train_index),) + train[0].shape)
                cv_test = np.zeros((len(test_index),) + train[0].shape)

                for i, j in enumerate(train_index):
                    cv_train[i, :] = train[j, :]

                for i, j in enumerate(test_index):
                    cv_test[i, :] = train[j, :]

                cv_train_labels = [train_labels[i] for i in train_index]
                cv_test_labels = [train_labels[i] for i in test_index]

                # For using data augmentation
                train_datagen = ImageDataGenerator(rotation_range=50,
                                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                                   horizontal_flip=True, fill_mode="nearest")

                train_generator = train_datagen.flow(cv_train, cv_train_labels, batch_size=batch)

                # Create the network
                resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling='avg',
                                              input_shape=input_shape)

                for layer in resnet.layers:
                    layer.trainable = False

                network = Sequential()
                network.add(resnet)
                network.add(Dense(512, activation='relu', input_dim=input_shape))
                network.add(Dropout(0.3))
                network.add(Dense(512, activation='relu'))
                network.add(Dropout(0.3))
                network.add(Dense(1, activation='sigmoid'))
                network.compile(loss='binary_crossentropy',
                                optimizer=optimizers.RMSprop(lr=2e-5),
                                metrics=['accuracy'])

                # network.summary()

                network.fit_generator(train_generator, epochs=epoch,
                                      steps_per_epoch=math.ceil(len(cv_train_labels) / batch))

                # network.fit(cv_train, cv_train_labels, epochs=epoch, batch_size=batch, verbose=1)

                predict = network.predict_classes(cv_test)
                current_error = getError(predict, cv_test_labels)
                errors.append(current_error)
                print(current_error)

            results['(' + str(epoch) + ',' + str(batch) + ')'] = np.mean(errors)


def plotEpochs(results, title):
    """
    Plot a graph of tuning the epochs
    :param results: the results of the tuning. dictionary contains tuple as a key (epoch, batch) and the average error
                    as the value
    :param title: the title to show on the plot
    :return: nothing. display the graph
    """
    # Tuning the epochs parameter
    epochs = []
    errors = []
    for key in results:
        epochs.append(eval(key)[0])
        errors.append(results[key])

    # Find the minimum and update the title
    min_index = np.argmin(errors)
    min_epoch = epochs[min_index]
    min_error = round(errors[min_index], 4)
    title = title + '\n' + 'Minimum: epoch = ' + str(min_epoch) + ', error = ' + str(min_error)

    # Debug
    # print('Epochs:', epochs)
    # print('Erros:', errors)

    # Create the plot
    plt.plot(epochs, errors)
    plt.xlabel('Epochs')
    plt.ylabel('Average Error')
    plt.title(title)

    plt.show()


def plotBatch(results, title):
    """
    Plot a graph of tuning the batch size
    :param results: the results of the tuning. dictionary contains tuple as a key (epoch, batch) and the average error
                    as the value
    :param title: the title to show on the plot
    :return: nothing. display the graph
    """
    # Tuning the epochs parameter
    batches = []
    errors = []
    for key in results:
        batches.append(eval(key)[1])
        errors.append(results[key])

    # Debug
    # print('Batches:', batches)
    # print('Erros:', errors)

    # Find the minimum and update the title
    min_index = np.argmin(errors)
    min_batch = batches[min_index]
    min_error = round(errors[min_index], 4)
    title = title + '\n' + 'Minimum: batch = ' + str(min_batch) + ', error = ' + str(min_error)

    # Create the plot
    plt.plot(batches, errors)
    plt.xlabel('Batch')
    plt.ylabel('Average Error')
    plt.title(title)
    plt.show()


def analysis():
    """
    The function display the analysis of the tuning stage. The tuning contains three main steps:
                1. Tune the basic ResNet
                2. Tune the enhanced ResNet
                3. Tune the enhanced ResNet with data augmentation.
    Finally the function display the results of each model on the train and test images
    :return: nothing. display the graphs
    """
    # ---------------------------------------- Basic CNN - ResNet50V2
    # Key Type: (Epoch, Batch)
    basicTuneEpoch = {
        '(1,32)': 0.5433333333333333,
        '(2,32)': 0.51,
        '(3,32)': 0.5766666666666667,
        '(4,32)': 0.45666666666666667,
        '(5,32)': 0.43,
        '(6,32)': 0.48,
        '(7,32)': 0.55,
        '(8,32)': 0.55,
        '(9,32)': 0.5,
        '(10,32)': 0.5733333333333334,
        '(11,32)': 0.5933333333333333,
        '(12,32)': 0.5700000000000001,
        '(13,32)': 0.4866666666666667,
        '(14,32)': 0.52,
        '(15,32)': 0.56}

    basicTuneBatch = {
        '(5,5)': 0.53,
        '(5,10)': 0.5166666666666667,
        '(5,15)': 0.52,
        '(5,20)': 0.43666666666666665,
        '(5,25)': 0.5066666666666667,
        '(5,30)': 0.3866666666666667,
        '(5,35)': 0.43333333333333335,
        '(5,40)': 0.5700000000000001,
        '(5,45)': 0.44,
        '(5,50)': 0.49,
        '(5,55)': 0.5366666666666666,
        '(5,60)': 0.4633333333333334,
        '(5,65)': 0.5833333333333333,
        '(5,70)': 0.5733333333333334,
        '(5,75)': 0.4833333333333334}

    plotEpochs(basicTuneEpoch, 'Basic ResNet50V2 - tune epoch (batch = 32)')
    plotBatch(basicTuneBatch, 'Basic ResNet50V2 - tune batch (epoch = 5)')

    # ------------------------------------ Enhanced CNN - ResNet50V2 with dropouts
    # Key Type: (Epoch, Batch)
    enhancedTuneEpoch = {
        '(1,32)': 0.5033333333333333,
        '(2,32)': 0.44333333333333336,
        '(3,32)': 0.42000000000000004,
        '(4,32)': 0.3833333333333333,
        '(5,32)': 0.39,
        '(6,32)': 0.39,
        '(7,32)': 0.3833333333333333,
        '(8,32)': 0.31,
        '(9,32)': 0.37,
        '(10,32)': 0.29333333333333333,
        '(11,32)': 0.3433333333333333,
        '(12,32)': 0.31,
        '(13,32)': 0.31333333333333335,
        '(14,32)': 0.33666666666666667,
        '(15,32)': 0.32333333333333336}

    enhancedTuneBatch = {
        '(10,5)': 0.31666666666666665,
        '(10,10)': 0.32666666666666666,
        '(10,15)': 0.2966666666666667,
        '(10,20)': 0.3333333333333333,
        '(10,25)': 0.3566666666666667,
        '(10,30)': 0.33666666666666667,
        '(10,35)': 0.31666666666666665,
        '(10,40)': 0.36,
        '(10,45)': 0.31,
        '(10,50)': 0.32,
        '(10,55)': 0.3833333333333333,
        '(10,60)': 0.37,
        '(10,65)': 0.27,
        '(10,70)': 0.33333333333333337,
        '(10,75)': 0.44}

    plotEpochs(enhancedTuneEpoch, 'Enhanced ResNet50V2 - tune epoch (batch = 32)')
    plotBatch(enhancedTuneBatch, 'Enhanced ResNet50V2 - tune batch (epoch = 10)')

    # ------------------------------------ Enhanced CNN - ResNet50V2 with data augmentation
    # Key Type: (Epoch, Batch)
    augmentationTuneEpoch = {
        '(15,32)': 0.36,
        '(16,32)': 0.3433333333333334,
        '(17,32)': 0.35,
        '(18,32)': 0.29333333333333333,
        '(19,32)': 0.33333333333333337,
        '(20,32)': 0.30666666666666664,
        '(21,32)': 0.32999999999999996,
        '(22,32)': 0.29000000000000004,
        '(23,32)': 0.32666666666666666,
        '(24,32)': 0.29000000000000004,
        '(25,32)': 0.30000000000000004,
        '(26,32)': 0.21666666666666667,
        '(27,32)': 0.35333333333333333,
        '(28,32)': 0.3,
        '(29,32)': 0.24666666666666667}

    augmentationTuneBatch = {
        '(26,5)': 0.24,
        '(26,10)': 0.27,
        '(26,15)': 0.29,
        '(26,20)': 0.2866666666666667,
        '(26,25)': 0.32,
        '(26,30)': 0.30333333333333334,
        '(26,35)': 0.25666666666666665,
        '(26,40)': 0.32,
        '(26,45)': 0.33666666666666667,
        '(26,50)': 0.2733333333333333,
        '(26,55)': 0.2866666666666667
        , '(26,60)': 0.25666666666666665,
        '(26,65)': 0.32999999999999996,
        '(26,70)': 0.3,
        '(26,75)': 0.3666666666666667}

    plotEpochs(augmentationTuneEpoch, 'Enhanced ResNet50V2 with augmentation - tune epoch (batch = 32)')
    plotBatch(augmentationTuneBatch, 'Enhanced ResNet50V2 with augmentation - tune batch (epoch = 26)')

    # ------------------------------------ All CNN - ResNet50V2 test results

    models = ['Basic', 'Enhanced', 'Enhanced & Augmentation']
    train_results = [0.38, 0.27, 0.24]
    test_results = [0.343, 0.29, 0.215]

    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, train_results, width, label='Train', color='tomato')
    ax.bar(x + width / 2, test_results, width, label='Test', color='royalblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error')
    ax.set_title('Summary - Train & Test Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.show()


# ------------------------------------Main-------------------------------


def main():
    np.random.seed(0)

    params = getDefaultParameters()

    data = getData(params['data'])

    splitData = trainTestSplit(data['data'], data['labels'])

    # Use the train with tuning in order to tune the hyper-parameters
    # trainWithTuning(splitData['train'], splitData['train_labels'], params['tune'])

    model = trainModel(splitData['train'], splitData['train_labels'], params['train'])

    results = testModel(model, splitData['test'], params['test'])

    evaluate(results, splitData['test_labels'], splitData['test'])


# ------------------------------------------------------------------------

# Call the main function in order to run the model
# Call the analysis function in order the see the plots we got while tuning the three models
main()

# analysis()
