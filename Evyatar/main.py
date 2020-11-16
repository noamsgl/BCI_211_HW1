import wfdb
from tqdm import tqdm
import pickle
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np

data_params = {
    'data_path': '..\\data\\dataset2',
    'pickle_path': '..\\data',
    'use_pickle': True
}

filter_params = {
    'sample_rate': 250,
    'bandpass': [5, 48]
}


def generate_record_names():
    strings = []
    for subject in range(1, 12):
        for session in ["a", "b", "c", "d", "e"]:
            record_name = "T0{:0>2d}{}".format(subject, session)
            strings.append(record_name)
    return strings


def session_to_windows(session, windows_index):
    """

    :param session: the session of the subject (ndarray)
    :param windows_index: list of start and end of each window
    :return: list with all the windows as ndarray
    """

    windows = []

    for i in range(0, len(windows_index), 2):

        windows.append(session[windows_index[i]: windows_index[i+1]])

    return windows


def get_labels(raw_labels):
    """
    get the raw labels (duplicate) and return list of un-duplicate labels
    :param raw_labels: list of string labels
    :return: list of string labels
    """

    current_labels = []

    for i in range(0, len(raw_labels), 2):

        current_labels.append(raw_labels[i])

    return current_labels


def load_data(path, use_pickle, pickle_path):

    """
    create list of ndarray when each element is a window.
    create list of labels for each window.
    :param pickle_path: path to the pickle path
    :param use_pickle: bool if to use the pickle file
    :param path: the path to the data directory
    :return: two lists, one of windows (ndarray) and one of labels (str)
    """

    # Load the pickle file
    if use_pickle:

        return pickle.load(open(pickle_path + '\\windows.pickle', 'rb')),\
               pickle.load(open(pickle_path + '\\labels.pickle', 'rb'))

    # Load data
    records_name = generate_record_names()  # names of all the sessions
    windows = []  # list for all the windows
    labels = []  # list for all the labels

    for session in tqdm(records_name):

        p_signal = wfdb.io.rdrecord(path + '\\' + session).p_signal  # get the full signal of the session

        window_index = wfdb.io.rdann(path + '\\' + session, 'win').sample  # get the indices of the windows

        windows += session_to_windows(p_signal, window_index)  # append the current session's windows

        labels += get_labels(wfdb.io.rdann(path + '\\' + session, 'win').aux_note)

    return windows, labels


def preprocess_window(window, params):

    """

    :param params:
    :param window: list with all the windows as ndarray
    :return:
    """

    bandpass = params['bandpass']
    fs = params['sample_rate']

    # Normalization (mean = 0, std = 1)
    window = np.nan_to_num((window - window.mean(axis=0)) / window.std(axis=0))

    # Bandpass filter
    window = butter_bandpass_filter(window.T, bandpass[0], bandpass[1], fs)

    return window.T


def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':

    windows, labels = load_data(path=data_params['data_path'],
                                use_pickle=data_params['use_pickle'],
                                pickle_path=data_params['pickle_path'])

    windows = [preprocess_window(w, filter_params) for w in windows]



