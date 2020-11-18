import pickle
import wfdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
        windows.append(session[windows_index[i]: windows_index[i + 1]])

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


def load_data(path, use_pickle=False):
    """
    create list of ndarray when each element is a window.
    create list of labels for each window.
    :return: two lists, one of windows (ndarray) and one of labels (str)
    """

    # Load the pickle file
    if use_pickle:
        return pickle.load(open(path + '\\windows.pickle', 'rb')), \
               pickle.load(open(path + '\\labels.pickle', 'rb'))

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


if __name__ == '__main__':
    path = '../data/dataset2'
    windows, labels = load_data(path)
