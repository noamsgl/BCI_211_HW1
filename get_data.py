import wfdb
from sklearn import preprocessing
from tqdm import tqdm
import pickle
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.model_selection import train_test_split

data_params = {
    'data_path': 'data\\dataset2',
    'pickle_path': {'X': 'data\\X.pickle',
                    'y': 'data\\y.pickle'},
    'use_pickle': True,
    'n_subject': 11
}

filter_params = {
    'sample_rate': 250,
    'bandpass': [5, 48],
    'notch': 50.0
}

features_params = {
    'selected_channel': 125,  # 126-channel when index start at 0
    'nfft': 512,
    'welch_window': 'hamm'
}

train_params = {
    'train_ratio': 0.85,
    'random_state': 23,
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


def load_sessions(path, use_pickle):
    """
    create list of ndarray when each element is a session.
    :param use_pickle: if to use the pickle file (bool)
    :param path: the path to the data directory
    :return: list with all the sessions (ndarray)
    """

    # Verbose
    print('Loading sessions...')

    # Load data
    if use_pickle:
        print('Using pickle file, not loading sessions')
        return None

    records_name = generate_record_names()  # names of all the sessions
    loaded_sessions = []  # list for all the loaded sessions

    for session in tqdm(records_name):
        loaded_sessions.append(wfdb.io.rdrecord(path + '\\' + session).p_signal)  # get the full signal of the session

    return loaded_sessions


def preprocess_sessions(raw_session, params):
    """

    :param raw_session: list with all the sessions as ndarray
    :param params: pre-processing params
    :return:
    """

    # Verbose
    print('\nPre-processing sessions...')

    # Use pickle?
    if raw_session is None:
        print('Using pickle file, not filtering sessions')
        return None

    # Params
    bandpass = params['bandpass']
    notch = params['notch']
    fs = params['sample_rate']

    preprocessed_sessions = []

    for s in tqdm(raw_session):
        # transpose the session
        s = s.T

        # Normalization (mean = 0, std = 1)
        s = np.nan_to_num((s - s.mean(axis=0)) / s.std(axis=0))

        # Bandpass filter
        s = mne.filter.filter_data(s, fs, bandpass[0], bandpass[1], verbose=False)
        # s = butter_bandpass_filter(s, bandpass[0], bandpass[1], fs)

        # Notch filter
        s = mne.filter.notch_filter(s, fs, notch, verbose=False)

        # Add to the preprocessed list
        preprocessed_sessions.append(s.T)

    return preprocessed_sessions


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


def split_to_windows(sessions, path, pickle_path):
    """
    Split the pre-processed sessions into windows.
    The function also return list of the labels
    :param pickle_path: path to the pickle file
    :param path: path to the data folder
    :param sessions: list with all the sessions
    :return: list to the windows (ndarray) and list of labels (str)
    """

    # Verbose
    print('\nSplitting to windows...')

    # Use pickle?
    if sessions is None:
        print('Loading windows from pickle file')
        return pickle.load(open(pickle_path['X'], 'rb')), pickle.load(open(pickle_path['y'], 'rb'))

    records_name = generate_record_names()  # names of all the sessions
    windows = []
    labels = []

    for i, session in enumerate(records_name):
        window_index = wfdb.io.rdann(path + '\\' + session, 'win').sample  # get the indices of the windows

        windows += session_to_windows(sessions[i], window_index)  # append the current session's windows

        labels += get_labels(wfdb.io.rdann(path + '\\' + session, 'win').aux_note)

    return windows, labels


def feature_selection(windows):
    """
    This function select the feature and return new X list where each element is vector of features.
    According to the article only the 126-channel was selected.
    Then we used welch to get the power spectral density.

    :param windows: list of ndarray
    :return: list of vectors
    """
    print('\nSelecting features...')

    # Params
    sample_rate = filter_params['sample_rate']
    channel = features_params['selected_channel']
    nfft = features_params['nfft']
    welch_window = features_params['welch_window']

    features = []

    for window in windows:
        features.append(welch(window[channel], window=welch_window, fs=sample_rate, nfft=nfft)[1])

    return features


def encode_y(y):
    """

    :param y:
    :return:
    """

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return y


def split_by_subjects(X, y, n_subjects):
    """
    Split X and y by the number of subjects, where each subject have X & y
    :param n_subjects:
    :param X:
    :param y:
    :return:
    """

    chunk_size = len(X) // n_subjects
    X = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
    y = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

    return X, y


def get_data():
    # Get the sessions of the data
    sessions = load_sessions(path=data_params['data_path'],
                             use_pickle=data_params['use_pickle'])

    # Filter the sessions
    sessions = preprocess_sessions(raw_session=sessions,
                                   params=filter_params)

    # Split to windows
    X, y = split_to_windows(sessions=sessions,
                            path=data_params['data_path'],
                            pickle_path=data_params['pickle_path'])

    # Feature extraction & selection
    X = feature_selection(X)

    # Encode y vector
    y = encode_y(y)

    # Split dataset into n=11 subjects
    X, y = split_by_subjects(X, y, data_params['n_subject'])

    # Split into lists of train & test
    X_train_lst, X_test_lst, y_train_lst, y_test_lst = [], [], [], []
    for i in range(data_params['n_subject']):
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i],
                                                            train_size=train_params['train_ratio'],
                                                            random_state=train_params['random_state'])
        X_train_lst.append(X_train)
        X_test_lst.append(X_test)
        y_train_lst.append(y_train)
        y_test_lst.append(y_test)

    return X_train_lst, X_test_lst, y_train_lst, y_test_lst
