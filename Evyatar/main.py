

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

