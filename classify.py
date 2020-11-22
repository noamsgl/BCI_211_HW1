import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from get_data import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    X_train_lst, X_test_lst, y_train_lst, y_test_lst = get_data()

    scores = []

    # For each subject
    for i in range(len(X_test_lst)):

        # Train model
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_lst[i], y_train_lst[i])

        # Test model and append evaluation
        scores.append(round(clf.score(X_test_lst[i], y_test_lst[i]), 3))

    # Print each subject score and mean score
    print('Subjects scores: {}'.format(scores))
    print('Mean score: {}'.format(np.mean(scores)))


