import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from get_data import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    X_train_lst, X_test_lst, y_train_lst, y_test_lst = get_data()

    scores = []

    for i in range(len(X_test_lst)):

        clf = DecisionTreeClassifier(max_depth=2)

        clf.fit(X_train_lst[i], y_train_lst[i])

        scores.append(round(clf.score(X_test_lst[i], y_test_lst[i]), 3))

        # y_pred = clf.predict(X_test_lst[i])

        # print(confusion_matrix(y_test_lst[i], y_pred))

    print(scores)

    print(np.mean(scores))


