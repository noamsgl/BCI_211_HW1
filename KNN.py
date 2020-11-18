from get_data import get_data
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_data()

    clf = KNeighborsClassifier(n_neighbors=5)

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


