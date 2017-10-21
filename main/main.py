import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from classifiers.logisitc_regression import LogisticRegression

def test_logReg_2classes():

    # Import first two features from iris data set and first two classes
    iris = datasets.load_iris()
    X = iris.data[iris.target != 2, :2]
    y = iris.target[iris.target != 2]

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7331)

    # Create and fit the Logistic Regression to
    logReg = LogisticRegression(l=0.01, n_iter=1000, warm_start=False)
    logReg.fit(X_train, y_train)

    # Plot train and test data
    plt.figure()
    ax = plt.gca()
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])
    for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
        plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
    for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
        plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
    # Plot decision boundaries
    weights = logReg.get_weights()
    x_min, x_max = min(X_train[:,0]), max(X_train[:,0])
    y_min, y_max = -(weights[0]+weights[1]*x_min)/weights[2], -(weights[0]+weights[1]*x_max)/weights[2]
    plt.plot([x_min, x_max], [y_min, y_max])

    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    pass

def test_logReg_3classes():

    # Import first two features from iris data set
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7331)

    # Create and fit the Logistic Regression to
    logReg = LogisticRegression(l=0.01, n_iter=100, warm_start=False)
    logReg.fit(X_train, y_train)

    # Plot train and test data
    plt.figure()
    ax = plt.gca()
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])
    for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
        plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
    for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
        plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
    # Plot decision boundaries
    weights = logReg.get_weights()
    x_min, x_max = min(X_train[:,0]), max(X_train[:,0])
    y_min, y_max = -(weights[0,0]+weights[1,0]*x_min)/weights[2,0], -(weights[0,0]+weights[1,0]*x_max)/weights[2,0]
    plt.plot([x_min, x_max], [y_min, y_max])
    y_min, y_max = -(weights[0,1]+weights[1,1]*x_min)/weights[2,1], -(weights[0,1]+weights[1,1]*x_max)/weights[2,1]
    plt.plot([x_min, x_max], [y_min, y_max])
    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    pass

def test_logReg_blobs():

    # Import first two features from iris data set
    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=2)

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7331)

    # Create and fit the Logistic Regression to
    logReg = LogisticRegression(l=0, n_iter=1000, warm_start=False)
    logReg.fit(X_train, y_train)

    # Plot train and test data
    plt.figure()
    ax = plt.gca()
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])
    for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
        plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
    for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
        plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
    # Plot decision boundaries
    weights = logReg.get_weights()
    x_min, x_max = min(X_train[:,0]), max(X_train[:,0])
    y_min, y_max = -(weights[0,0]+weights[1,0]*x_min)/weights[2,0], -(weights[0,0]+weights[1,0]*x_max)/weights[2,0]
    plt.plot([x_min, x_max], [y_min, y_max], )
    y_min, y_max = -(weights[0,1]+weights[1,1]*x_min)/weights[2,1], -(weights[0,1]+weights[1,1]*x_max)/weights[2,1]
    plt.plot([x_min, x_max], [y_min, y_max])
    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    pass

if __name__ == '__main__':

    from sklearn.utils.estimator_checks import check_estimator

    def test_classifier():
        return check_estimator(LogisticRegression)

    #test_classifier()
    #test_logReg_2classes()
    #test_logReg_3classes()
    test_logReg_blobs()



    pass