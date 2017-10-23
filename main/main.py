import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.patches import Ellipse

from classifiers.logisitc_regression import LogisticRegression
from mixtures.gmm import GaussianMixture

# returns an Ellipse object when given a center and covariance matrix
def get_ellipse(mean, cov):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 4 * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False)

    return ellip

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

def test_gmm():
    # create a synthetic data set made of gaussians
    N1 = 500
    N2 = 500
    N3 = 500
    X = np.zeros((N1 + N2 + N3, 2))
    y = np.zeros((N1 + N2 + N3))
    X[:N1, :] = np.random.multivariate_normal(mean=[0, 0], cov=[[7, -3], [-3, 8]], size=N1)
    X[N1:N1 + N2, :] = np.random.multivariate_normal(mean=[20, 20], cov=[[6, -5], [-5, 5]], size=N2)
    X[N1 + N2:N1 + N2 + N3, :] = np.random.multivariate_normal(mean=[-20, 20], cov=[[2, 2], [2, 3]], size=N3)

    gmm = GaussianMixture(n_mixtures=3, covariance_type='full', reg_covar=1e-6, n_iter=10, init_params='kmeans',
                          weights_init=None, means_init=None, random_state=None, warm_start=True)

    gmm.fit(X, weights_update=None)
    gmm.fit(X, weights_update=np.array([1,100,1]))

    centers = gmm.means_
    cov_matrices = gmm.covariances_

    # plot data
    plt.figure()
    ax = plt.gca()
    plt.scatter(X[:, 0], X[:, 1], color='red', marker='*')
    for i in range(len(centers)):
        plt.scatter(centers[i][0], centers[i][1], color='black', marker='o', label='versicolor')
    for i in range(len(centers)):
        ellipse = get_ellipse(centers[i], cov_matrices[i, :, :])
        ax.add_patch(ellipse)

    plt.tight_layout()
    plt.show()
    pass

if __name__ == '__main__':

    from sklearn.utils.estimator_checks import check_estimator

    def test_classifier():
        return check_estimator(LogisticRegression)

    #test_classifier()
    #test_logReg_2classes()
    #test_logReg_3classes()
    #test_logReg_blobs()
    test_gmm()


    pass