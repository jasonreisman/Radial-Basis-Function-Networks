import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.patches import Ellipse

from classifiers.rbfn import RadialBasisFunctionNetwork
#from classifiers.logisitc_regression import LogisticRegression
from mixtures.gmm import GaussianMixture
from sklearn.metrics import log_loss

def log_reg_log_loss(y_true, y_pred, C, coef, intercept=np.array([0])):

    loss = log_loss(y_true, y_pred)
    loss = loss + (1 / (2 * C)) * (np.sum(np.square(intercept)) + np.sum(np.square(coef)))

    return loss

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

# plots the level curves of the probability estimations in 2D space
def plot_decision_regions(X_train, X_test, y_train, y_test, classifier, resolution=0.1):

    # setup color map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])

    # plot the decision surface
    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)
    for i in range(Z.shape[1]):
        Zi = Z[:, i].reshape(xx1.shape)
        cp = plt.contour(xx1, xx2, Zi, alpha=0.4, cmap=None)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot train samples
        for idx, cl in enumerate(np.sort(np.unique(y_train))):
            plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
        # plot test samples
        for idx, cl in enumerate(np.sort(np.unique(y_test))):
            plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')


        plt.tight_layout()
        plt.show()

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
    y_min, y_max = -(weights[0, 0]+weights[1, 0]*x_min)/weights[2, 0], -(weights[0, 0]+weights[1, 0]*x_max)/weights[2, 0]
    plt.plot([x_min, x_max], [y_min, y_max])

    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
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
    print(accuracy)
    pass

def test_logReg_blobs():

    # create a synthetic data set made of gaussians
    N1 = 500
    N2 = 500
    N3 = 500
    X = np.zeros((N1 + N2 + N3, 5))
    y = np.zeros((N1 + N2 + N3))
    y[N1:N1 + N2] = 1
    y[N1 + N2:N1 + N2 + N3] = 2
    X[:N1, :] = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0], cov=[[7, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 9, 0], [0, 0, 0, 0, 50]], size=N1)
    X[N1:N1 + N2, :] = np.random.multivariate_normal(mean=[29, 100, 50, 200, 100], cov=[[20, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 30, 0], [0, 0, 0, 0, 0.1]], size=N2)
    X[N1 + N2:N1 + N2 + N3, :] = np.random.multivariate_normal(mean=[-29, -100, -50, -200, 0], cov=[[5, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 4.5, 0, 0], [0, 0, 0, 6.5, 0], [0, 0, 0, 0, 0]], size=N3)


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
    y_min, y_max = -(weights[0,0]+weights[1,0]*x_min)/weights[2,0], -(weights[0,0]+weights[1,0]*x_max)/weights[2,0]
    plt.plot([x_min, x_max], [y_min, y_max], )
    y_min, y_max = -(weights[0,1]+weights[1,1]*x_min)/weights[2,1], -(weights[0,1]+weights[1,1]*x_max)/weights[2,1]
    plt.plot([x_min, x_max], [y_min, y_max])
    plt.tight_layout()
    plt.show()

    #plot_decision_regions(X_train, X_test, y_train, y_test, logReg, resolution=0.1)

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
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

    gmm.fit(X)

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

def test_gmm_D():
    np.random.seed(1)
    # create a synthetic data set made of gaussians
    N1 = 500
    N2 = 500
    N3 = 500
    X = np.zeros((N1 + N2 + N3, 5))
    y = np.zeros((N1 + N2 + N3))
    X[:N1, :] = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0], cov=[[7, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 9, 0], [0, 0, 0, 0, 50]], size=N1)
    X[N1:N1 + N2, :] = np.random.multivariate_normal(mean=[29, 100, 50, 200, 100], cov=[[20, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 30, 0], [0, 0, 0, 0, 0.1]], size=N2)
    X[N1 + N2:N1 + N2 + N3, :] = np.random.multivariate_normal(mean=[-29, -100, -50, -200, 0], cov=[[5, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 4.5, 0, 0], [0, 0, 0, 6.5, 0], [0, 0, 0, 0, 0]], size=N3)

#    gmm = GaussianMixture(n_mixtures=3, covariance_type='full', reg_covar=1e-6, n_iter=100, init_params='kmeans',
#                          weights_init=None, means_init=None, random_state=None, warm_start=True)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, max_iter=100)
    gmm.fit(X)

    centers = gmm.means_
    cov_matrices = gmm.covariances_
    pass


def test_rbfn():

    # Import first two features from iris data set
    X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=2)

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression to
    rbfn = RadialBasisFunctionNetwork(link=1, n_iter=100, n_mixtures=3, covariance_type='full', reg_covar=1e-6,
                                      n_iter_gmm=1, init_params='kmeans', weights_init=None, means_init=None,
                                      random_state=None, l=0.001, n_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Plot train and test data
    plt.figure()
    ax = plt.gca()
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])
    for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
        plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
    for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
        plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
    # Plot gaussians
    centers, cov_matrices = rbfn.get_mixtures()
    for i in range(len(centers)):
        plt.scatter(centers[i][0], centers[i][1], color='black', marker='o', label='versicolor')
    for i in range(len(centers)):
        ellipse = get_ellipse(centers[i], cov_matrices[i, :, :])
        ax.add_patch(ellipse)

    plt.tight_layout()
    plt.show()

    # Make predictions
    y_pred_prob = rbfn.predict_proba(X_test)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    plot_decision_regions(X_train, X_test, y_train, y_test, rbfn, resolution=0.1)
    pass

def mnist():
    np.random.seed(1)
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression
    rbfn = RadialBasisFunctionNetwork(link=1, n_iter=100, n_components=5, covariance_type='full',
                                      feature_type='post_prob', reg_covar=1e-6, n_iter_gmm=1, init_params='kmeans',
                                      weights_init=None, means_init=None, random_state=None, l=0.01, n_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = rbfn.predict_proba(X_test)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

#    log_loss = log_reg_log_loss(y, rbfn.predict_proba(X), 1.0/rbfn.l, rbfn.logReg_.weights_)
    log_loss = log_reg_log_loss(y, rbfn.predict_proba(X), 1.0/rbfn.l, rbfn.logReg_.coef_, rbfn.logReg_.intercept_)

    pass

def wine():
    #np.random.seed(1)
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression
    rbfn = RadialBasisFunctionNetwork(link=0, n_iter=500, n_components=3, covariance_type='full',
                                      feature_type='post_prob', reg_covar=1e-5, n_iter_gmm=1, init_params='kmeans',
                                      weights_init=None, means_init=None, random_state=None, l=1, n_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = rbfn.predict_proba(X_test)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    pass

def cancer():
    #np.random.seed(1)
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression
    rbfn = RadialBasisFunctionNetwork(link=0, n_iter=500, n_components=2, covariance_type='full',
                                      feature_type='post_prob', reg_covar=1e-3, n_iter_gmm=1, init_params='kmeans',
                                      weights_init=None, means_init=None, random_state=None, l=1, n_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = rbfn.predict_proba(X_test)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    pass

def iris():
    #np.random.seed(1)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression
    rbfn = RadialBasisFunctionNetwork(link=0, n_iter=100, n_components=3, covariance_type='full',
                                      feature_type='proj_log_likelihood', reg_covar=1e-3, n_iter_gmm=1, init_params='kmeans',
                                      weights_init=None, means_init=None, random_state=None, l=0.01, n_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = rbfn.predict_proba(X_test)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    pass


if __name__ == '__main__':

    from sklearn.utils.estimator_checks import check_estimator

    def test_classifier():
        return check_estimator(RadialBasisFunctionNetwork)

    #test_classifier()
    #test_logReg_2classes()
    #test_logReg_3classes()
    #test_logReg_blobs()
    #test_gmm()
    #test_gmm_D()
    #test_rbfn()
    #mnist()
    #wine()
    #cancer()
    #iris()

    import numpy as np
    from scipy.misc import logsumexp

    B = np.array([[1e12, 1], [1, 1e25]], dtype=np.longdouble)

    n = 100
    reg = np.ones(n) * 1e20/n
    sum = np.array([0.01])
    c = np.array([0.0])
    for i in range(reg.size):
        y = reg[i] - c    # So far, so good: c is zero.
        t = sum + y         # Alas, sum is big, y small, so low-order digits of y are lost.
        c = (sum - t) - y       # (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        sum = t                 # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!

    pass