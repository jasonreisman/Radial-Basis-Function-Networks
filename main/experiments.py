import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.patches import Ellipse
from classifiers.rbfn import RadialBasisFunctionNetwork

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
        plt.figure()
        ax = plt.gca()
        Zi = Z[:, i].reshape(xx1.shape)
        labels = (0.25, 0.5, 0.85)
        cp = plt.contour(xx1, xx2, Zi, alpha=0.4, levels=labels, cmap=ListedColormap(('gray', 'orange', 'red')))
        plt.clabel(cp, inline=True, fontsize=10)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for i in range(len(labels)):
            cp.collections[i].set_label(labels[i])

        # plot train samples
        for idx, cl in enumerate(np.sort(np.unique(y_train))):
            plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
        # plot test samples
#        for idx, cl in enumerate(np.sort(np.unique(y_test))):
#            plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')

        plt.legend(loc='upper right')
        plt.tight_layout()
    plt.show()

def test_rbfn():

    # Import first two features from iris data set and first two classes
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7331)

    # Create and fit the Logistic Regression to
    rbfn = RadialBasisFunctionNetwork(link=0, max_iter=1000, tol=1e-3, n_components=3, feature_type='post_prob',
                                      covariance_type='full',
                                      equal_covariances=False, component_kill=True, ind_cat_features=(),
                                      laplace_smoothing=0.001,
                                      reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans', random_state=None, l1=0.01,
                                      l2=0.01,
                                      max_iter_logreg=1)
    rbfn.fit(X_train, y_train)

    # Plot train and test data
    centers_, cov_matrices_ = rbfn.get_mixtures()
    for j in range(len(np.unique(y_train))):
        plt.figure()
        ax = plt.gca()
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y_train))])
        for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
            plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
        for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
            plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
        # Plot gaussians
        centers = centers_[j]
        cov_matrices = cov_matrices_[j]
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

if __name__ == '__main__':

    test_rbfn()