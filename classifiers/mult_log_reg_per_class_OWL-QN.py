import numbers

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from utils.one_hot_encoder import OneHotEncoder
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b

class LogisticRegressionperClass(BaseEstimator, ClassifierMixin):
    """ Multinomial Logistic Regression classifier with specific features for each class.

    This class implements a l2 regularized multinomial logistic regression using a
    bound optimization solver.
    This classifier diverges from the classical mult log reg by dividing the dataset by features
    in the number of classes and uses each group of features only to a parcel of the softmax.
    It was created with the intention to use in the Radial Basis Function Networks package.
    Keep in mind that some features were design with the objective to optimize the usage with
    this package.

    Parameters
    ----------
    l : float, default: 0.01
        Regularization strength; must be a positive float.
        Bigger values specify stronger regularization.

    n_iter : int, default: 1000
        Number of iterations performed by the optimization algorithm
        in search for the optimal weights.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to
        fit as initialization, otherwise, just erase the previous solution.
        Set this to true when the log reg is to be trained in a similar train
        and target arrays (obliged to have same shapes as previous).
    """

    def __init__(self, l=0.01, n_iter=1000, warm_start=False):
        self.l = l
        self.n_iter = n_iter
        self.warm_start = warm_start

    def fit(self, X, y):  # verificar se y está one-hot encoded, se l e n_iter são maiores que 0
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check if the initialized parameters are correct
        if not isinstance(self.l, numbers.Number) or self.l < 0:
            raise ValueError("Penalty (l) term must be positive; got (l=%r)"
                             % self.l)
        if not isinstance(self.n_iter, numbers.Number) or self.n_iter < 0:
            raise ValueError("Maximum number of iterations (max_iter) must be positive;"
                             " got (max_iter=%r)" % self.n_iter)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit by extracting an ordered array of unique labels from target
        self.classes_ = unique_labels(y)
        # Store the number of classes
        self.n_classes_ = self.classes_.size

        X_dim = X.shape[1]

        # Check if there are at least 2 classes
        if self.n_classes_ < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        # If features_per_class=True but the number of features is not a multiple of the number of classes throw error.
        if X_dim % self.n_classes_ != 0:
            raise ValueError("The number of features of X must be a multiple of the number of classes when "
                             "features_per_class=True;")

        # Saves training and target arrays
        # Ads n_classes columns of 1s (bias feature) to X
        self.X_ = np.ones((X.shape[0], 1))
        for i in range(self.n_classes_):
            aux = X[:, i * int(X_dim/self.n_classes_): (i+1) * int(X_dim/self.n_classes_)]
            ones = np.ones((X.shape[0], 1))
            self.X_ = np.hstack([self.X_, aux, ones])
        self.X_ = self.X_[:, :-1]
        self.y_ = y

        # One hot encodes target
        self.oneHot_ = OneHotEncoder().fit(self.y_)
        self.y_1hot = self.oneHot_.transform(self.y_)

        # Calculate new weights if warm_start is set to False or the method fit was never ran.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            # shape (n_features). Saves the feature weights for each class.
            self.weights_ = np.ones(self.X_.shape[1]) * np.finfo(np.float64).eps


        # Start of bound optimization algorithm

        cost = self.cost(self.weights_)
        gradient = self.gradient(self.weights_)

        #res = minimize(fun=self.cost, x0=self.weights_, method='BFGS', jac=self.gradient, options = {'maxiter' : self.n_iter, 'disp': False})
        res = fmin_l_bfgs_b(func=self.cost, x0=self.weights_, fprime=self.gradient, bounds=None, iprint=-1, maxiter=15000)
        self.weights_ = res.x

        # Return the classifiers
        return self

    def cost(self, weights):

        X_ = np.zeros((self.X_.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            X_[:, i] = np.dot(self.X_[:, i * int(self.X_.shape[1]/self.n_classes_): (i+1) * int(self.X_.shape[1]/self.n_classes_)],
                             weights[i * int(self.X_.shape[1]/self.n_classes_): (i+1) * int(self.X_.shape[1]/self.n_classes_)])


        exp_X = np.exp(X_ - np.max(X_, axis=1).reshape(-1, 1))
        softmax = exp_X / (np.sum(exp_X, axis=1)).reshape((self.X_.shape[0], 1))

        cost_ = np.log(softmax + np.finfo(np.float64).eps) * self.y_1hot

        cost = - np.sum(cost_) + self.l * np.dot(weights, weights)

        return cost

    def gradient(self, weights):

        X_ = np.zeros((self.X_.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            X_[:, i] = np.dot(self.X_[:, i * int(self.X_.shape[1]/self.n_classes_): (i+1) * int(self.X_.shape[1]/self.n_classes_)],
                             weights[i * int(self.X_.shape[1]/self.n_classes_): (i+1) * int(self.X_.shape[1]/self.n_classes_)])

        exp_X = np.exp(X_ - np.max(X_, axis=1).reshape(-1, 1))
        softmax = exp_X / (np.sum(exp_X, axis=1)).reshape((self.X_.shape[0], 1))

        dif = self.y_1hot - softmax

        g = np.zeros(self.X_.shape[1], dtype=np.longdouble)
        for j in range(self.n_classes_):
            g_aux = dif[:, j].reshape(dif.shape[0], 1) * self.X_[:, j * int(self.X_.shape[1] / self.n_classes_):
            (j + 1) * int(self.X_.shape[1] / self.n_classes_)]
            g[j * int(self.X_.shape[1] / self.n_classes_): (j + 1) * int(self.X_.shape[1] / self.n_classes_)] \
                = np.sum(g_aux, axis=0)

        aux = - g + self.l * weights
        return - g + self.l * weights

    def hessian(self, weights):

        H = -0.5 * (1 - 1 / self.n_classes_) * np.dot(self.X_.T, self.X_)

        return H

    def softmax(self, X):
        """Calculates the softmax of each row of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Design matrix.

        Returns
        -------
        y : array-like, shape (n_samples, n_classes)
            Logistic of X.
        """

        X_ = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            X_[:, i] = np.dot(X[:, i * int(X.shape[1]/self.n_classes_): (i+1) * int(X.shape[1]/self.n_classes_)],
                             self.weights_[i * int(X.shape[1]/self.n_classes_): (i+1) * int(X.shape[1]/self.n_classes_)])


        exp_X = np.exp(X_ - np.max(X_, axis=1).reshape(-1, 1))
        return exp_X / (np.sum(exp_X, axis=1)).reshape((X.shape[0], 1))

    def get_weights(self):
        """ Returns the feature weights of the classifiers.

        Returns
        -------
        w : array-like of shape = [n_features+1, n_classes-1]
            Feature weights of the classifiers.
        """

        # Check if fit had been called
        check_is_fitted(self, 'weights_')

        w = self.weights_.reshape(self.X_.shape[1], self.n_classes_, order='F')

        return w

    def predict_proba(self, X):
        """ Predict the probabilities of the data belonging to each class.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        probs : array of int of shape = [n_samples, n_classes]
            Contains the probabilities of each sample belonging to a class.
        """

        # Input validation
        X = check_array(X)

        # Ads the columns of 1s
        X_ = np.ones((X.shape[0], 1))
        for i in range(self.n_classes_):
            aux = X[:, i * int(X.shape[1]/self.n_classes_): (i+1) * int(X.shape[1]/self.n_classes_)]
            ones = np.ones((X.shape[0], 1))
            X_ = np.hstack([X_, aux, ones])
        X_ = X_[:, :-1]

        # Calculate probabilities
        probs = self.softmax(X_)

        return probs

    def predict(self, X):
        """ Predict the classes each sample belongs to.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples,]
            Contains the predicted classes.
        """

        # Get a matrix with the probabilities of a sample belonging to each class.
        probs = self.predict_proba(X)

        # Get the predicted classes by choosing the class which has biggest probability.
        y_ = np.argmax(probs, axis=1)

        # Get the original class ints before one hot encoding
        y = self.oneHot_.retransform(y_)

        return y

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    #np.random.seed(1)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = iris.data[iris.target != 2, :]
    y = iris.target[iris.target != 2]
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create and fit the Logistic Regression
    logReg = LogisticRegressionperClass(l=0.01, n_iter=5, warm_start=False)
    logReg.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    pass