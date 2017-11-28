import numbers

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from utils.one_hot_encoder import OneHotEncoder

def softmax(X):
    """Calculates the softmax of each row of X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_classes)

    Returns
    -------
    y : array-like, shape (n_samples, n_classes)
        Logistic of X.
    """

    exp_X = np.exp(X - np.max(X, axis=1).reshape(-1,1))
    return exp_X / (np.sum(exp_X, axis=1)).reshape((X.shape[0], 1))

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """ Logistic Regression (aka logit, MaxEnt) classifiers.

    This class implements a l2 regularized logistic regression using a
    bound optimization solver.
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

        # Check if there are at least 2 classes
        if self.n_classes_ < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        # Saves training and target arrays
        ones = np.ones((X.shape[0], 1))
        self.X_ = np.hstack([ones, X])  # Ads a first column of 1s (bias feature) to X
        self.y_ = y

        # One hot encodes target
        self.oneHot_ = OneHotEncoder().fit(self.y_)
        y_1hot = self.oneHot_.transform(self.y_)

        # Calculate new weights if warm_start is set to False or the method fit was never ran.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            # shape (n_classes * n_features). Saves the feature weights for each class.
            self.weights_ = np.ones(self.n_classes_ * self.X_.shape[1]) * np.finfo(np.float64).eps

        # Start of bound optimization algorithm

        # B : array-like, shape (n_classes * n_features, n_classes- * n_features)
        #    Negative definite lower bound of the Hessian.
        B = np.kron(
            -0.5 * (np.eye(self.n_classes_) - np.ones((self.n_classes_, self.n_classes_)) / self.n_classes_)
            ,np.dot(self.X_.T, self.X_))

        # denom : array-like, shape ((n_classes-1)*n_features, (n_classes-1)*n_features)
        #    Left side of the weight update step.

#        denom = np.linalg.inv(B - self.l * np.eye(B.shape[0], B.shape[1]))
        B_reg = np.copy(B).astype(np.longdouble)
        reg = np.log(self.l).astype(np.longdouble)
        for i in range(B_reg.shape[0]):
            aux = [reg, np.log(-B[i, i]).astype(np.longdouble)]
            lse = logsumexp(aux)
            B_reg[i, i] = -np.exp(lse)
        denom = np.linalg.inv(B_reg.astype(np.float64))

        for i in range(self.n_iter):
            p = softmax(np.dot(self.X_, self.weights_.reshape(self.X_.shape[1], self.n_classes_, order='F')))
            dif = y_1hot - p
            g = np.zeros((self.n_classes_) * self.X_.shape[1])
            for j in range(self.X_.shape[0]):  # (to be improved)
                g += np.kron(dif[j, :], self.X_[j, :])
            self.weights_ = np.dot(denom, np.dot(B, self.weights_) - g)

        # Return the classifiers
        return self

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

        weights = self.get_weights()

        # Ads a first column of 1s (bias feature) to X
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

        # Calculate probabilities
        probs = softmax(np.dot(X, weights))

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