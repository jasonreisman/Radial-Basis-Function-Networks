import numbers

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from utils.one_hot_encoder import OneHotEncoder
from scipy.optimize import fmin_l_bfgs_b

class LogisticRegressionperClass(BaseEstimator, ClassifierMixin):
    """ Multinomial Logistic Regression classifier with specific features for each class.

    This class implements a l2 and l1 regularized multinomial logistic regression.
    This classifier diverges from the classical mult log reg by dividing the dataset by features
    in the number of classes and uses each group of features only to a parcel of the softmax.
    It was created with the intention to use in the Radial Basis Function Networks package.
    Keep in mind that some features were design with the objective to optimize the usage with
    this package.

    Parameters
    ----------
    l1 : float, default: 0.01
        l1 regularization strength; must be a positive float.
        Bigger values specify stronger regularization.

    l2 : float, default: 0.01
        l2 regularization strength; must be a positive float.
        Bigger values specify stronger regularization.

    max_iter : int, default: 15000
        Number of iterations performed by the optimization algorithm
        in search for the optimal weights.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to
        fit as initialization, otherwise, just erase the previous solution.
        Set this to true when the log reg is to be trained in a similar train
        and target arrays (obliged to have same shapes as previous).
    """

    def __init__(self, l1=0.01, l2=0.01, max_iter=15000, warm_start=False):
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.warm_start = warm_start

    def fit(self, X, y, n_weights_per_class=None, weights2kill=None,):  # verificar se y está one-hot encoded, se l e max_iter são maiores que 0
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        n_weights_per_class : array-like, shape = (n_classes,)
            Number of weights for each class.
            The sum of ths vector is the total number of variables.

        weights2kill : array-like, shape = (n_weights in past iteration,)
            Weights of gaussian mixtures components in past iteration.
            The positions with 0 are to be erased in the gmm parameters.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check if the initialized parameters are correct
        if not isinstance(self.l1, numbers.Number) or self.l1 < 0:
            raise ValueError("Penalty (l1) term must be positive; got (l1=%r)" % self.l1)

        if not isinstance(self.l2, numbers.Number) or self.l2 < 0:
            raise ValueError("Penalty (l2) term must be positive; got (l2=%r)" % self.l2)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iterations (max_iter) must be positive; got (max_iter=%r)"
                             % self.max_iter)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, ensure_min_features=0)


        # Store the classes seen during fit by extracting an ordered array of unique labels from target
        self.classes_ = unique_labels(y)
        # Store the number of classes
        self.n_classes_ = self.classes_.size

        n_samples, n_features = X.shape

        # Check if there are at least 2 classes
        if self.n_classes_ < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        self.n_weights_per_class = n_weights_per_class

        # Saves training and target arrays
        # Ads n_classes columns of 1s (bias feature) to X
        X_ = np.ones((n_samples, 1))
        old_ind = 0
        for n in n_weights_per_class:
            new_ind = int(old_ind + n)
            aux = X[:, old_ind: new_ind]
            ones = np.ones((n_samples, 1))
            X_ = np.hstack([X_, aux, ones])
            old_ind = new_ind
        X_ = X_[:, :-1]

        n_samples, n_features = X_.shape

        # One hot encodes target
        self.oneHot_ = OneHotEncoder().fit(y)
        y_1hot = self.oneHot_.transform(y)

        # Calculate initializing weights if warm_start is set to False or the method fit was never ran.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):
            self.x0 = np.ones(n_features * 2) * np.finfo(np.float64).eps

        # Delete from the last iteration parameters the ones that were created by discarded gmm components
        else:
            ind = np.argwhere(weights2kill == 0)
            ind = np.append(ind, ind + weights2kill.size)
            self.x0 = np.delete(self.x0, ind)

        # Optimization of the weights
        w, f, d = fmin_l_bfgs_b(func=self.logistic_loss_grad, x0=self.x0, fprime=None, bounds=[(0, None)] * n_features * 2,
                                args=(X_, y_1hot, self.l1, self.l2), maxiter=self.max_iter)
        self.x0 = w
        self.weights_ = w[:n_features] - w[n_features:]

        # Return the classifiers
        return self

    def logistic_loss_grad(self, w_extended, X, y_1hot, l1, l2):

        n_samples, n_features = X.shape
        w = w_extended[:n_features] - w_extended[n_features:]

        X_ = np.empty((n_samples, self.n_classes_))
        old_ind = 0
        for i, n in enumerate(self.n_weights_per_class):
            new_ind = int(old_ind + n + 1)
            X_[:, i] = np.dot(X[:, old_ind: new_ind], w[old_ind: new_ind])
            old_ind = new_ind

        exp_X = np.exp(X_ - np.max(X_, axis=1).reshape(-1, 1))
        softmax = exp_X / (np.sum(exp_X, axis=1)).reshape((n_samples, 1))

        cost = - np.sum(np.log(softmax + np.finfo(np.float64).eps) * y_1hot)

        cost += l1 * np.sum(w_extended) # l1 regularization
        cost += l2 * np.dot(w, w)  # l2 regularization

        dif = y_1hot - softmax

        grad = np.empty(n_features)
        old_ind = 0
        for i, n in enumerate(self.n_weights_per_class):
            new_ind = int(old_ind + n + 1)
            g_aux = dif[:, i].reshape(dif.shape[0], 1) * X[:, old_ind: new_ind]
            grad[old_ind: new_ind] = - np.sum(g_aux, axis=0)
            old_ind = new_ind

        grad = np.concatenate([grad, -grad])
        grad += self.l1 # l1 regularization
        grad += self.l2 * w_extended # l2 regularization

        return cost, grad

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

        n_samples, _ = X.shape

        X_ = np.empty((n_samples, self.n_classes_))
        old_ind = 0
        for i, n in enumerate(self.n_weights_per_class):
            new_ind = int(old_ind + n + 1)
            X_[:, i] = np.dot(X[:, old_ind: new_ind], self.weights_[old_ind: new_ind])
            old_ind = new_ind

        exp_X = np.exp(X_ - np.max(X_, axis=1).reshape(-1, 1))
        softmax = exp_X / (np.sum(exp_X, axis=1)).reshape((n_samples, 1))

        return softmax

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
        X = check_array(X, ensure_min_features=0)

        n_samples, n_features = X.shape

        # Ads n_classes columns of 1s (bias feature) to X
        X_ = np.ones((n_samples, 1))
        old_ind = 0
        for n in self.n_weights_per_class:
            new_ind = int(old_ind + n)
            aux = X[:, old_ind: new_ind]
            ones = np.ones((n_samples, 1))
            X_ = np.hstack([X_, aux, ones])
            old_ind = new_ind
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
    logReg = LogisticRegressionperClass(l1=0.01, l2=0.01, max_iter=5, warm_start=False)
    logReg.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = logReg.predict_proba(X_test)
    y_pred = logReg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    pass