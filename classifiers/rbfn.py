import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from classifiers.logisitc_regression import LogisticRegression
from mixtures.gmm import GaussianMixture


class RadialBasisFunctionNetwork(BaseEstimator, ClassifierMixin):
    """ Implements a Radial Basis Function Network classifier.

    Uses Gaussian Mixture Models for the Radial Functions. The way of training allows
    a link between the EM step of the GMM and the Logistic Regression, where the weights
    of the Logistic Regression are to be used for the training of the Gaussian Mixture Models.

    Parameters
    ----------
    link : integer
        Proportion of log reg weights used for the mixture weights.
        0 : no log reg weights.
        1 : only log reg weights
    n_iter : int, defaults to 100
        Number of iterations performed in method fit.

    (for the Gaussian Mixture Models)
    n_mixtures : int, defaults to 2.
        The number of mixture components.
    covariance_type : {'full', 'diag'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive definite.
    n_iter_gmm : int, defaults to 1.
        The number of EM iterations to perform.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
    weights_init : array-like, shape (n_mixtures, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.
    means_init : array-like, shape (n_mixtures, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    (For the Logistic Regression)
    l : float, default: 0.01
        Regularization strength; must be a positive float.
        Bigger values specify stronger regularization.
    n_iter_logreg : int, default: 1
        Number of iterations performed by the optimization algorithm
        in search for the optimal weights.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :method:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :method:`fit`
    """


    def __init__(self, link=True, n_iter=100, n_mixtures=5, covariance_type='full', reg_covar=1e-6, n_iter_gmm=1,
                 init_params='kmeans', weights_init=None, means_init=None, random_state=None, l=0.01, n_iter_logreg=1):
        self.link = link
        self.n_iter = n_iter
        self.n_mixtures = n_mixtures
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_iter_gmm = n_iter_gmm
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.random_state = random_state
        self.l = l
        self.n_iter_logreg = n_iter_logreg

    def fit(self, X, y):
        """Fit a Radial Basis Function Network classifier to the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the number of classes
        self.n_classes_ = self.classes_.size

        # Check link parameter
        if not isinstance(self.link, numbers.Number) or self.link < 0 or self.link > 1:
            raise ValueError("link must be a number in the interval [0,1]. Valor passed: %r" % self.link)

        # Check if there are at least 2 classes
        if self.n_classes_ < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        self.X_ = X
        self.y_ = y

        self.gmm_ = GaussianMixture(n_mixtures=self.n_mixtures, covariance_type=self.covariance_type,
                              reg_covar=self.reg_covar, n_iter=self.n_iter_gmm, init_params=self.init_params,
                              weights_init=self.weights_init, means_init=self.means_init,
                              random_state=self.random_state, warm_start=True)
        self.logReg_ = LogisticRegression(l=self.l, n_iter=self.n_iter_logreg)


        if self.link > 0:

            gmm_weights = np.ones(self.n_mixtures) / self.n_mixtures
            for i in range(self.n_iter):
                self.gmm_ = self.gmm_.fit(X=self.X_, weights=gmm_weights)
                design_matrix = self.gmm_.predict_proba(X=self.X_)
                self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

                logReg_weights = self.logReg_.get_weights()
                gmm_weights = self.gmm_.weights_
                gmm_weights = self.calculate_gmm_weights(logReg_weights[1:, :], gmm_weights)

        else:

            for i in range(self.n_iter):
                self.gmm_ = self.gmm_.fit(X=self.X_, weights=None)
                design_matrix = self.gmm_.predict_proba(X=self.X_)
                self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

        # Return the classifier
        return self

    def get_mixtures(self):
        """Returns the centers and co-variance matrices of the GMM.

        Returns
        -------
        centers : array, shape (n_mixtures,)
            Centers of the GMM.
        cov_matrices : array-type, shape (.n_mixtures, n_features, n_features)
            Co-variance matrices of the GMM.
        """

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        centers = self.gmm_.means_
        cov_matrices = self.gmm_.covariances_

        return centers, cov_matrices

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

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        X_likelihood = self.gmm_.predict_proba(X=X)
        probs = self.logReg_.predict_proba(X_likelihood)

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

        # Input validation
        X = check_array(X)

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        X_likelihood = self.gmm_.predict_proba(X=X)
        y = self.logReg_.predict(X_likelihood)

        return y


    def calculate_gmm_weights(self, logReg_weights, gmm_weights):
        """ Calculates the weights to be passed to the GMM.

        The result is proportional to the parameter link and
        is based based on the weights of the Log Reg and the degrees
        of belonging of the points to the mixtures.

            Parameters
            ----------
            logReg_weights :  array, shape (1, n_features) or (n_classes, n_features)
                Coefficient of the features in the decision function.
                logReg_weights is of shape (1, n_features) when the given problem is binary.
            gmm_weights : array-like, shape (n_components,)
                The weights of each mixture components.

            Returns
            -------
            gmm_weights : array, shape (n_features, )
                Weights for the GMM.
            """

        # Weights based on the log reg weights.
        logReg_weights_abs = np.absolute(logReg_weights)
        gmm_weights_logReg = np.sum(logReg_weights_abs, axis=1) / np.sum(logReg_weights_abs)

        gmm_weights = (1 - self.link) * gmm_weights + self.link * gmm_weights_logReg

        return gmm_weights