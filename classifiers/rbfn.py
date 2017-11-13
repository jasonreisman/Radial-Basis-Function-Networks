import numbers

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal

from classifiers.logisitc_regression import LogisticRegression
#from sklearn.linear_model import LogisticRegression
from mixtures.gmm import GaussianMixture
#from sklearn.mixture import GaussianMixture


def forward(z):
    """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    z = z - np.max(z, axis=1).reshape(-1, 1)

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]


    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)

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
        self.logReg_ = LogisticRegression(l=self.l, n_iter=self.n_iter_logreg, warm_start=True)

        gmm_weights = np.ones(self.n_mixtures) / self.n_mixtures
        for i in range(self.n_iter):

            if True:
                self.gmm_ = self.gmm_.fit(X=self.X_, weights=gmm_weights)
                design_matrix = self.gmm_.resp_
                self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

            if False:
                self.gmm_ = self.gmm_.fit(X=self.X_, weights=gmm_weights)
#                design_matrix = self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_)
                design_matrix = np.log(self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_) + np.finfo(np.float64).eps)
                self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

            if False:
                self.gmm_ = self.gmm_.fit(X=self.X_, weights=gmm_weights)
#                likelihoods = self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_)
                likelihoods = np.log(self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_) + np.finfo(np.float64).eps)
                design_matrix = forward(likelihoods)
                self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

            logReg_weights = self.logReg_.get_weights()
            gmm_weights = self.gmm_.weights_
            gmm_weights = self.calculate_gmm_weights(logReg_weights[1:, :], gmm_weights)

        # Return the classifier
        return self

    def predict_likelihoods(self, X, means, covariances):
        """ Predict likelihood of each sample given the mixtures.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        means : array-like, shape (n_components, n_features)
            The mean of each mixture component.
        covariances : array-like
            The covariance of each mixture component. The shape depends on covariance_type:
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

        Returns
        -------
        post_probs : array-like of shape = [n_samples, n_mixtures]
            Matrix containing the likelihoods for each gaussian in the mixture.
        """

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        # Input validation
        X = check_array(X)

        post_probs = np.zeros((X.shape[0], means.shape[0]))

        for i in range(means.shape[0]):
            try:
                post_probs[:, i] = multivariate_normal.pdf(X, mean=means[i, :], cov=covariances[i, :, :])
            except np.linalg.LinAlgError as err:
                if 'singular matrix' in str(err):
                    cov = covariances[i, :, :] + np.eye(covariances[i, :, :].shape[0]) * self.reg_covar  # Add regularization to matrix
                    post_probs[:, i] = multivariate_normal.pdf(X, mean=means[i, :], cov=cov)
                else:
                    raise

        return post_probs #np.log(post_probs + np.finfo(np.float64).eps)

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

        gmm_probs = self.gmm_.predict_proba(X)
        probs = self.logReg_.predict_proba(gmm_probs)

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


        gmm_probs = self.gmm_.predict_proba(X)
        y = self.logReg_.predict(gmm_probs)

#        gmm_like = np.log(self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_) + np.finfo(np.float64).eps)
#        y = self.logReg_.predict(gmm_like)

#        likelihoods = self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_)
#        likelihoods = np.log(self.predict_likelihoods(X, self.gmm_.means_, self.gmm_.covariances_) + np.finfo(np.float64).eps)
#        likelihoods_proj = forward(likelihoods)
#        y = self.logReg_.predict(likelihoods_proj)

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

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    def mnist():
        #np.random.seed(1)
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create and fit the Logistic Regression
        rbfn = RadialBasisFunctionNetwork(link=0, n_iter=100, n_mixtures=10, covariance_type='full', reg_covar=1e-6,
                                          n_iter_gmm=1, init_params='kmeans', weights_init=None, means_init=None,
                                          random_state=None, l=0.01, n_iter_logreg=1)
        rbfn.fit(X_train, y_train)

        # Make predictions
        y_pred_prob = rbfn.predict_proba(X_test)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        pass

    mnist()