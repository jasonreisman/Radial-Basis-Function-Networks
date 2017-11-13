import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

class GaussianMixture(object):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.
    It was created with the intention to use in the Radial Basis Function Networks package.
    Keep in mind that some features were design with the objective to optimize the usage with
    this package.

    Parameters
    ----------
    n_mixtures : int, defaults to 1.
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
    n_iter : int, defaults to 100.
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
    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    Attributes
    ----------
    weights_ : array-like, shape (n_mixtures, )
        The weights of each mixture components.
    means_ : array-like, shape (n_mixtures, n_features)
        The mean of each mixture component.
    covariances_ : array-like, shape (n_mixtures, n_features, n_features)
        The covariance of each mixture component.
    """

    def __init__(self, n_mixtures=1, covariance_type='full', reg_covar=1e-6, n_iter=100, init_params='kmeans',
                 weights_init=None, means_init=None, random_state=None, warm_start=False):
        self.n_mixtures = n_mixtures
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_iter = n_iter
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X, y=None, weights=None):
        """Fits the gaussian mixtures to the data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The data samples to be fit.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        weights : array-like, shape (n_mixtures, ), default = None
            Outside vector giving the mixtures weights.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check parameters (Incomplete!)
        if (weights is not None) and (weights.size != self.n_mixtures):
            raise ValueError("Vector (weights_update) must have lenght equal to n_mixtures; got (weights_update lenght = %r)"
                             % weights.size)

        X = check_array(X)

        # If warm_start is false or the method has not been fit before, initialize the parameters.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            if self.weights_init != None:
                self.weights_ = self.weights_init
            else:
                self.weights_ = np.ones(self.n_mixtures) / self.n_mixtures

            if self.means_init != None:
                self.means_ = self.means_init
            else:
                if self.init_params == 'kmeans':
                    kmeans = KMeans(n_clusters=self.n_mixtures).fit(X)
                    self.means_ = kmeans.cluster_centers_

                elif self.init_params == 'random':
                    self.means_ = X[np.random.randint(X.shape[0], size=self.n_mixtures), :]

            self.covariances_ = np.zeros((self.n_mixtures, X.shape[1], X.shape[1]))
            cov = np.cov(X.T)
            for i in range(self.n_mixtures):
                self.covariances_[i, :, :] = cov

            self.z_ = np.zeros((X.shape[0], self.n_mixtures))

        # Update weights here if weights vector is passed. (This vector may be calculated with the weights of a log reg)
        if weights is not None:
            self.weights_ = weights


        for j in range(self.n_iter):

            post_probs = self.predict_likelihood(X)

            for i in range(self.n_mixtures):

                # upgrade the degrees of belonging to the gauss function (z)
                numerator = post_probs[:, i] * self.weights_[i]
                denominator = np.sum(post_probs * self.weights_, axis=1)  # denominator of z
                z = numerator / denominator
                z = z.reshape((z.size, 1))

                # update mean
                numerator = np.sum(X * z, axis=0)
                denominator = np.sum(z)
                self.means_[i] = numerator / denominator

                # update cov_matrix
                centered_X = (X - self.means_[i])
                numerator = np.dot(centered_X.T, centered_X * z)
                self.covariances_[i, :, :] = numerator / denominator

                # maintains only the elements of the diagonal matrix if the covariance type is diagonal
                if self.covariance_type == 'diag':
                    diag = self.covariances_[i, :, :].diagonal()
                    self.covariances_[i, :, :] = np.diag(diag)

                # Update weights based on the z
                self.weights_[i] = (1 / z.shape[0]) * np.sum(z)

                self.z_[:, i] = z.flat

        return self

    def predict_likelihood(self, X):
        """ Predict likelihood of each component given the data.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        post_probs : array-like of shape = [n_samples, n_mixtures]
            Matrix containing the predicted posterior probabilities for each gaussian in the mixture.
        """

        # Check is fit had been called
        check_is_fitted(self, 'weights_')

        # Input validation
        X = check_array(X)

        post_probs = np.zeros((X.shape[0], self.n_mixtures))

        for i in range(self.n_mixtures):
            try:
                post_probs[:, i] = multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i, :, :])
            except np.linalg.LinAlgError as err:
                if 'singular matrix' in str(err):
                    cov = self.covariances_[i, :, :] + np.eye(self.covariances_[i, :, :].shape[0]) * self.reg_covar  # Add regularization to matrix
                    post_probs[:, i] = multivariate_normal.pdf(X, mean=self.means_[i], cov=cov)
                else:
                    raise

        return post_probs