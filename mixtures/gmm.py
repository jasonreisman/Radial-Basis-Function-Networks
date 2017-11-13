import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

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
            cov = np.cov(X.T) / self.n_mixtures
            for i in range(self.n_mixtures):
                self.covariances_[i, :, :] = cov

        # Update weights here if weights vector is passed. (This vector may be calculated with the weights of a log reg)
        if weights is not None:
            self.weights_ = weights


        for j in range(self.n_iter):

            self.resp_ = self._e_step(X)

            self.weights_, self.means_, self.covariances_ = self._m_step(X)

        return self

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """

        # Check is fit had been called
        check_is_fitted(self, 'resp_')

        log_resp = self._estimate_log_resp(X)
        return np.exp(log_resp)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        responsibility : array, shape (n_samples, n_mixtures)
            Posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """

        responsibility = np.exp(self._estimate_log_resp(X))

        return responsibility

    def _estimate_log_resp(self, X):
        """Estimate log responsibilities for each sample.

        Compute the responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """

        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_responsibilities = weighted_log_prob - logsumexp(weighted_log_prob, axis=1).reshape(-1, 1)

        return log_responsibilities


    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | theta) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """

        weighted_log_prob = self._estimate_log_prob(X) + np.log(self.weights_)

        return weighted_log_prob

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | theta).

        Compute the log-probabilities per each mixture for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """

        log_prob = np.empty((X.shape[0], self.n_mixtures))

        for i in range(self.n_mixtures):
            try:
                log_prob[:, i] = multivariate_normal.logpdf(X, mean=self.means_[i], cov=self.covariances_[i, :, :])
            except np.linalg.LinAlgError as err:
                if 'singular matrix' in str(err):
                    reg_cov = self.covariances_[i, :, :] + np.eye(self.covariances_[i, :, :].shape[0]) * self.reg_covar
                    log_prob[:, i] = multivariate_normal.logpdf(X, mean=self.means_[i], cov=reg_cov)
            except ValueError as err:
                if 'the input matrix must be positive semidefinite' in str(err):
                    reg_cov = self.covariances_[i, :, :] + np.eye(self.covariances_[i, :, :].shape[0]) * self.reg_covar
                    log_prob[:, i] = multivariate_normal.logpdf(X, mean=self.means_[i], cov=reg_cov)
                else:
                    raise

        return log_prob

    def _m_step(self, X):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        weights : array, shape (n_mixtures,)

        means : array, shape (n_mixtures, n_features)

        covariances : array-like, shape (n_mixtures, n_features, n_features)
        """

        weights = self._calculate_new_weights()
        means = self._calculate_new_means(X)
        covariances = self._calculate_new_covariances(X, means)

        return weights, means, covariances

    def _calculate_new_weights(self):
        """Updates the weights based on the new responsibilities.

         Returns
         -------
         weights : array, shape (n_mixtures,)
         """

        n_samples = self.resp_.shape[0]
        weights = np.sum(self.resp_, axis=0) / n_samples

        return weights

    def _calculate_new_means(self, X):
        """Updates the means based on the new responsibilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        means : array, shape (n_mixtures, n_features)
        """
        X_dim = X.shape[1]
        means = np.empty((self.n_mixtures, X_dim))

        for i in range(self.n_mixtures):
            numerator = np.sum(X * self.resp_[:, i].reshape(-1, 1), axis=0)
            denominator = np.sum(self.resp_[:, i])
            means[i, :] = numerator / denominator

        return means

    def _calculate_new_covariances(self, X, means):
        """Updates the covariances based on the new responsibilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        means : array, shape (n_mixtures, n_features)

        Returns
        -------
        covariances : array-like, shape (n_mixtures, n_features, n_features)
        """

        X_dim = X.shape[1]
        covariances = np.empty((self.n_mixtures, X_dim, X_dim))

        for i in range(self.n_mixtures):
            centered_X = (X - means[i, :])
            numerator = np.dot(centered_X.T, centered_X * self.resp_[:, i].reshape(-1, 1))
            covariances[i, :, :] = numerator / np.sum(self.resp_[:, i])

            # Maintains only the elements of the diagonal matrix if the covariance type is diagonal
            if self.covariance_type == 'diag':
                diag = covariances[i, :, :].diagonal()
                covariances[i, :, :] = np.diag(diag)

        return covariances

if __name__ == '__main__':

    from sklearn import datasets
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.model_selection import train_test_split


    # returns an Ellipse object when given a center and covariance matrix
    def get_ellipse(mean, cov):

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 4 * np.sqrt(vals)
        ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False)

        return ellip


    np.random.seed(1)
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    gmm = GaussianMixture(n_mixtures=10, covariance_type='full', reg_covar=1e-6, n_iter=100, init_params='kmeans',
                 weights_init=None, means_init=None, random_state=None, warm_start=False)

    gmm.fit(X_train)
    probs = gmm.predict_proba(X_test)
    pass
