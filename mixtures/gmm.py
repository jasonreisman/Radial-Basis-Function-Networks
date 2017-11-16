import warnings

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
    n_components : int, defaults to 1.
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
    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.
    means_init : array-like, shape (n_components, n_features), optional
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
    weights_ : array-like, shape (n_components, )
        The weights of each mixture components.
    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : array-like, shape (n_components, n_features, n_features)
        The covariance of each mixture component.
    """

    def __init__(self, n_components=1, covariance_type='full', reg_covar=1e-6, n_iter=100, init_params='kmeans',
                 weights_init=None, means_init=None, random_state=None, warm_start=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_iter = n_iter
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X, y=None, prior_weights=None):
        """Fits the gaussian mixture to the data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The data samples to be fit.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        prior_weights : array-like, shape (n_components, ), default = ones(n_components)
            Weights of the dirichlet prior used in the m step when calculating
            the MAP estimate of the mixture weights.


        Returns
        -------
        self : object
            Returns self.
        """

        # Check parameters (Incomplete!)
        if (prior_weights is not None) and ((prior_weights.size != self.n_components) or (any(weight <= 0 for weight in prior_weights))):
            raise ValueError("Vector prior_weights must have lenght equal to n_components and be all positive; got (prior_weights lenght = %r)"
                             % prior_weights)

        X = check_array(X)

        # If warm_start is false or the method has not been fit before, initialize the parameters.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            if self.weights_init != None:
                self.weights_ = self.weights_init
            else:
                self.weights_ = np.ones(self.n_components) / self.n_components

            if self.means_init != None:
                self.means_ = self.means_init
            else:
                if self.init_params == 'kmeans':
                    kmeans = KMeans(n_clusters=self.n_components).fit(X)
                    self.means_ = kmeans.cluster_centers_

                elif self.init_params == 'random':
                    self.means_ = X[np.random.randint(X.shape[0], size=self.n_components), :]

            self.covariances_ = np.zeros((self.n_components, X.shape[1], X.shape[1]))
            cov = np.cov(X.T) / self.n_components
            for i in range(self.n_components):
                self.covariances_[i, :, :] = cov

        self.prior_weights = prior_weights
        # In case no dirichlet prior was passed then a uniform distribution is assumed and MLE calculated instead
        if self.prior_weights is None:
            self.prior_weights = np.ones(self.n_components)


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
        responsibility : array, shape (n_samples, n_components)
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

        log_prob = np.empty((X.shape[0], self.n_components))

        for i in range(self.n_components):
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
        weights : array, shape (n_components,)

        means : array, shape (n_components, n_features)

        covariances : array-like, shape (n_components, n_features, n_features)
        """

        weights = self._calculate_new_weights()
        means = self._calculate_new_means(X)
        covariances = self._calculate_new_covariances(X, means)

        return weights, means, covariances

    def _calculate_new_weights(self):
        """Updates the weights based on the new responsibilities.

         Returns
         -------
         weights : array, shape (n_components,)
         """

        weights = np.sum(self.resp_, axis=0) + self.prior_weights - 1
        weights = weights / np.sum(weights)

        return weights

    def _calculate_new_means(self, X):
        """Updates the means based on the new responsibilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        means : array, shape (n_components, n_features)
        """
        X_dim = X.shape[1]
        means = np.empty((self.n_components, X_dim))

        for i in range(self.n_components):
            numerator = np.sum(X * self.resp_[:, i].reshape(-1, 1), axis=0)
            denominator = np.sum(self.resp_[:, i])
            means[i, :] = numerator / denominator

        return means

    def _calculate_new_covariances(self, X, means):
        """Updates the covariances based on the new responsibilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        means : array, shape (n_components, n_features)

        Returns
        -------
        covariances : array-like, shape (n_components, n_features, n_features)
        """

        X_dim = X.shape[1]
        covariances = np.empty((self.n_components, X_dim, X_dim))

        for i in range(self.n_components):
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

        gmm = GaussianMixture(n_components=4, covariance_type='full', reg_covar=1e-6, n_iter=10, init_params='kmeans',
                              weights_init=None, means_init=None, random_state=None, warm_start=True)

        gmm.fit(X, prior_weights=np.array([1,1,1,1]))

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


    test_gmm()
