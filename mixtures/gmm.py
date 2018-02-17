import warnings
import time

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

from utils.stats import mult_gauss_pdf, log_multivariate_normal_density_diag, log_multivariate_normal_density_full

class GaussianMixture(object):
    """Gaussian Mixture.
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.
    It was created with the intention to use in the Radial Basis Function Networks package.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.
    covariance_type : {'full', 'diag'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix)
    equal_covariances : {True, False},
            defaults to False.
            If True in each mixture every component has the same covariance.
    gauss_kill : {True, False},
        defaults to False.
        If True when a prior of the gmm weights is 0 the corresponding
        component weight will be set entirely to 0.
    n_comp_mixt : array, shape = [n_mixtures, ]
        Number of features to be used by each parcell of the softmax.
        Also number of weights per class.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive definite.
    max_iter : int, defaults to 100.
        The max number of EM iterations to perform.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of:
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
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
    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    """

    def __init__(self, n_components=1, covariance_type='full', equal_covariances=False, gauss_kill=False,
                 reg_covar=1e-6, max_iter=100, init_params='kmeans', random_state=None, warm_start=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.equal_covariances = equal_covariances
        self.gauss_kill = gauss_kill
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
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

        # Check type parameter
        if self.covariance_type not in ['full', 'diag']:
            raise ValueError("covariance_type must be a string contained in ['full', 'diag']. Valor passed: "
                             "%s" % self.covariance_type)

        self.X_ = check_array(X)

        # If warm_start is false or the method has not been fit before, initialize the parameters.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            self.weights_ = np.ones(self.n_components) / self.n_components

            if self.init_params == 'kmeans':
                kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state).fit(X)
                self.means_ = kmeans.cluster_centers_

            elif self.init_params == 'random':
                self.means_ = X[np.random.randint(X.shape[0], size=self.n_components), :]

            if self.covariance_type == "full":
                self.covariances_ = np.zeros((self.n_components, X.shape[1], X.shape[1]))
                self.covariances_[range(self.n_components), :, :] = np.eye(X.shape[1])
            else:
                self.covariances_ = np.zeros((self.n_components, X.shape[1]))
                self.covariances_[range(self.n_components), :] = np.ones(X.shape[1])

        self.prior_weights = prior_weights

        for j in range(self.max_iter):

            if self.n_components == 0:
                self.old_weights = self.weights_.copy()
                break


            self.resp_ = self._e_step(X)

            self._m_step(X)

            # Works only for when self.max_iter = 1
            self.old_weights = self.weights_.copy()
            if 0 in self.weights_:
                self.del_0comp()

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

        weighted_log_prob = self._estimate_log_prob(X) + np.log(self.weights_ + np.finfo(np.float64).eps)

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

        if self.covariance_type == "full":
            log_prob = log_multivariate_normal_density_full(X, means=self.means_, covars=self.covariances_, reg=self.reg_covar)
        else:
            log_prob = log_multivariate_normal_density_diag(X, means=self.means_, covars=self.covariances_, reg=self.reg_covar)

        return log_prob


    def _estimate_log_prob_old(self, X):
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

        if self.covariance_type == "full":
            for i in range(self.n_components):
                try:
                    log_prob[:, i] = mult_gauss_pdf(X, mean=self.means_[i], cov=self.covariances_[i, :, :], log=True)
                except ValueError as err:
                    if 'singular matrix' in str(err):
                        reg_cov = self.covariances_[i, :, :] + np.eye(self.covariances_[i, :, :].shape[0]) * self.reg_covar
                        log_prob[:, i] = mult_gauss_pdf(X, mean=self.means_[i], cov=reg_cov, log=True)
                    else:
                        raise

        else:
            for i in range(self.n_components):
                try:
                    log_prob[:, i] = mult_gauss_pdf(X, mean=self.means_[i], cov=self.covariances_[i, :], log=True)
                except ValueError as err:
                    if 'singular matrix' in str(err):
                        reg_cov = self.covariances_[i, :] + self.reg_covar
                        log_prob[:, i] = mult_gauss_pdf(X, mean=self.means_[i], cov=reg_cov, log=True)
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
        self.weights_ = self._calculate_new_weights()
        self.means_ = self._calculate_new_means(X)
        self.covariances_ = self._calculate_new_covariances(X)

        return

    def _calculate_new_weights(self):
        """Updates the weights based on the new responsibilities.
         Returns
         -------
         weights : array, shape (n_components,)
         """
        resp_components = np.sum(self.resp_, axis=0)
        weights = resp_components.copy()

        if self.prior_weights is not None:
            weights += self.prior_weights - 1
            weights[resp_components == 0] = 0 # ignore prior if responsibility of component is 0
            if self.gauss_kill is True:
                weights[self.prior_weights == 1] = 0

        weights = weights / (np.sum(weights) + np.finfo(np.float64).eps)

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
            if self.weights_[i] == 0:
                continue

            numerator = np.sum(X * self.resp_[:, i].reshape(-1, 1), axis=0)
            denominator = np.sum(self.resp_[:, i])
            means[i, :] = numerator / denominator

        return means

    def _calculate_new_covariances(self, X):
        """Updates the covariances based on the new responsibilities.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        means : array, shape (n_components, n_features)
        Returns
        -------
        covariances : array-like, shape (n_components, n_features, n_features)
        """

        # Investigar se é possivel optimizar isto para fazer sem ciclos

        X_dim = X.shape[1]

        if self.covariance_type == "full":
            covariances = np.empty((self.n_components, X_dim, X_dim))
            mean_cov = np.zeros((X_dim, X_dim))
            for i in range(self.n_components):
                if self.weights_[i] == 0:
                    continue

                centered_X = (X - self.means_[i, :])
                numerator = np.dot(centered_X.T, centered_X * self.resp_[:, i].reshape(-1, 1))
                covariances[i, :, :] = numerator / np.sum(self.resp_[:, i])

                if self.equal_covariances is True:
                    mean_cov += self.weights_[i] * covariances[i, :, :]

            if self.equal_covariances is True:
                covariances[range(self.n_components), :, :] = mean_cov
        else:
            covariances = np.empty((self.n_components, X_dim))
            mean_cov = np.zeros(X_dim)
            for i in range(self.n_components):
                if self.weights_[i] == 0:
                    continue

                centered_X = (X - self.means_[i, :])
                numerator = np.sum(centered_X.T * (centered_X * self.resp_[:, i].reshape(-1, 1)).T, axis=1)
                covariances[i, :] = numerator / np.sum(self.resp_[:, i])

                if self.equal_covariances is True:
                    mean_cov += self.weights_[i] * covariances[i, :]

            if self.equal_covariances is True:
                covariances[range(self.n_components), :] = mean_cov

        return covariances

    def del_0comp(self):

        ind = np.argwhere(self.weights_ == 0)
        self.n_components -= ind.size
        self.weights_ = np.delete(self.weights_, ind)
        self.means_ = np.delete(self.means_, ind, axis=0)
        self.covariances_ = np.delete(self.covariances_, ind, axis=0)
        self.resp_ = np.delete(self.resp_, ind, axis=1)

        return




    def get_covariances(self):

        if self.covariance_type == "full":
            return self.covariances_
        else:
            covariances = np.empty((self.n_components, self.X_.shape[1], self.X_.shape[1]))
            for i in range(self.n_components):
                covariances[i, :, :] = np.diag(self.covariances_[i,:])

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
        X = np.zeros((N1 + N2, 2))
        y = np.zeros((N1 + N2))
        X[:N1, :] = np.random.multivariate_normal(mean=[0, 0], cov=[[7, -3], [-3, 8]], size=N1)
        X[N1:N1 + N2, :] = np.random.multivariate_normal(mean=[5, 5], cov=[[7, -3], [-3, 8]], size=N2)

        gmm = GaussianMixture(n_components=2, covariance_type='diag', equal_covariances=False, reg_covar=1e-6, max_iter=100, init_params='kmeans',
                              weights_init=None, means_init=None, random_state=None, warm_start=True)

        gmm.fit(X, prior_weights=np.array([1,1]))

        centers = gmm.means_
        cov_matrices = gmm.get_covariances()

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
