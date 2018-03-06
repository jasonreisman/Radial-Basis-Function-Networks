import warnings
import time

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from utils.one_hot_encoder import OneHotEncoder

from utils.stats import mult_gauss_pdf, log_multivariate_normal_density_diag, log_multivariate_normal_density_full

class GaussMultMixture(object):
    """Gaussian and Multinoulli Mixture.
    Representation of a Gaussian and Multinoulli mixture model probability distribution.
    This class allows to estimate the parameters of a mixture distribution for a mixture of
    real and categorical features.
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
    component_kill : {True, False},
        defaults to False.
        If True when a prior of the mm weights is 0 the corresponding
        component weight will be set entirely to 0.
    laplace_smoothing : float, defalt = 0.01
        Constant responsible for the laplace smoothing in the MAP estimation of the categorical weights.
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

    def __init__(self, n_components=1, covariance_type='full', equal_covariances=False, component_kill=False,
                 laplace_smoothing=0.01, reg_covar=1e-6, max_iter=100, init_params='kmeans',
                 random_state=None, warm_start=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.equal_covariances = equal_covariances
        self.component_kill = component_kill
        self.laplace_smoothing = laplace_smoothing
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X_real, X_categorical_1hot, n_categories=np.array([]), y=None, prior_weights=None):
        """Fits the gaussian mixture to the data.
        Parameters
        ----------
        X_real : array-like of shape = [n_samples, n_real_features]
            The input real samples to be fit.
        X_categorical_1hot : array-like of shape = [n_samples, n_categorical_features]
            The input categorical samples one-hot encoded to be fit.
        n_categories : array-like of shape = [n_categorical_features,]
            Array that contains how many categories each categorical feature has.
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

        self.X_real = check_array(X_real, ensure_min_features=0)
        self.X_categorical_1hot = check_array(X_categorical_1hot, ensure_min_features=0)

        # If warm_start is false or the method has not been fit before, initialize the parameters.
        if (self.warm_start is False) or (hasattr(self, 'weights_') is False):

            self.weights_ = np.ones(self.n_components) / self.n_components

            self.n_real_features = self.X_real.shape[1]
            self.n_categorical_features = self.X_categorical_1hot.shape[1]

            self.mult_weights_ = None
            self.means_ = None
            self.covariances_ = None

            if self.n_real_features > 0:

                n_samples, n_features = self.X_real.shape

                if self.init_params == 'kmeans':
                    kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state).fit(self.X_real)
                    self.means_ = kmeans.cluster_centers_

                elif self.init_params == 'random':
                    self.means_ = self.X_real[np.random.randint(n_samples, size=self.n_components), :]

                if self.covariance_type == "full":
                    self.covariances_ = np.zeros((self.n_components, n_features, n_features))
                    self.covariances_[range(self.n_components), :, :] = np.eye(n_features)
                else:
                    self.covariances_ = np.zeros((self.n_components, n_features))
                    self.covariances_[range(self.n_components), :] = np.ones(n_features)

            if self.n_categorical_features > 0:

                self.n_categories = n_categories

                self.mult_weights_ = []

                for i in range(self.n_components):
                    mult_weights_aux = np.array([])
                    for n in self.n_categories:
                        aux = np.random.random(int(n))
                        mult_weights_aux = np.append(mult_weights_aux, aux / np.sum(aux))

                    self.mult_weights_.append(mult_weights_aux)


        self.prior_weights = prior_weights

        for j in range(self.max_iter):

            if self.n_components == 0:
                self.old_weights = self.weights_.copy()
                break


            self.resp_ = self._e_step(self.X_real, self.X_categorical_1hot)

            self._m_step()

            # Works only for when self.max_iter = 1
            self.old_weights = self.weights_.copy()
            if 0 in self.weights_:
                self.del_0comp()

        return self

    def predict_proba(self, X_real, X_categorical_1hot):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X_real : array-like of shape = [n_samples, n_real_features]
            The input real samples.
        X_categorical_1hot : array-like of shape = [n_samples, n_categorical_features_1hoted]
            The input categorical samples one-hot encoded.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """

        # Check is fit had been called
        check_is_fitted(self, 'resp_')

        log_resp = self._estimate_log_resp(X_real, X_categorical_1hot)
        return np.exp(log_resp)

    def _e_step(self, X_real, X_categorical_1hot):
        """E step.

        Parameters
        ----------
        X_real : array-like, shape (n_samples, n_realfeatures)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        X_categorical_1hot : array-like, shape (n_samples, n_1hotencoded_categorical_features)
            List one-hot encoded categorical features. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibility : array, shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """

        responsibility = np.exp(self._estimate_log_resp(X_real, X_categorical_1hot))

        return responsibility

    def _estimate_log_resp(self, X_real, X_categorical_1hot):
        """Estimate log responsibilities for each sample.
        Compute the responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X_real : array-like, shape (n_samples, n_realfeatures)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        X_categorical_1hot : array-like, shape (n_samples, n_1hotencoded_categorical_features)
            List one-hot encoded categorical features. Each row
            corresponds to a single data point.

        Returns
        -------
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """

        weighted_log_prob = self._estimate_weighted_log_prob(X_real, X_categorical_1hot)
        log_responsibilities = weighted_log_prob - logsumexp(weighted_log_prob, axis=1).reshape(-1, 1)

        return log_responsibilities


    def _estimate_weighted_log_prob(self, X_real, X_categorical_1hot):
        """Estimate the weighted log-probabilities, log P(X | theta) + log weights.

        Parameters
        ----------
        X_real : array-like, shape (n_samples, n_realfeatures)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        X_categorical_1hot : array-like, shape (n_samples, n_1hotencoded_categorical_features)
            List one-hot encoded categorical features. Each row
            corresponds to a single data point.

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """

        weighted_log_prob = self._estimate_log_prob(X_real, X_categorical_1hot) + np.log(self.weights_ +
                                                                                         np.finfo(np.float64).eps)

        return weighted_log_prob

    def _estimate_log_prob(self, X_real, X_categorical_1hot):
        """Estimate the log-probabilities log P(X | theta).
        Compute the log-probabilities per each mixture for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X_real : array-like, shape (n_samples, n_realfeatures)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        X_categorical_1hot : array-like, shape (n_samples, n_1hotencoded_categorical_features)
            List one-hot encoded categorical features. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """

        log_prob = np.zeros((X_real.shape[0], self.n_components))

        if self.n_real_features > 0:
            if self.covariance_type == "full":
                log_prob += log_multivariate_normal_density_full(X_real, means=self.means_, covars=self.covariances_, reg=self.reg_covar)
            else:
                log_prob += log_multivariate_normal_density_diag(X_real, means=self.means_, covars=self.covariances_, reg=self.reg_covar)

        if self.n_categorical_features > 0:
            for i in range(self.n_components):
                log_prob[:, i] += np.sum(X_categorical_1hot * np.log(self.mult_weights_[i] + np.finfo(np.float64).eps), axis=1)

        return log_prob


    def _m_step(self):
        """M step.

        Returns
        -------
        weights : array, shape (n_components,)
        means : array, shape (n_components, n_features)
        covariances : array-like, shape (n_components, n_features, n_features)
        """

        self.weights_ = self._calculate_new_weights()

        if self.n_real_features > 0:
            self.means_ = self._calculate_new_means()
            self.covariances_ = self._calculate_new_covariances()

        if self.n_categorical_features > 0:
            self.mult_weights_ = self._calculate_new_mult_weights()

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
            if self.component_kill is True:
                weights[self.prior_weights == 1] = 0

        weights = weights / (np.sum(weights) + np.finfo(np.float64).eps)

        return weights

    def _calculate_new_means(self):
        """Updates the means based on the new responsibilities.

        Returns
        -------
        means : array, shape (n_components, n_features)
        """

        X = self.X_real
        X_dim = X.shape[1]
        means = np.empty((self.n_components, X_dim))

        for i in range(self.n_components):
            if self.weights_[i] == 0:
                continue

            numerator = np.sum(X * self.resp_[:, i].reshape(-1, 1), axis=0)
            denominator = np.sum(self.resp_[:, i])
            means[i, :] = numerator / denominator

        return means

    def _calculate_new_covariances(self):
        """Updates the covariances based on the new responsibilities.

        Returns
        -------
        covariances : array-like, shape (n_components, n_features, n_features)
        """

        # Investigar se é possivel optimizar isto para fazer sem ciclos

        X = self.X_real
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

    def _calculate_new_mult_weights(self):
        """Updates the multinoulli weights based on the new responsibilities.

         Returns
         -------
         weights : list of n_component arrays each of shape (X_categorical_1hot.shape[1], )
         """

        weights = []
        for i in range(self.n_components):
            aux = np.sum(self.X_categorical_1hot * self.resp_[:, i][:, None], axis=0)
            old_ind = 0
            weights_aux = np.array([])
            for n in self.n_categories:
                new_ind = int(old_ind + n)
                temp = aux[old_ind:new_ind]
                weights_aux = np.append(weights_aux, temp / np.sum(temp))
                old_ind = new_ind
                
            weights.append(weights_aux)

        return weights

    def del_0comp(self):

        ind = np.argwhere(self.weights_ == 0)
        self.n_components -= ind.size
        self.weights_ = np.delete(self.weights_, ind)

        if self.n_real_features > 0:
            self.means_ = np.delete(self.means_, ind, axis=0)
            self.covariances_ = np.delete(self.covariances_, ind, axis=0)

        if self.n_categorical_features > 0:
            for i in ind:
                del self.mult_weights_[i]

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

    import pandas as pd
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.model_selection import train_test_split

    def test_mm():
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/balance_scale'
        df = pd.read_csv(filename, sep=",", header=None)
        y = df[0].values
        X = df[[1, 2, 3, 4]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mm = GaussMultMixture(n_components=3, covariance_type='full', equal_covariances=False, component_kill=False,
                 categorical_features=[0,1,2,3], laplace_smoothing=0.01, reg_covar=1e-6, max_iter=100, init_params='kmeans',
                 random_state=None, warm_start=False)

        mm.fit(X_train)
        mm.predict_proba(X_test)


    test_mm()
