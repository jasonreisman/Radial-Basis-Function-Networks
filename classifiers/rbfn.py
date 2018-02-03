import numbers
import time

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal

from classifiers.mult_log_reg_per_class import LogisticRegressionperClass
#from classifiers.mult_log_reg_per_class_BFGS import LogisticRegressionperClass
#from classifiers.mult_log_reg_per_class_OWL_QN import LogisticRegressionperClass
from sklearn.linear_model import LogisticRegression
from mixtures.gmm import GaussianMixture
#from sklearn.mixture import GaussianMixture

from utils.stats import mult_gauss_pdf, log_multivariate_normal_density_diag, log_multivariate_normal_density_full


def simplex_proj(z):
    """Projectets rows of z in the simplex.

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
    link : integer, default 0
        Scales the Dirichlet priors created from the log reg weights.
        0 : no prior.
        (>>1) : basically only prior used.
    n_iter : int, defaults to 100
        Number of iterations performed in method fit.

    (for the Gaussian Mixture Models)
    n_components : int, defaults to 2.
        The number of mixture components.
    feature_type : string, default to 'likelihoods'
        Must be one of:
            'likelihood' (design matrix passed to the Logistic Regression
                are the likelihoods of the samples of each mixture),
            'log_likelihood' (design matrix passed to the Logistic Regression
                are the log likelihoods of the samples of each mixture),
            'proj_log_likelihood' (design matrix passed to the Logistic Regression
                are the log likelihoods of the samples of each mixture projected on the simplex),
            'post_prob' (design matrix passed to the Logistic Regression
                are the component probabilities of the samples of each mixture),
    covariance_type : {'full', 'diag'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix)
    equal_covariances : {True, False},
            defaults to False.
            If True in each mixture every component has the same covariance.
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

    (For the Logistic Regression)
    l1 : float, default: 0.01
        l1 regularization strength; must be a positive float.
        Bigger values specify stronger regularization.
    l2 : float, default: 0.01
        l2 regularization strength; must be a positive float.
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


    def __init__(self, link=0, n_iter=100, n_components=5, feature_type='likelihood', covariance_type='full',
                 equal_covariances=False, reg_covar=1e-6, n_iter_gmm=1, init_params='kmeans', weights_init=None,
                 means_init=None, random_state=None, l1=0.01, l2=0.01, n_iter_logreg=1):
        self.link = link
        self.n_iter = n_iter
        self.n_components = n_components
        self.feature_type = feature_type
        self.covariance_type = covariance_type
        self.equal_covariances = equal_covariances
        self.reg_covar = reg_covar
        self.n_iter_gmm = n_iter_gmm
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.random_state = random_state
        self.l1 = l1
        self.l2 = l2
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
        if not isinstance(self.link, numbers.Number) or self.link < 0:
            raise ValueError("link must be a number in the interval [0, inf]. Valor passed: %r" % self.link)

        # Check if there are at least 2 classes
        if self.n_classes_ < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        # Check type parameter
        if self.feature_type not in ['likelihood', 'log_likelihood', 'proj_log_likelihood', 'post_prob']:
            raise ValueError("feature_type must be a string contained in ['likelihood', 'log_likelihood', "
                             "'proj_log_likelihood', 'post_prob']. Valor passed: %s" % self.feature_type)

        self.X_ = X
        self.y_ = y

        self.gmm_ = []
        for i in range(self.n_classes_):
            self.gmm_.append(GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                             equal_covariances=self.equal_covariances, reg_covar=self.reg_covar,
                                             n_iter=self.n_iter_gmm, init_params=self.init_params,
                                             weights_init=self.weights_init, means_init=self.means_init,
                                             random_state=self.random_state, warm_start=True))


#        self.logReg_ = LogisticRegression(penalty='l2',  tol=1e-10, C=1.0/self.l, solver='sag',
#                                          max_iter=self.n_iter_logreg, multi_class='multinomial', warm_start=True)
        self.logReg_ = LogisticRegressionperClass(l1=self.l1, l2=self.l2, n_iter=self.n_iter_logreg, warm_start=True)


        # design_matrix with shape (n_samples, number of total components in all the gmms), there are n_classes mixtures
        design_matrix = np.empty((self.X_.shape[0], self.n_classes_ * self.n_components))
        priors = np.ones((self.n_classes_, self.n_components))
        for j in range(self.n_iter):

            if self.feature_type == 'post_prob':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i, :])
                    design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = self.gmm_[i].resp_

            elif self.feature_type == 'likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i, :])
                    design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                        self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='norm')

            elif self.feature_type == 'log_likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i, :])
                    design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                        self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')

            elif self.feature_type == 'proj_log_likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i, :])
                    likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')
                    design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = simplex_proj(likelihoods)

            self.logReg_ = self.logReg_.fit(design_matrix, self.y_)

            if self.link != 0:
                priors = self.calculate_priors(self.logReg_.weights_)

        # Return the classifier
        return self

    def predict_likelihoods(self, X, means, covariances, type='norm'):
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
        type : string, defaults to 'norm'
            'norm' (normal likelihoods)
            'log' (log likelihoods)

        Returns
        -------
        post_probs : array-like of shape = [n_samples, n_components]
            Matrix containing the likelihoods for each gaussian in the mixture.
        """

        # Check type parameter
        if type not in ['norm', 'log']:
            raise ValueError("link must be a string contained in ['norm', 'log']. Valor passed: %s" % type)

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        # Input validation
        X = check_array(X)

        post_probs = np.empty((X.shape[0], means.shape[0]))

        if self.covariance_type == 'full':
            post_probs = log_multivariate_normal_density_full(X, means=means, covars=covariances, reg=self.reg_covar)
            if type == 'norm':
                post_probs = np.exp(post_probs)

        else:
            post_probs = log_multivariate_normal_density_diag(X, means=means, covars=covariances, reg=self.reg_covar)
            if type == 'norm':
                post_probs = np.exp(post_probs)

        return post_probs

    def get_mixtures(self):
        """Returns the centers and co-variance matrices of the GMM.

        Returns
        -------
        centers : array, shape (n_components,)
            Centers of the GMM.
        cov_matrices : array-type, shape (.n_components, n_features, n_features)
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

        design_matrix = np.zeros((X.shape[0], self.n_classes_ * self.n_components))
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = self.gmm_[i].predict_proba(X)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                    self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='norm')

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                    self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = simplex_proj(likelihoods)

        probs = self.logReg_.predict_proba(design_matrix)
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

        design_matrix = np.zeros((X.shape[0], self.n_classes_ * self.n_components))
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = self.gmm_[i].predict_proba(X)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                    self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='norm')

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = \
                    self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')
                design_matrix[:, i * self.n_components: (i + 1) * self.n_components] = simplex_proj(likelihoods)

        y = self.logReg_.predict(design_matrix)
        return y



    def calculate_priors(self, logReg_weights):
        """ Calculates the priors to be passed to the GMM.

        The log reg weights are normalized and scales by the link parameter

            Parameters
            ----------
            logReg_weights :  array, shape (n_features, )
                Coefficient of the features in the decision function.

            Returns
            -------
            priors : array, shape (n_classes, n_components)
                Dirichlet priors for the GMM.
            """

        priors = np.ones((self.n_classes_, self.n_components))
        logReg_weights_abs = np.absolute(logReg_weights)

        for i in range(self.n_classes_):
            prior_temp = logReg_weights_abs[i * int(logReg_weights.size/self.n_classes_): (i+1) * int(logReg_weights.size/self.n_classes_)][1:]
            priors[i, :] = priors[i, :] + self.link * (prior_temp / np.sum(prior_temp))

        return priors

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    def mnist():
        np.random.seed(1)
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create and fit the Logistic Regression
        rbfn = RadialBasisFunctionNetwork(link=1000, n_iter=100, n_components=10, covariance_type='diag', equal_covariances=False, feature_type='post_prob', reg_covar=1e-6,
                                          n_iter_gmm=1, init_params='kmeans', weights_init=None, means_init=None,
                                          random_state=None, l1=0.01, l2=0.01, n_iter_logreg=100)
        bef = time.time()
        rbfn.fit(X_train, y_train)
        now = time.time()
        print(now-bef)

        # Make predictions
        y_pred_prob = rbfn.predict_proba(X_test)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        pass

    mnist()