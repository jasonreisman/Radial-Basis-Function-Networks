import numbers
import time

import numpy as np
import pandas as pd
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
        Must be one of:
            'full' (each component has its own general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix)
    equal_covariances : {True, False},
            defaults to False.
            If True in each mixture every component has the same covariance.
    gauss_kill : {True, False},
        defaults to False.
        If True when a prior of the gmm weights is 0 the corresponding
        component weight will be set entirely to 0.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive definite.
    max_iter_gmm : int, defaults to 1.
        The maximum number of EM iterations to perform.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
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
    max_iter_logreg : int, default: 1
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
                 equal_covariances=False, gauss_kill=False, reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans',
                 random_state=None, l1=0.01, l2=0.01, max_iter_logreg=1):
        self.link = link
        self.n_iter = n_iter
        self.n_components = n_components
        self.feature_type = feature_type
        self.covariance_type = covariance_type
        self.equal_covariances = equal_covariances
        self.gauss_kill=gauss_kill
        self.reg_covar = reg_covar
        self.max_iter_gmm = max_iter_gmm
        self.init_params = init_params
        self.random_state = random_state
        self.l1 = l1
        self.l2 = l2
        self.max_iter_logreg = max_iter_logreg

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

        n_samples, n_features = self.X_.shape

        self.gmm_ = []
        for i in range(self.n_classes_):
            self.gmm_.append(GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                             equal_covariances=self.equal_covariances, gauss_kill=self.gauss_kill, reg_covar=self.reg_covar,
                                             max_iter=self.max_iter_gmm, init_params=self.init_params,
                                             random_state=self.random_state, warm_start=True))

        self.logReg_ = LogisticRegressionperClass(l1=self.l1, l2=self.l2, max_iter=self.max_iter_logreg, warm_start=True)

        priors = [None] * self.n_classes_
        for j in range(self.n_iter):

            design_matrix = np.array([]).reshape(n_samples, 0)
            n_comp_mixt = np.array([])
            weights2kill = np.array([])

            if self.feature_type == 'post_prob':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i])

                    design_matrix = np.append(design_matrix, self.gmm_[i].resp_, axis=1)
                    n_comp_mixt = np.append(n_comp_mixt, self.gmm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.gmm_[i].old_weights)

            elif self.feature_type == 'likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i])
                    if self.gmm_[i].n_components != 0:
                        design_matrix = np.append(design_matrix, self.predict_likelihoods(X, self.gmm_[i].means_,
                                                                        self.gmm_[i].covariances_, type='norm'), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.gmm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.gmm_[i].old_weights)

            elif self.feature_type == 'log_likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i])
                    if self.gmm_[i].n_components != 0:
                        design_matrix = np.append(design_matrix, self.predict_likelihoods(X, self.gmm_[i].means_,
                                                                        self.gmm_[i].covariances_, type='log'), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.gmm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.gmm_[i].old_weights)

            elif self.feature_type == 'proj_log_likelihood':
                for i in range(self.n_classes_):
                    self.gmm_[i] = self.gmm_[i].fit(X=self.X_, prior_weights=priors[i])
                    if self.gmm_[i].n_components != 0:
                        likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_,
                                                               type='log')
                        design_matrix = np.append(design_matrix, simplex_proj(likelihoods), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.gmm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.gmm_[i].old_weights)

            self.logReg_ = self.logReg_.fit(design_matrix, self.y_, n_comp_mixt, weights2kill)
            pass

            if self.link != 0:
                priors = self.calculate_priors(self.logReg_.weights_, n_comp_mixt)

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

        n_samples, n_features = X.shape

        design_matrix = np.array([]).reshape(n_samples, 0)
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix, self.gmm_[i].predict_proba(X), axis=1)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_,
                                                                   type='norm'), axis=1)

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_,
                                                                   type='log'), axis=1)

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')
                design_matrix = np.append(design_matrix, simplex_proj(likelihoods), axis=1)

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

        n_samples, n_features = X.shape

        design_matrix = np.array([]).reshape(n_samples, 0)
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix, self.gmm_[i].predict_proba(X), axis=1)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_,
                                                                   type='norm'), axis=1)

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_,
                                                                   type='log'), axis=1)

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                if self.gmm_[i].n_components == 0:
                    continue

                likelihoods = self.predict_likelihoods(X, self.gmm_[i].means_, self.gmm_[i].covariances_, type='log')
                design_matrix = np.append(design_matrix, simplex_proj(likelihoods), axis=1)

        y = self.logReg_.predict(design_matrix)
        return y



    def calculate_priors(self, logReg_weights, n_weights_per_class):
        """ Calculates the priors to be passed to the GMM.
        The log reg weights are normalized and scales by the link parameter
            Parameters
            ----------
            logReg_weights :  array, shape (n_features, )
                Coefficient of the features in the decision function.

            n_weights_per_class : array-like, shape = (n_classes,)
                Number of weights for each class.
                The sum of ths vector is the total number of variables.

            Returns
            -------
            priors : array, shape (n_classes, n_components)
                Dirichlet priors for the GMM.
            """

        priors = []
        logReg_weights_abs = np.absolute(logReg_weights)

        old_ind = 1
        for n in n_weights_per_class:
            new_ind = int(old_ind + n)
            prior_temp = logReg_weights_abs[old_ind: new_ind]
            priors = priors + [1.0 + self.link * (prior_temp / (np.sum(prior_temp) + np.finfo(np.float64).eps))]
            old_ind = new_ind + 1

        return priors

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import scipy.io as io

    def evaluate(X_train, X_test, y_train, y_test):
        # Create and fit the Logistic Regression
        rbfn = RadialBasisFunctionNetwork(link=100, n_iter=10, n_components=2, covariance_type='diag',
                                          equal_covariances=False, feature_type='proj_log_likelihood', gauss_kill=True,
                                          reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans', random_state=None,
                                          l1=0.01, l2=0.01, max_iter_logreg=15000)
        bef = time.time()
        rbfn.fit(X_train, y_train)
        now = time.time()
        print(now - bef)

        # Make predictions
        #        y_pred_prob = rbfn.predict_proba(X_test)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        pass

    def iris():
        np.random.seed(1)
        digits = datasets.load_iris()
        X = digits.data
        y = digits.target
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        evaluate(X_train, X_test, y_train, y_test)

    def wbdc():
        np.random.seed(1)
        df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data', header=None)
        y = df[[1]].values.ravel()
        X = df.drop([0,1], axis=1).values
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        evaluate(X_train, X_test, y_train, y_test)

    def glass():
        np.random.seed(1)
        df = pd.read_csv('../datasets/glass.csv')
        y = df[['Type']].values.ravel()
        X = df.drop(['Type'], axis=1).values
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.49)

        evaluate(X_train, X_test, y_train, y_test)

    def sonar():
        np.random.seed(1)
        df = pd.read_csv('../datasets/sonar.csv')
        X = np.vstack((df.columns.values, df.values))
        y = X[:, -1]
        X = X[:, :-1]
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.495)

        evaluate(X_train, X_test, y_train, y_test)

    def wine():
        np.random.seed(1)
        digits = datasets.load_wine()
        X = digits.data
        y = digits.target
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.49)

        evaluate(X_train, X_test, y_train, y_test)

    def colon():
        np.random.seed(1)
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon.csv'
        df = pd.read_csv(filename, sep=' ', header=None)
        X = df.values.T
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon_target.csv'
        df = pd.read_csv(filename, sep=' ', header=None)
        df.values[df.values < 0] = 0
        df.values[df.values > 0] = 1
        y = df.values.ravel()
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        evaluate(X_train, X_test, y_train, y_test)

    def leukemia():
        np.random.seed(1)
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/leukemia.csv'
        df = pd.read_csv(filename)
        X = df.values.T
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon_target.csv'
        y = np.zeros(df.columns.size)
        for ind, i in enumerate(df.columns):
            if 'AML' in i:
                y[ind] = 1

        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.46)

        evaluate(X_train, X_test, y_train, y_test)

    def orl():
        file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/ORL"
        data = io.loadmat(file)
        X_ = np.array(data['fea'], dtype=np.float64)
        X = np.array([]).reshape(0, int(X_.size / 400))
        for x in np.split(X_, 400):
            X = np.append(X, x, axis=0)
        y = np.asarray(data['gnd'], dtype=np.float64).ravel()

        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        evaluate(X_train, X_test, y_train, y_test)

    #iris()
    #wbdc()
    #glass()
    #sonar()
    #wine()
    #colon()
    #leukemia()
    orl()

