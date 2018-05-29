import numbers
import time
import multiprocessing

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import compute_class_weight
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from classifiers.mult_log_reg_per_class import LogisticRegressionperClass
#from classifiers.mult_log_reg_per_class_BFGS import LogisticRegressionperClass
#from classifiers.mult_log_reg_per_class_OWL_QN import LogisticRegressionperClass
from sklearn.linear_model import LogisticRegression
from mixtures.mm import GaussMultMixture
#from sklearn.mixture import GaussianMixture
from utils.one_hot_encoder import OneHotEncoder

from utils.stats import mult_gauss_pdf, log_multivariate_normal_density_diag, log_multivariate_normal_density_full


def _job(p):
    mm = p[0]
    X_real = p[1]
    X_categorical_1hot = p[2]
    n_categories = p[3]
    prior_weights = p[4]

    mm = mm.fit(X_real=X_real, X_categorical_1hot=X_categorical_1hot, n_categories=n_categories,
                prior_weights=prior_weights)

    return mm

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

def unique_classes(y):

    unique = []
    for i in y:
        if i not in unique and i != -1:
            unique.append(i)

    return np.array(unique)


class RadialBasisFunctionNetwork(BaseEstimator, ClassifierMixin):
    """ Implements a Radial Basis Function Network classifier.
    Uses Gaussian Mixture Models for the Radial Functions. The way of training allows
    a link between the EM step of the GMM and the Logistic Regression, where the weights
    of the Logistic Regression are to be used for the training of the Gaussian Mixture Models.

    This algorithm support semi-suppervised learning, meaning that certain samples can be used
    in the fitting of the densities but not in the fitting of the log reg.
    For that every unlabeled sample should be identified with a -1 in the y vector.
    Parameters
    ----------
    link : integer, default 0
        Scales the Dirichlet priors created from the log reg weights.
        0 : no prior.
        (>>1) : basically only prior used.
    max_iter : int, defaults to 1000
        Maximum number of iterations performed in method fit.
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
        Calculate the differences between two consecutive design_matrices,
        if the biggest feature norm of the difference is bellow this value stop.
    ind_cat_features : tuple type, shape (n_cat_features,), default = None
        array with the indexes of the categorical features.
    cat_features : list of lists, default = None
        The inner lists contain the categories for each categorical variable.
        Must be in the same data type as passed in X.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    (for the Gaussian Mixture Models)
    n_components : int, defaults to 2.
        The number of mixture components.
    feature_type : string, default to 'post_prob'
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
    component_kill : {True, False},
        defaults to True.
        If True when a prior of the gmm weights is 0 the corresponding
        component weight will be set entirely to 0.
    laplace_smoothing : float, defalt = 0.01
        Constant responsible for the laplace smoothing in the MAP estimation of the categorical weights.
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

    (For the Logistic Regression)
    l1 : float, default: 0.01
        l1 regularization strength; must be a positive float.
        Bigger values specify stronger regularization.
    l2 : float, default: 0.01
        l2 regularization strength; must be a positive float.
        Bigger values specify stronger regularization.
    max_iter_logreg : int, default: 5
        Number of iterations performed by the optimization algorithm
        in search for the optimal weights.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :method:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :method:`fit`
    """


    def __init__(self, link=0, max_iter=1000, tol=1e-3, n_components=5, feature_type='post_prob', covariance_type='full',
                 equal_covariances=False, component_kill=True, ind_cat_features=(), cat_features=None, laplace_smoothing=0.001,
                 reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans', random_state=None, l1=0.01, l2=0.01,
                 max_iter_logreg=5):
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.feature_type = feature_type
        self.covariance_type = covariance_type
        self.equal_covariances = equal_covariances
        self.component_kill = component_kill
        self.ind_cat_features = ind_cat_features
        self.cat_features = cat_features
        self.laplace_smoothing = laplace_smoothing
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
        X, y = check_X_y(X, y, dtype=None)#, dtype=float64)
        y = column_or_1d(y, warn=True)
        #check_classification_targets(y)

        #self.classes_, _ = np.unique(y, return_inverse=False)

        #i, = np.where(self.classes_ == -1)
        #self.classes_ = np.delete(self.classes_, i)
        self.classes_ = unique_classes(y)

        supervised_ind = []  # boolean vector that indicates if a sample has label
        for i in range(y.size):
            if y[i] != -1:
                supervised_ind.append(True)
            else:
                supervised_ind.append(False)
        self.y_ = y[supervised_ind]

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

        n_samples, self.n_features_ = X.shape

        #self.ind_cat_features = np.array(self.ind_cat_features)
        self.real_features_ = np.delete(np.arange(self.n_features_), np.array(self.ind_cat_features))

        self.X_real_ = X[:, self.real_features_]
        self.X_categorical_1hot_ = np.array([]).reshape((n_samples, 0))
        self.onehot_encoders_ = []
        self.n_categories_ = np.array([])
        for ind, i in enumerate(self.ind_cat_features):
            if self.cat_features is not None:
                categories = self.cat_features[ind]
            else:
                categories = None
            oneHot = OneHotEncoder().fit(X[:, i], categories)
            self.onehot_encoders_.append(oneHot)
            oneHotTrans = oneHot.transform(X[:, i])
            self.n_categories_ = np.append(self.n_categories_, oneHotTrans.shape[1])
            self.X_categorical_1hot_ = np.append(self.X_categorical_1hot_, oneHotTrans, axis=1)

        self.n_real_features_ = self.real_features_.size
        self.n_cat_features_ = len(self.ind_cat_features)

        self.X_real_ = check_array(self.X_real_, ensure_min_features=0)
        self.X_categorical_1hot_ = check_array(self.X_categorical_1hot_, ensure_min_features=0)

        old_design_matrix = np.ones((sum(supervised_ind), self.n_components * self.n_classes_)) * np.inf

        self.mm_ = []
        for i in range(self.n_classes_):
            self.mm_.append(GaussMultMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                             equal_covariances=self.equal_covariances,
                                             component_kill=self.component_kill,
                                             laplace_smoothing=self.laplace_smoothing, reg_covar=self.reg_covar,
                                             max_iter=self.max_iter_gmm, init_params=self.init_params,
                                             random_state=self.random_state, warm_start=True))

        self.logReg_ = LogisticRegressionperClass(l1=self.l1, l2=self.l2, max_iter=self.max_iter_logreg, warm_start=True)

        self.n_iter_ = 0
        priors = [None] * self.n_classes_
        for j in range(self.max_iter):
            #print("Iteration", j)

            design_matrix = np.array([]).reshape(sum(supervised_ind), 0)
            n_comp_mixt = np.array([])
            weights2kill = np.array([])

            # For parelelization
            # params = []
            # for i in range(self.n_classes_):
            #
            #     params.append([self.mm_[i], self.X_real_, self.X_categorical_1hot_, self.n_categories_, priors[i]])
            #
            # pool = multiprocessing.Pool(2)
            # self.mm_ = pool.map(_job, params)
            # pool.close()

            for i in range(self.n_classes_):
                 self.mm_[i] = self.mm_[i].fit(X_real=self.X_real_, X_categorical_1hot=self.X_categorical_1hot_,
                                               n_categories=self.n_categories_, prior_weights=priors[i])



            if self.feature_type == 'post_prob':
                for i in range(self.n_classes_):

                    design_matrix = np.append(design_matrix, self.mm_[i].resp_[supervised_ind, :], axis=1)
                    n_comp_mixt = np.append(n_comp_mixt, self.mm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.mm_[i].old_weights)

            elif self.feature_type == 'likelihood':
                for i in range(self.n_classes_):

                    if self.mm_[i].n_components != 0:
                        design_matrix = np.append(design_matrix, self.predict_likelihoods(self.X_real_[supervised_ind, :],
                                                            self.X_categorical_1hot_[supervised_ind, :], self.mm_[i], type='norm'), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.mm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.mm_[i].old_weights)

            elif self.feature_type == 'log_likelihood':
                for i in range(self.n_classes_):

                    if self.mm_[i].n_components != 0:
                        design_matrix = np.append(design_matrix, self.predict_likelihoods(self.X_real_[supervised_ind, :], self.X_categorical_1hot_[supervised_ind, :], self.mm_[i], type='log'), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.mm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.mm_[i].old_weights)

            elif self.feature_type == 'proj_log_likelihood':
                for i in range(self.n_classes_):

                    if self.mm_[i].n_components != 0:
                        likelihoods = self.predict_likelihoods(self.X_real_[supervised_ind, :], self.X_categorical_1hot_[supervised_ind, :], self.mm_[i], type='log')
                        design_matrix = np.append(design_matrix, simplex_proj(likelihoods), axis=1)

                    n_comp_mixt = np.append(n_comp_mixt, self.mm_[i].n_components)
                    weights2kill = np.append(weights2kill, self.mm_[i].old_weights)

            self.logReg_ = self.logReg_.fit(design_matrix, self.y_, n_comp_mixt, weights2kill)

            # Check stop criterion
            ind = np.argwhere(weights2kill == 0)
            aux = np.hstack((old_design_matrix[:, ind].reshape(old_design_matrix.shape[0], -1), np.delete(old_design_matrix, ind, axis=1) - design_matrix))
            norms = np.linalg.norm(aux, axis=0)
            max_tol = np.max(norms)
            #print("Max diff between datasets:", max_tol)

            if max_tol < self.tol:
                break

            old_design_matrix = design_matrix.copy()

            # Check if all elements of the design matrix (mixture components) were discarded
            if design_matrix.size == 0:
                break

            if self.link != 0:
                priors = self.calculate_priors(self.logReg_.weights_, n_comp_mixt)

            self.n_iter_ += 1

        # Return the classifier
        return self



    def predict_likelihoods(self, X_real, X_categorical_1hot, mm, type='norm'):
        """ Predict likelihood of each sample given the mixtures.
        Parameters
        ----------
        X_real : array-like of shape = [n_samples, n_real_features]
            The input real samples.
        X_categorical_1hot : array-like of shape = [n_samples, n_cat_features]
            The input categorical samples one-hot encoded.
        mm : type object
            mixture model with atributes:
                means : array-like, shape (n_components, n_features)
                    The mean of each mixture component.
                covariances : array-like
                    The covariance of each mixture component. The shape depends on covariance_type:
                    (n_components, n_features)             if 'diag',
                    (n_components, n_features, n_features) if 'full'
                mult_weights : list with n_components elements,
                    each with the multinoulli weights of each component.
        type : string, defaults to 'norm'
            'norm' (normal likelihoods)
            'log' (log likelihoods)
        mult_weights : list with n_components elements,
            each with the multinoulli weights of each component.

        Returns
        -------
        post_probs : array-like of shape = [n_samples, n_components]
            Matrix containing the likelihoods for each gaussian in the mixture.
        """

        post_probs = np.zeros((X_real.shape[0], mm.n_components))

        if self.n_real_features_ > 0:
            if self.covariance_type == 'full':
                post_probs += log_multivariate_normal_density_full(X_real, means=mm.means_, covars=mm.covariances_, reg=self.reg_covar)

            else:
                post_probs += log_multivariate_normal_density_diag(X_real, means=mm.means_, covars=mm.covariances_, reg=self.reg_covar)

        if self.n_cat_features_ > 0:
            for i, weights in enumerate(mm.mult_weights_):
                post_probs[:, i] += np.sum(X_categorical_1hot * np.log(weights + np.finfo(np.float64).eps), axis=1)

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
        check_is_fitted(self, 'mm_')

        centers = []
        cov_matrices = []
        for i in range(len(self.mm_)):
            centers.append(self.mm_[i].means_)
            cov_matrices.append(self.mm_[i].covariances_)

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
        X = check_array(X, dtype=None)

        # Check is fit had been called
        check_is_fitted(self, 'mm_')

        n_samples, n_features = X.shape

        if n_features != self.n_features_:
            raise ValueError()

        X_real = X[:, self.real_features_]
        X_categorical_1hot = np.array([]).reshape((X.shape[0], 0))
        for ind, n in enumerate(self.ind_cat_features):
            oneHotTrans = self.onehot_encoders_[ind].transform(X[:, n])
            X_categorical_1hot = np.append(X_categorical_1hot, oneHotTrans, axis=1)

        X_real = check_array(X_real, ensure_min_features=0)
        X_categorical_1hot = check_array(X_categorical_1hot, ensure_min_features=0)

        design_matrix = np.array([]).reshape(n_samples, 0)
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix, self.mm_[i].predict_proba(X_real, X_categorical_1hot), axis=1)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i],
                                                                   type='norm'), axis=1)

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i],
                                                                   type='log'), axis=1)

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                likelihoods = self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i], type='log')
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
        X = check_array(X, dtype=None)

        # Check is fit had been called
        check_is_fitted(self, 'mm_')

        n_samples, n_features = X.shape

        if n_features != self.n_features_:
            raise ValueError()

        X_real = X[:, self.real_features_]
        X_categorical_1hot = np.array([]).reshape((X.shape[0], 0))
        for ind, n in enumerate(self.ind_cat_features):
            oneHotTrans = self.onehot_encoders_[ind].transform(X[:, n])
            X_categorical_1hot = np.append(X_categorical_1hot, oneHotTrans, axis=1)

        X_real = check_array(X_real, ensure_min_features=0)
        X_categorical_1hot = check_array(X_categorical_1hot, ensure_min_features=0)

        design_matrix = np.array([]).reshape(n_samples, 0)
        if self.feature_type == 'post_prob':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix, self.mm_[i].predict_proba(X_real, X_categorical_1hot), axis=1)

        if self.feature_type == 'likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i],
                                                                   type='norm'), axis=1)

        if self.feature_type == 'log_likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                design_matrix = np.append(design_matrix,
                                          self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i],
                                                                   type='log'), axis=1)

        if self.feature_type == 'proj_log_likelihood':
            for i in range(self.n_classes_):
                if self.mm_[i].n_components == 0:
                    continue

                likelihoods = self.predict_likelihoods(X_real, X_categorical_1hot, self.mm_[i], type='log')
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

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import scipy.io as io
    from sklearn import preprocessing
    from mnist import MNIST
    import random

    def evaluate(X_train, X_test, y_train, y_test):
        # Create and fit the Logistic Regression

        np.random.seed(1)
        Parameters = {'l1': 0.01, 'l2': 0.01, 'link': 1000, 'n_components': 10, 'component_kill': True, 'covariance_type': 'full'}
        rbfn = RadialBasisFunctionNetwork(**Parameters)

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
        #random.seed(1)
        #ind = random.sample(range(0, y_train.shape[0]), int((1 - 1) * y_train.shape[0]))
        #y_train[ind] = -1

        evaluate(X_train, X_test, y_train, y_test)

    def wbdc():
        np.random.seed(1)
        df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data', header=None)
        y = df[[1]].values.ravel()
        X = df.drop([0,1], axis=1).values
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        random.seed(1)
        ind = random.sample(range(0, y_train.shape[0]), int((1 - 1) * y_train.shape[0]))
        y_train[ind] = -1
        evaluate(X_train, X_test, y_train, y_test)

    def glass():
        np.random.seed(1)
        df = pd.read_csv('../datasets/glass.csv')
        y = df[['Type']].values.ravel()
        X = df.drop(['Type'], axis=1).values
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.49)
        random.seed(1)
        ind = random.sample(range(0, y_train.shape[0]), int((1 - 0.05) * y_train.shape[0]))
        y_train[ind] = -1
        evaluate(X_train, X_test, y_train, y_test)

    def sonar():
        np.random.seed(1)
        df = pd.read_csv('../datasets/sonar.csv')
        X = np.vstack((df.columns.values, df.values))
        y = X[:, -1]
        X = X[:, :-1]
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.495)
        #random.seed(1)
        ind = random.sample(range(0, y_train.shape[0]), int((1 - 1) * y_train.shape[0]))
        y_train[ind] = -1
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

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        evaluate(X_train, X_test, y_train, y_test)

    def balance_scale():
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/balance_scale'
        df = pd.read_csv(filename, sep=",", header=None)
        y = df[0].values
        X = df[[1, 2, 3, 4]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        evaluate(X_train, X_test, y_train, y_test)

    def histpopindex(): # with categories
        filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/HistoricalPopularityIndex.csv'
        df = pd.read_csv(filename)
        df = df.dropna(axis=0)
        y = df['continent'].values
        df = df.drop(df.columns[[0, 1, 4, 5, 7]], axis=1)
        X = df.values
        cat_features = None#[np.unique(X[:, 0]), np.unique(X[:, 2]), np.unique(X[:, 5]), np.unique(X[:, 6]), np.unique(X[:, 7])]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=6)

        rbfn = RadialBasisFunctionNetwork(link=100, max_iter=10, n_components=3, feature_type='post_prob', covariance_type='full',
                 equal_covariances=False, component_kill=False, ind_cat_features=(0,2,5,6,7), cat_features=cat_features, laplace_smoothing=0.001,
                 reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans', random_state=None, l1=0.01, l2=0.01,
                 max_iter_logreg=1)

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

    def yaleB():
        file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/YaleB"
        data = io.loadmat(file)
        X = data['fea']
        y = data['gnd'].ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.499, random_state=1)
        evaluate(X_train, X_test, y_train, y_test)

    def TDT2():
        file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/TDT2"
        data = io.loadmat(file)
        X = data['fea']#.toarray()
        y = data['gnd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        evaluate(X_train, X_test, y_train, y_test)

    def usps():
        import pandas as pd
        file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/usps"
        df = pd.read_csv(file, sep=' ', header=None)  # , index_col=0)
        X_train = df.values[:, 1:-1]
        y_train = df[0]
        file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/usps.t"
        df = pd.read_csv(file, sep=' ', header=None)  # , index_col=0)
        X_test = df.values[:, 1:-1]
        y_test = df[0]

        X_ = np.zeros((X_train.shape[0], X_train.shape[1]))
        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                X_[i, j] = X_train[i, j].split(":")[-1]
        X_train = X_

        X_ = np.zeros((X_test.shape[0], X_test.shape[1]))
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                X_[i, j] = X_test[i, j].split(":")[-1]
        X_test = X_

        evaluate(X_train, X_test, y_train, y_test)

    def mnist():
        mndata = MNIST('/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets')
        images, labels = mndata.load_training()
        X_train = np.zeros((len(images), len(images[0])))
        for i in range(X_train.shape[0]):
            X_train[i, :] = np.array(images[i])
        y_train = np.array(labels)

        images, labels = mndata.load_testing()
        X_test = np.zeros((len(images), len(images[0])))
        for i in range(X_test.shape[0]):
            X_train[i, :] = np.array(images[i])
        y_test = np.array(labels)

        pca = PCA(n_components=260)
        pca = pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        evaluate(X_train, X_test, y_train, y_test)

    def check_estimator():
        from sklearn.utils.estimator_checks import check_estimator
        check_estimator(RadialBasisFunctionNetwork)
        pass


    #iris()
    #wbdc()
    #glass()
    #sonar()
    #wine()
    #colon()
    #leukemia()
    #orl()
    #yaleB()
    #TDT2()
    #usps()
    mnist()
    #check_estimator()


    #balance_scale()
    #histpopindex()

