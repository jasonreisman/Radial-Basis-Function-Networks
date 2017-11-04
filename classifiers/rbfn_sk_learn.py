import numbers

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal

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
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    (for the Gaussian Mixture Models)
    n_components : int, defaults to 1.
        The number of mixture components.
    covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}, defaults to ‘full’.
        String describing the type of covariance parameters to use. Must be one of:
        'full' (each component has its own general covariance matrix),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance).
    tol_gmm : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the lower
        bound average gain is below this threshold.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter_gmm : int, defaults to 1.
        The number of EM iterations to perform.
    init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
        The method used to initialize the weights, the means and the precisions.
        Must be one of:
        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.

    (For the Logistic Regression)
    penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
        Used to specify the norm used in the penalization.
        The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
    tol_logreg : float, default: 1e-4
        Tolerance for stopping criteria.
    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    class_weight : dict or ‘balanced’, default: None
        Weights associated with classes in the form {class_label: weight}.
         If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional
        to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        Note that these weights will be multiplied with sample_weight (passed through the fit method)
        if sample_weight is specified.
    solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘saga’
        Algorithm to use in the optimization problem.
        For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
        ‘saga’ are faster for large ones.
        For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
        handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
        ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
        Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.
        You can preprocess the data with a scaler from sklearn.preprocessing.
    max_iter_logreg : int, default: 1
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.
    multi_class : str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
        Multiclass option can be either ‘ovr’ or ‘multinomial’.
        If the option chosen is ‘ovr’, then a binary problem is fit for each label.
        Else the loss minimised is the multinomial loss fit across the entire probability distribution.
        Does not work for liblinear solver.
    n_jobs : int, default: -1
        Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.
        This parameter is ignored when the ``solver``is set to ‘liblinear’
        regardless of whether ‘multi_class’ is specified or not. If given a value of -1, all cores are used.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :method:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :method:`fit`
    """


    def __init__(self, link=0, random_state=None, n_iter=100, n_components=2, covariance_type="full", tol_gmm=1e-3,
                 reg_covar=1e-06, max_iter_gmm=1, init_params="kmeans", penalty="l2", tol_logreg=1e-4, C=100,
                 class_weight=None, solver="newton-cg", max_iter_logreg=1, multi_class="multinomial", n_jobs=-1):
        self.link = link
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol_gmm = tol_gmm
        self.reg_covar = reg_covar
        self.max_iter_gmm = max_iter_gmm
        self.init_params = init_params
        self.penalty = penalty
        self.tol_logreg = tol_logreg
        self.C = C
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter_logreg = max_iter_logreg
        self.multi_class = multi_class
        self.n_jobs = n_jobs

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

        # Check parameters
        if (not isinstance(self.link, numbers.Number)) or (self.link < 0) or (self.link > 1):
            raise ValueError("link must be a number in the interval [0,1]. Valor passed: %r" % self.link)

        if (not isinstance(self.n_iter, numbers.Number)) or (self.n_iter < 1):
            raise ValueError("The number of iterations performed must be a positive integer. "
                             "The parameter passed was: %r" % self.n_iter)

        self.X_ = X
        self.y_ = y

        self.gmm_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                    tol=self.tol_gmm, reg_covar=self.reg_covar, max_iter=self.max_iter_gmm, n_init=1,
                                    init_params=self.init_params, weights_init=None, means_init=None,
                                    precisions_init=None, random_state=self.random_state, warm_start=False,
                                    verbose=0, verbose_interval=10)
        self.logReg_ = LogisticRegression(penalty=self.penalty, dual=False, tol=self.tol_logreg, C=self.C,
                                          fit_intercept=True, intercept_scaling=1, class_weight=self.class_weight,
                                          random_state=self.random_state, solver=self.solver,
                                          max_iter=self.max_iter_logreg, multi_class=self.multi_class, verbose=0,
                                          warm_start=True, n_jobs=self.n_jobs)

        for iter in range(self.n_iter):

            if iter != 0:
                self.gmm_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                    tol=self.tol_gmm, reg_covar=self.reg_covar, max_iter=self.max_iter_gmm, n_init=1,
                                    init_params=self.init_params, weights_init=gmm_weights, means_init=gmm_means,
                                    precisions_init=gmm_precisions, random_state=self.random_state, warm_start=False,
                                    verbose=0, verbose_interval=10)

            if 1 == 1:
                self.gmm_.fit(X=self.X_)
                design_matrix = self.gmm_.predict_proba(X=self.X_)
                self.logReg_.fit(X=design_matrix, y=self.y_)

            else:
                self.gmm_.fit(X=self.X_)
                design_matrix = self.predict_likelihoods(X=self.X_, means=self.gmm_.means_, covariances=self.gmm_.covariances_)
                self.logReg_.fit(X=design_matrix, y=self.y_)

            logReg_weights = self.logReg_.coef_

            gmm_means = self.gmm_.means_
            gmm_precisions = self.gmm_.precisions_
            gmm_weights = self.gmm_.weights_
            gmm_weights = self.calculate_gmm_weights(logReg_weights, gmm_weights)

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

        return post_probs

    def get_mixtures(self):
        """Returns the centers and co-variance matrices of the GMM.

        Returns
        -------
        means : array, shape (n_mixtures,)
            Centers of the GMM.
        covariances : array-type
            The covariance of each mixture component. The shape depends on covariance_type:
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
        """

        # Check is fit had been called
        check_is_fitted(self, 'gmm_')

        means = self.gmm_.means_
        covariances = self.gmm_.covariances_

        return means, covariances

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

        #X_likelihood = self.predict_likelihoods(X=X, means=self.gmm_.means_, covariances=self.gmm_.covariances_)
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
        gmm_weights_logReg = np.sum(logReg_weights_abs, axis=0) / np.sum(logReg_weights_abs)

        gmm_weights = (1 - self.link) * gmm_weights + self.link * gmm_weights_logReg

        return gmm_weights


if __name__ == '__main__':

    from sklearn.utils.estimator_checks import check_estimator
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from matplotlib.patches import Ellipse
    from sklearn.metrics import accuracy_score

    def test_classifier():
        return check_estimator(RadialBasisFunctionNetwork)


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


    # plots the level curves of the probability estimations in 2D space
    def plot_decision_regions(X_train, X_test, y_train, y_test, classifier, resolution=0.1):

        # setup color map
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y_train))])

        # plot the decision surface
        x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)
        for i in range(Z.shape[1]):
            Zi = Z[:, i].reshape(xx1.shape)
            cp = plt.contour(xx1, xx2, Zi, alpha=0.4, cmap=None)
            plt.clabel(cp, inline=True, fontsize=10)
            plt.xlim(xx1.min(), xx1.max())
            plt.ylim(xx2.min(), xx2.max())

            # plot train samples
            for idx, cl in enumerate(np.sort(np.unique(y_train))):
                plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
            # plot test samples
            for idx, cl in enumerate(np.sort(np.unique(y_test))):
                plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')

            plt.tight_layout()
            plt.show()

    #test_classifier()
    def test_rbfn():
        # Import first two features from iris data set
        X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0),
                                   shuffle=True, random_state=2)

        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # Create and fit the Logistic Regression to
        rbfn = RadialBasisFunctionNetwork(link=0, random_state=None, n_iter=300, n_components=2, covariance_type="full", tol_gmm=1e-3,
                     reg_covar=1e-06, max_iter_gmm=1, init_params="kmeans", penalty="l2", tol_logreg=1e-4, C=100,
                     class_weight=None, solver="saga", max_iter_logreg=1, multi_class="multinomial", n_jobs=-1)
        rbfn.fit(X_train, y_train)

        # Plot train and test data
        plt.figure()
        ax = plt.gca()
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y_train))])
        for idx, cl in enumerate(np.sort(np.unique(y_train))): # plot train samples
            plt.scatter(X_train[y_train == cl, 0], X_train[y_train == cl, 1], color=cmap(idx), marker='*')
        for idx, cl in enumerate(np.sort(np.unique(y_test))): # plot test samples
            plt.scatter(X_test[y_test == cl, 0], X_test[y_test == cl, 1], color=cmap(idx), marker='x')
        # Plot gaussians
        centers, cov_matrices = rbfn.get_mixtures()
        for i in range(len(centers)):
            plt.scatter(centers[i][0], centers[i][1], color='black', marker='o', label='versicolor')
        for i in range(len(centers)):
            ellipse = get_ellipse(centers[i], cov_matrices[i, :, :])
            ax.add_patch(ellipse)

        plt.tight_layout()
        plt.show()

        # Make predictions
        y_pred_prob = rbfn.predict_proba(X_test)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        plot_decision_regions(X_train, X_test, y_train, y_test, rbfn, resolution=0.1)
        pass

    def mnist():
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Create and fit the Logistic Regression to
        rbfn = RadialBasisFunctionNetwork(link=0, random_state=None, n_iter=300, n_components=10, covariance_type="full",
                                          tol_gmm=1e-3,
                                          reg_covar=1e-06, max_iter_gmm=1, init_params="kmeans", penalty="l2",
                                          tol_logreg=1e-4, C=100,
                                          class_weight=None, solver="saga", max_iter_logreg=1,
                                          multi_class="multinomial", n_jobs=-1)
        rbfn.fit(X_train, y_train)

        # Make predictions
        y_pred_prob = rbfn.predict_proba(X_test)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        pass

    mnist()