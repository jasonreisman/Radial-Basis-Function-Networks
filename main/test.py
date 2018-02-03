"""l-bfgs-b L1-Logistic Regression solver"""

# Author: Vlad Niculae <vlad@vene.ro>
# Suggested by Mathieu Blondel

from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
from scipy.special import expit


def _l1_logistic_loss_grad(w_extended, X, y, alpha):
    _, n_features = X.shape
    w = w_extended[:n_features] - w_extended[n_features:]

    yz = y * safe_sparse_dot(X, w)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(log_logistic(yz))
    # out += .5 * alpha * np.dot(w, w)  # L2
    out += alpha * w_extended.sum()  # L1, w_extended is non-negative

    z = expit(yz)
    z0 = (z - 1) * y

    grad = safe_sparse_dot(X.T, z0)
    grad = np.concatenate([grad, -grad])
    # grad += alpha * w  # L2
    grad += alpha  # L1

    return out, grad

class LbfgsL1Logistic(BaseEstimator, ClassifierMixin):

    def __init__(self, tol=1e-3, alpha=1.0):
        """Logistic Regression Lasso solved by L-BFGS-B
        Solves the same objective as sklearn.linear_model.LogisticRegression
        Parameters
        ----------
        alpha: float, default: 1.0
            The amount of regularization to use.
        tol: float, default: 1e-3
            Convergence tolerance for L-BFGS-B.
        """
        self.tol = tol
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape

        coef0 = np.zeros(2 * n_features)
        w, f, d = fmin_l_bfgs_b(_l1_logistic_loss_grad, x0=coef0, fprime=None,
                                pgtol=self.tol,
                                bounds=[(0, None)] * n_features * 2,
                                args=(X, y, self.alpha))
        self.coef_ = w[:n_features] - w[n_features:]

        return self

    def predict(self, X):
        return np.sign(safe_sparse_dot(X, self.coef_))


if __name__ == '__main__':
    from scipy.spatial.distance import jaccard
    from sklearn.linear_model import LogisticRegression
    from time import time

    # Generate data with known sparsity pattern
    n_samples, n_features, n_relevant = 100, 80, 20
    X = np.random.randn(n_samples, n_features)
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
    true_coef = np.zeros(n_features)
    nonzero_idx = np.random.randint(n_features, size=n_relevant)
    true_coef[nonzero_idx] = np.random.randn(n_relevant)
    y = np.dot(X, true_coef) + np.random.randn(n_samples) * 0.01

    # classification, note: y must be {-1, +1}
    y = np.sign(y)

    C = 1.0
    # Run this solver
    t0 = time()
    lasso_1 = LbfgsL1Logistic(alpha=1. / C, tol=1e-8).fit(X, y)
    t0 = time() - t0
    print("l-bfgs-b:  time = {:.4f}s acc = {:.8f}  ||w - w_true|| = {:.6f}  "
          "Jacc. sparsity = {:.2f}".format(t0, lasso_1.score(X, y),
            np.linalg.norm(lasso_1.coef_ - true_coef),
            jaccard(true_coef > 0, lasso_1.coef_ > 0)))
