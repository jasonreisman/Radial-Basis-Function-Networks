import numpy as np
from scipy import linalg

def log_multivariate_normal_density_diag(X, means, covars, reg=1e-6):
    """Compute Gaussian log-density at X for a diagonal model."""

    n_samples, n_dim = X.shape

    if (covars == 0).any():
        covars += reg

    log_dets = np.sum(np.log(covars), axis=1)

    lmnd = -0.5 * (n_dim * np.log(2 * np.pi) + log_dets + np.sum((means ** 2) / covars, 1) -
                  2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))

    return lmnd

def log_multivariate_normal_density_full(X, means, covars, reg=1.e-6):
    """Log probability for full covariance matrices."""

    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))

    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + reg * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - 0.5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def mult_gauss_pdf(X, mean, cov, log=False):

    dim = mean.size

    if cov.ndim == 1:
        det = np.prod(cov)
        if det == 0:
            raise ValueError("singular matrix")

        const = - 0.5 * (dim * np.log(2 * np.pi) + np.log(det))
        inv = 1.0 / cov
        X_c = X - mean
        result = - 0.5 * np.sum((X_c ** 2) * inv, axis=1)

    else:
        det = np.linalg.det(cov)
        if det == 0:
            raise ValueError("singular matrix")

        const = - 0.5 * (dim * np.log(2 * np.pi) + np.log(det))
        inv = np.linalg.inv(cov)
        X_c = X - mean
        result = - 0.5 * np.sum(np.dot(X_c, inv) * X_c, axis=1)

    result += const

    if log is True:
        return result
    else:
        return np.exp(result)
