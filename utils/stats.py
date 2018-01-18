import numpy as np
from scipy.stats import multivariate_normal

def mult_gauss_pdf(X, mean, cov, log=False):

    dim = mean.size

    if cov.ndim == 1:
        det = np.prod(cov)
        if det == 0:
            raise ValueError("singular matrix")

        const = - 0.5 * ( dim * np.log(2 * np.pi) + np.log(det))
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
