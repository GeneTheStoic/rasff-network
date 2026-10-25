"""Sparse group lasso for logistic regression (proximal gradient, FISTA).

Penalty: lam * [ alpha * ||b||_1 + (1 - alpha) * sum_g sqrt(p_g) * ||b_g||_2 ]
alpha = 1 gives the lasso, alpha = 0 the group lasso (Simon, Friedman, Hastie and
Tibshirani, 2013). The intercept is not penalized.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _codes(groups):
    _, code = np.unique(np.asarray(groups), return_inverse=True)
    size = np.bincount(code)
    return code, np.sqrt(size)


def _prox(b, code, sqrt_size, step, lam, alpha):
    b = np.sign(b) * np.maximum(np.abs(b) - step * lam * alpha, 0)          # L1 part
    if alpha >= 1:
        return b
    norm = np.sqrt(np.bincount(code, weights=b * b, minlength=len(sqrt_size)))
    thr = step * lam * (1 - alpha) * sqrt_size
    scale = np.where(norm > thr, 1 - thr / np.maximum(norm, 1e-300), 0.0)    # group part
    return b * scale[code]


def fit(X, y, groups, lam, alpha, b0=None, max_iter=5000, tol=1e-5):
    n, p = X.shape
    code, sqrt_size = _codes(groups)
    X = sparse.csr_matrix(X)                                                  # indicator data are sparse
    XT = X.T.tocsr()
    X1 = sparse.hstack([np.ones((n, 1)), X]).tocsr()                          # intercept included
    smax = svds(X1, k=1, return_singular_vectors=False)[0]
    L = 0.25 * smax ** 2 / n + 1e-12                                          # Lipschitz constant
    step = 1 / L
    b = np.zeros(p) if b0 is None else b0[1:].copy()
    c = np.log(y.mean() / (1 - y.mean())) if b0 is None else b0[0]
    zb, zc, t = b.copy(), c, 1.0
    for _ in range(max_iter):
        r = _sigmoid(X @ zb + zc) - y
        gb, gc = XT @ r / n, r.mean()
        b_new = _prox(zb - step * gb, code, sqrt_size, step, lam, alpha)
        c_new = zc - step * gc
        # stopping rule: norm of the proximal gradient mapping (first-order optimality)
        gmap = np.sqrt(np.sum((zb - b_new) ** 2) + (zc - c_new) ** 2) / step
        # adaptive restart of the momentum (O'Donoghue and Candes, 2015)
        if np.dot(zb - b_new, b_new - b) + (zc - c_new) * (c_new - c) > 0:
            t = 1.0
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        zb = b_new + (t - 1) / t_new * (b_new - b)
        zc = c_new + (t - 1) / t_new * (c_new - c)
        b, c, t = b_new, c_new, t_new
        if gmap < tol:
            break
    return np.r_[c, b]


def predict(beta, X):
    return _sigmoid(X @ beta[1:] + beta[0])


def lambda_max(X, y, groups, alpha):
    """Smallest penalty (approximately) at which all coefficients are zero."""
    n = len(y)
    g = np.abs(X.T @ (y - y.mean()) / n)
    if alpha >= 1:
        return g.max()
    code, sqrt_size = _codes(groups)
    gnorm = np.sqrt(np.bincount(code, weights=g * g))
    return max((gnorm / sqrt_size).max(), g.max()) * 1.05
