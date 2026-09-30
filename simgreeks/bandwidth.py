"""Optimal ISD size (bandwidth) selection, Appendix A.2 of the paper.

The t = 0 regression of eq. (22) is a local polynomial regression (LPR) of
order p = M0 with a uniform kernel (plain OLS on all paths within the
bandwidth). Following Fan & Gijbels (1995b, 1996), the MSE-optimal local
bandwidth for the nu-th derivative is

    h_nu = [ (2 nu + 1) a_nu sigma^2(x0)
             / ( 2 (q - nu) b_nu^2 beta_q^2 N f(x0) ) ]^(1 / (2q + 1))      (A.1)

where q = p + 1 if p - nu is odd and q = p + 2 if p - nu is even (for a
symmetric kernel and f'(x0) = 0 the leading odd-order bias term vanishes),
a_nu = [S^-1 S* S^-1]_{nu,nu}, b_nu = [S^-1 c_q]_nu with S = (mu_{j+l}),
S* = (nu_{j+l}) and c_q = (mu_q, ..., mu_{q+p}), mu_j = int u^j K(u) du,
nu_j = int u^j K(u)^2 du, and beta_q = m^(q)(x0) / q!.

sigma^2(x0) and beta_q are estimated in two stages, as described in the paper:
  1. a global rule-of-thumb (ROT) bandwidth (A.2), obtained from a global
     polynomial fit of order p + 3;
  2. a local polynomial fit of order q with the ROT bandwidth to estimate
     sigma^2(x0) and beta_q (its leading coefficient), which are plugged into
     (A.1). Order q is the lowest order that identifies beta_q; higher orders
     make the estimate of beta_q, and hence alpha*, much noisier.
f(x0) is known exactly since we choose the ISD ourselves.
"""
from __future__ import annotations

from functools import lru_cache

from math import comb

import numpy as np


@lru_cache(maxsize=None)
def _kernel_constants(p: int, nu: int, kernel: str = "uniform"):
    """Return (q, a_nu, b_nu) for an order-p LPR with the given kernel."""
    if kernel != "uniform":
        raise ValueError("only the uniform (OLS) kernel is implemented")
    # uniform kernel K(u) = 1/2 on [-1, 1]
    mu = lambda j: 0.0 if j % 2 else 1.0 / (j + 1)            # noqa: E731
    nuj = lambda j: 0.0 if j % 2 else 0.5 / (j + 1)           # noqa: E731
    Smat = np.array([[mu(j + l) for l in range(p + 1)] for j in range(p + 1)])
    Sstar = np.array([[nuj(j + l) for l in range(p + 1)] for j in range(p + 1)])
    Sinv = np.linalg.inv(Smat)
    a = (Sinv @ Sstar @ Sinv)[nu, nu]
    q = p + 1 if (p - nu) % 2 == 1 else p + 2
    c = np.array([mu(q + j) for j in range(p + 1)])
    b = (Sinv @ c)[nu]
    return q, a, b


def _poly_fit_scaled(X, Y, x0, h, order):
    """OLS fit of Y on ((X - x0)/h)^k, k=0..order. Returns (coef_in_x, rss, n)."""
    u = (X - x0) / h
    A = np.vander(u, order + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ c
    beta = c / h ** np.arange(order + 1)   # coefficients of (X - x0)^k
    return beta, float(resid @ resid), len(Y)


def optimal_alpha(X: np.ndarray, Y: np.ndarray, x0: float, alpha: float,
                  f0: float, M0: int = 9, nu: int = 2,
                  alpha_max: float | None = None, debug: bool = False) -> float:
    """Estimate the optimal ISD size alpha* (= local bandwidth h_nu(x0)).

    X, Y   : initially dispersed states and the (value-function) payoffs
    alpha  : current ISD size (X lies in [x0 - alpha, x0 + alpha])
    f0     : design density of X at x0
    M0     : polynomial order used in the t = 0 regression
    nu     : derivative the bandwidth is optimised for (0 price, 1 Delta, 2 Gamma)
    alpha_max : optional cap on alpha* (keeps rescaled initial prices positive)
    """
    N = len(X)
    p = M0
    q, a, b = _kernel_constants(p, nu)
    num_const = (2 * nu + 1) * a
    den_const = 2 * (q - nu) * b ** 2

    # --- Stage 1: global ROT bandwidth (A.2), pilot fit of order p + 3 ------
    order_g = max(p + 3, q + 1)
    beta_g, rss_g, n_g = _poly_fit_scaled(X, Y, x0, alpha, order_g)
    sigma2_g = rss_g / max(n_g - (order_g + 1), 1)
    # q-th derivative / q! of the global polynomial evaluated at each X_j
    k = np.arange(q, order_g + 1)
    coefs = np.array([comb(int(kk), q) * beta_g[kk] for kk in k])
    dx = X - x0
    beta_q_j = np.zeros_like(X)
    for c_, kk in zip(coefs, k):
        beta_q_j += c_ * dx ** (kk - q)
    int_w0 = 2.0 * alpha                     # w0 = indicator of the ISD support
    denom = den_const * np.sum(beta_q_j ** 2)
    if denom <= 0:
        h_rot = alpha
    else:
        h_rot = (num_const * sigma2_g * int_w0 / denom) ** (1.0 / (2 * q + 1))
    h_rot = float(np.clip(h_rot, 0.05 * alpha, alpha))

    # --- Stage 2: local fit with the ROT bandwidth ---------------------------
    order_l = q
    mask = np.abs(X - x0) <= h_rot
    if mask.sum() < 20 * (order_l + 1):      # too few points: widen
        idx = np.argsort(np.abs(X - x0))[: 20 * (order_l + 1)]
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True
        h_rot = float(np.max(np.abs(X[mask] - x0)))
    beta_l, rss_l, n_l = _poly_fit_scaled(X[mask], Y[mask], x0, h_rot, order_l)
    sigma2 = rss_l / max(n_l - (order_l + 1), 1)
    beta_q = beta_l[q]

    # --- Plug-in (A.1) --------------------------------------------------------
    denom = den_const * beta_q ** 2 * N * f0
    if denom <= 0 or not np.isfinite(denom):
        h = alpha_max if alpha_max is not None else alpha
    else:
        h = (num_const * sigma2 / denom) ** (1.0 / (2 * q + 1))
    if alpha_max is not None:
        h = min(h, alpha_max)
    if debug:
        print(f'h_rot={h_rot:.3f} sigma2={sigma2:.4f} beta_q={beta_q:.3e} h={h:.3f}')
    return float(h)
