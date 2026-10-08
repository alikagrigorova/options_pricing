"""Optimal ISD size alpha* (Appendix A.2 of the paper).

The selector follows the paper's three steps:

  1. Global rule-of-thumb bandwidth h_ROT, (A.2)-(A.3): a pilot polynomial of
     order M + 3 is fitted to all data; sigma^2 is its residual variance and
     F^(M+1) its (M+1)-th derivative, and
        h_ROT = [ (2nu+1) a sigma^2 int w0/f  /  (2 (M+1-nu) b_rot^2 sum_j beta_{M+1}(X_j)^2 w0(X_j)) ]^(1/(2M+3))
     with beta_{M+1}(x) = F^(M+1)(x) / (M+1)! (this is (A.2)-(A.3) with the
     (M+1)!^2 of C_{nu,M}(K) absorbed) and b_rot = int t^(M+1) K*_nu(t) dt.
  2. Local polynomial fit on |X - x0| <= h_ROT, giving sigma^2(x0) (residual
     variance) and beta_{M+1} (its leading coefficient).
  3. Plug-in (A.1):
        alpha* = [ (2nu+1) a sigma^2(x0) / (2 (M+1-nu) b^2 beta_{M+1}^2 N f(x0)) ]^(1/(2M+3)).

Kernel: uniform, since the t = 0 regression is plain OLS (eq. 21).
Constants (reading="literal", the default, as printed in the paper):
  a = (nu+1)-th diagonal element of Q^-1 Q* Q^-1,  b = (nu+1)-th diagonal
  element of Q,  b_rot = int t^(M+1) K*_nu(t) dt  (A.3).
For nu = 1 and odd M the (A.3) integral is zero, so h_ROT is infinite and the
local fit in step 2 uses all data.

reading="fg" is the Fan & Gijbels (1996) version of the same formulas:
b = [Q^-1 c]_nu = int t^q K*_nu, with q = M + 2 instead of M + 1 when M - nu
is even (the leading bias term vanishes for symmetric kernels).

Stated in the paper: the constants a, b (diagonal elements, as above); the
order-(M+3) global pilot "using all data" with sigma^2 its standardized RSS;
"w0 can be taken as the indicator function"; (A.2)-(A.3); "use the ROT to
locally fit a polynomial and estimate sigma^2(x0) and beta_{M+1}"; plug-in
(A.1); one pass; OLS = uniform kernel (eq. 21).

Not specified by the paper, chosen here:
  * w0 = indicator of |x - x0| <= 0.9 alpha0 (the interval is not given, and
    int w0 / f is infinite if w0 covers the whole Epanechnikov support);
  * the local fit has order q = M + 1 (the lowest order identifying beta_{M+1})
    and is unweighted within |X - x0| <= h_ROT (the paper recommends "a weighted
    regression" but gives no weights);
  * h_ROT is not clipped; for nu = 1 the printed (A.3) integral is zero, so
    h_ROT is infinite and the local fit uses all data;
  * alpha* is capped at 0.75 x0 only to keep rescaled prices positive;
  * reading="fg" (Fan & Gijbels' constants) is offered as an alternative.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb, log

import numpy as np

C_W0 = 0.9
READINGS = ("literal", "fg")


def _mu(j):                      # moments of the uniform kernel K = 1/2 on [-1, 1]
    return 0.0 if j % 2 else 1.0 / (j + 1)


def _nu2(j):                     # moments of K^2
    return 0.0 if j % 2 else 0.5 / (j + 1)


@lru_cache(maxsize=None)
def constants(M: int, nu: int, reading: str = "literal"):
    """Return dict(q, a, b, q_rot, b_rot) for order-M fits and derivative nu."""
    if reading not in READINGS:
        raise ValueError(f"unknown reading {reading!r}")
    Q = np.array([[_mu(j + l) for l in range(M + 1)] for j in range(M + 1)])
    Qs = np.array([[_nu2(j + l) for l in range(M + 1)] for j in range(M + 1)])
    Qi = np.linalg.inv(Q)
    a = float((Qi @ Qs @ Qi)[nu, nu])

    def equiv_moment(q):         # int t^q K*_nu(t) dt
        return float((Qi @ np.array([_mu(q + j) for j in range(M + 1)]))[nu])

    if reading == "literal":
        q = M + 1
        return dict(q=q, a=a, b=float(Q[nu, nu]), q_rot=q, b_rot=equiv_moment(q))
    q = M + 1 if (M - nu) % 2 == 1 else M + 2
    b = equiv_moment(q)
    return dict(q=q, a=a, b=b, q_rot=q, b_rot=b)


def isd_density_at(alpha0: float, isd_kernel: str) -> float:
    """f(x0) of the ISD X = x0 + alpha0 K_ISD(U)."""
    return (0.5 if isd_kernel == "uniform" else 0.75) / alpha0


def int_w0_over_f(alpha0: float, isd_kernel: str, c: float = C_W0) -> float:
    """int over |x - x0| <= c alpha0 of 1 / f_ISD(x) dx."""
    if isd_kernel == "uniform":                  # f = 1 / (2 alpha0)
        return 4.0 * c * alpha0 ** 2
    return alpha0 ** 2 / 0.75 * log((1 + c) / (1 - c))      # f = 0.75 (1 - u^2) / alpha0


def poly_fit(X, Y, x0, h, order):
    """OLS of Y on ((X - x0)/h)^k, k = 0..order. Returns (coefficients of
    (X - x0)^k, residual variance)."""
    u = (X - x0) / h
    A = np.vander(u, order + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r = Y - A @ c
    return c / h ** np.arange(order + 1), float(r @ r) / max(len(Y) - order - 1, 1)


def select_alpha(X, Y, x0: float, alpha0: float, isd_kernel: str = "epanechnikov",
                 M: int = 9, nu: int = 2, reading: str = "literal", glob=None) -> dict:
    """alpha* and the selector's intermediate quantities.

    X, Y   : initially dispersed states and the value-function payoffs
    alpha0 : ISD size used to generate X
    M      : polynomial order of the t = 0 regression (M0)
    nu     : derivative alpha* is optimised for (0 price, 1 Delta, 2 Gamma)
    glob   : optional precomputed global pilot poly_fit(X, Y, x0, alpha0, M + 3)
    """
    k = constants(M, nu, reading)
    q, qr, a = k["q"], k["q_rot"], k["a"]
    N = len(X)
    f0 = isd_density_at(alpha0, isd_kernel)
    dx = X - x0

    # Step 1: global pilot of order M + 3 and rule-of-thumb bandwidth
    beta_g, s2g = glob if glob is not None else poly_fit(X, Y, x0, alpha0, M + 3)
    Fq = np.zeros_like(X)                        # beta_{q_rot}(X_j) = F^(q_rot)(X_j) / q_rot!
    for kk in range(qr, len(beta_g)):
        Fq += comb(kk, qr) * beta_g[kk] * dx ** (kk - qr)
    w0 = np.abs(dx) <= C_W0 * alpha0
    den = 2 * (qr - nu) * k["b_rot"] ** 2 * float(np.sum(Fq[w0] ** 2))
    num = (2 * nu + 1) * a * s2g * int_w0_over_f(alpha0, isd_kernel)
    h_rot = (num / den) ** (1 / (2 * qr + 1)) if den > 0 else np.inf

    # Step 2: local fit of order q on the h_ROT window
    mask = np.abs(dx) <= h_rot
    if mask.sum() < 20 * (q + 1):
        mask = np.argsort(np.abs(dx))[: 20 * (q + 1)]
    Xl, Yl = X[mask], Y[mask]
    beta_l, s2 = poly_fit(Xl, Yl, x0, float(np.max(np.abs(Xl - x0))), q)
    beta_q = float(beta_l[q])

    # Step 3: plug-in (A.1)
    den = 2 * (q - nu) * k["b"] ** 2 * beta_q ** 2 * N * f0
    raw = ((2 * nu + 1) * a * s2 / den) ** (1 / (2 * q + 1)) if den > 0 else np.inf
    cap = 0.75 * x0
    return dict(alpha_star=float(min(raw, cap)), alpha_raw=float(raw), capped=bool(raw > cap),
                h_rot=float(h_rot), n_local=int(len(Xl)), beta_q=beta_q, sigma2_local=s2,
                sigma2_global=s2g, f_x0=f0, q=q, a=a, b=k["b"], b_rot=k["b_rot"],
                global_coefs=beta_g)
