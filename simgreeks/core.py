"""Core building blocks for Letourneau & Stentoft (2019), "Simulated Greeks for
American Options".

* Initial state dispersion (ISD), eq. (23) and (25)
* GBM path simulation with a separate "shock" matrix so that paths can be
  rescaled to a new initial value (Section 3.4)
* Least Squares Monte Carlo (Longstaff & Schwartz, 2001) returning both the
  naive pathwise discounted payoffs and the "value function" payoffs of
  Section 3.2
* The t = 0 cross-sectional polynomial regression, eq. (22), giving price,
  Delta and Gamma
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.polynomial import chebyshev as cheb


# --------------------------------------------------------------------------
# Option / model specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PutSpec:
    """American (Bermudan) put under Black-Scholes-Merton dynamics."""

    S0: float = 40.0
    K: float = 40.0
    sigma: float = 0.20
    r: float = 0.06
    d: float = 0.0          # continuous dividend yield
    T: float = 1.0
    steps_per_year: int = 50  # J = 50 exercise points per year

    @property
    def J(self) -> int:
        return max(1, int(round(self.steps_per_year * self.T)))

    @property
    def dt(self) -> float:
        return self.T / self.J

    def payoff(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - S, 0.0)


# --------------------------------------------------------------------------
# Initial state dispersion, eq. (23) and (25)
# --------------------------------------------------------------------------
def isd_kernel(U: np.ndarray, kernel: str = "epanechnikov") -> np.ndarray:
    """Map U in (0,1) to a symmetric kernel on [-1, 1].

    "epanechnikov": K_ISD(U) = 2 sin(asin(2U - 1) / 3), eq. (25), i.e. the
    inverse CDF of the Epanechnikov density 3/4 (1 - u^2).
    "uniform":      K_ISD(U) = 2U - 1.
    """
    if kernel == "epanechnikov":
        return 2.0 * np.sin(np.arcsin(2.0 * U - 1.0) / 3.0)
    if kernel == "uniform":
        return 2.0 * U - 1.0
    raise ValueError(f"unknown ISD kernel {kernel!r}")


def isd_density_at_center(alpha: float, kernel: str = "epanechnikov") -> float:
    """Density f(x0) of X = x0 + alpha K_ISD(U) at x0."""
    if kernel == "epanechnikov":
        return 0.75 / alpha
    if kernel == "uniform":
        return 0.5 / alpha
    raise ValueError(f"unknown ISD kernel {kernel!r}")


def isd_sample(N: int, x0: float, alpha: float, kernel: str = "epanechnikov",
               deterministic: bool = True, rng: np.random.Generator | None = None
               ) -> np.ndarray:
    """Initially dispersed state variables X_n = x0 + alpha K_ISD(U_n), eq. (23).

    With ``deterministic=True`` the U_n are equidistributed, U_n = (n - 1/2)/N
    ("D." in Table 9); otherwise they are i.i.d. uniform ("R." in Table 9).
    """
    if deterministic:
        U = (np.arange(N) + 0.5) / N
    else:
        U = rng.random(N)
    return x0 + alpha * isd_kernel(U, kernel)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def simulate_growth(spec: PutSpec, N: int, rng: np.random.Generator) -> np.ndarray:
    """Gross growth factors G[:, j] = S(t_j) / S(0), shape (N, J + 1).

    Paths are S = X[:, None] * G, which makes the "rescaling" of Section 3.4
    (dividing the path by a constant) trivial: reuse G with a new X.
    """
    J, dt = spec.J, spec.dt
    drift = (spec.r - spec.d - 0.5 * spec.sigma ** 2) * dt
    vol = spec.sigma * np.sqrt(dt)
    logG = np.empty((N, J + 1))
    logG[:, 0] = 0.0
    np.cumsum(drift + vol * rng.standard_normal((N, J)), axis=1, out=logG[:, 1:])
    return np.exp(logG)


# --------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------
def _cheb_fit(x: np.ndarray, y: np.ndarray, order: int):
    """Least-squares polynomial fit of order ``order`` in a Chebyshev basis on
    the data range (numerically stable for orders up to ~15)."""
    lo, hi = x.min(), x.max()
    if hi <= lo:
        hi = lo + 1e-12
    z = (2.0 * x - (lo + hi)) / (hi - lo)
    A = cheb.chebvander(z, order)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, lo, hi


def _cheb_eval(x: np.ndarray, fit) -> np.ndarray:
    coef, lo, hi = fit
    z = (2.0 * x - (lo + hi)) / (hi - lo)
    return cheb.chebval(z, coef)


@dataclass
class LSMResult:
    Y_naive: np.ndarray   # discounted pathwise payoff from the estimated tau, eq. (24)
    Y_vf: np.ndarray      # discounted value function from t_1 (Section 3.2)


def lsm(spec: PutSpec, S: np.ndarray, M_tau: int = 9) -> LSMResult:
    """Longstaff-Schwartz backward induction on paths S (N x (J+1)), S[:,0]=X.

    Exercise decisions at t_1..t_{J-1} use a cross-sectional regression of order
    ``M_tau`` on in-the-money paths only, eq. (5)-(6). No exercise at t_0.

    Value function variant (Section 3.2): at t_1 the payoff is replaced by
    V(t_1) = Z(t_1) if the LSM rule exercises and the estimated continuation
    value otherwise. The continuation value used here is fitted on *all* paths
    at t_1 so that it is also defined for out-of-the-money paths (their
    exercise value is zero and they are never exercised).
    """
    N, Jp1 = S.shape
    J = Jp1 - 1
    disc = np.exp(-spec.r * spec.dt)
    cash = spec.payoff(S[:, J])
    Y_vf = None
    for j in range(J - 1, 0, -1):
        cash *= disc                      # now valued at t_j
        Sj = S[:, j]
        ex_val = spec.payoff(Sj)
        itm = ex_val > 0.0
        exercise = np.zeros(N, dtype=bool)
        if itm.sum() > M_tau + 1:
            fit = _cheb_fit(Sj[itm], cash[itm], M_tau)
            cont = _cheb_eval(Sj[itm], fit)
            exercise[itm] = ex_val[itm] >= cont
        if j == 1:
            fit_all = _cheb_fit(Sj, cash, M_tau)
            V1 = np.where(exercise, ex_val, _cheb_eval(Sj, fit_all))
            Y_vf = disc * V1
        cash = np.where(exercise, ex_val, cash)
    if J == 1:  # European-like degenerate case
        cash_t1 = spec.payoff(S[:, 1])
        return LSMResult(disc * cash_t1, disc * cash_t1)
    return LSMResult(disc * cash, Y_vf)


def greeks_regression(X: np.ndarray, Y: np.ndarray, x0: float, M0: int = 9):
    """OLS of Y on (X - x0)^i, i = 0..M0 (eq. 22).

    Returns (price, delta, gamma) = (P*(0), P*(1), P*(2)); the regression is run
    on u = (X - x0)/h, h = max|X - x0|, and coefficients are rescaled.
    """
    h = np.max(np.abs(X - x0))
    u = (X - x0) / h
    A = np.vander(u, M0 + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    price = c[0]
    delta = c[1] / h
    gamma = 2.0 * c[2] / h ** 2 if M0 >= 2 else np.nan
    return price, delta, gamma


def poly_derivative_coef(i: int) -> float:
    return float(factorial(i))
