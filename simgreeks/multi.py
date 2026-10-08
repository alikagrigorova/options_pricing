"""Price and Greeks with respect to S, sigma, r and d from a multivariate initial
state dispersion. NOT PART OF THE PAPER: it extends the paper's value-function
method (Section 3.2) and 2-step method (Section 3.4) from S to the model
parameters.

A label is built from independent runs, one per group of dispersed coordinates
(default: (S, sigma), (S, r), (S, d)). In a run, each path n starts from its
own z_n, e.g. for the group (S, sigma)

    z_n = z0 + (alpha_S k_S(U_n1), alpha_sigma k_p(U_n2), 0, 0),

with U_n a scrambled Sobol point (product design), k_S the Epanechnikov map of
eq. (25) and k_p the map of MultiConfig.param_kernel (uniform by default). The
parameters stay constant along the path,

    S_n(t_j) = S_n exp((r_n - d_n - sigma_n^2 / 2) t_j + sigma_n W_n(t_j)),

and cash flows are discounted with each path's own r_n. The parameters are
part of the Markov state, so the LSM regressions (exercise rule and the t_1
value function) use a basis in (S, dispersed parameters).

Value function and control variate. With control_variate=True (default) the
t_1 continuation value is C_1 = E_1 + pi_1: E_1 the closed-form European put
for the path's own state and parameters, pi_1 the regression of the
early-exercise premium cash flow. The label is
    Y_n = e^{-r_n dt} max(Z(t_1) - E_1, pi_1) = e^{-r_n dt} (V(t_1) - E_1),
whose conditional mean is P_American(z) - P_European(z); the closed-form
European Greeks at z0 are added back.

Taylor regression at t = 0. Y is regressed on monomials of x_k = (z_k - z0_k)/h_k.
The coefficient c of x_S^i x_sigma^a1 x_r^a2 x_d^a3 gives
    d^(i+|a|) P / dS^i dsigma^a1 dr^a2 dd^a3 = i! a1! a2! a3! c / (h_S^i h_sigma^a1 h_r^a2 h_d^a3),
with the terms i + weight |a| <= M and |a| <= param_degree.

Each run has two phases on independent paths:

  1. Pilot (N_pilot paths). The coordinates in MultiConfig.select (S and
     sigma) are dispersed widely: alpha0_S = c0_S S0 (the paper's alpha = 10
     at S0 = 40) and alpha0_sigma = c0_sigma sigma; the others by pilot_widen
     times their fixed width. The pilot
       * fits the exercise rule (exercise_rule="pilot"), and
       * chooses each selected width with the paper's selector (Appendix A.2,
         local order M = nu + 1, see methods.SELECTOR_ORDERS) applied to the
         pilot's partial residuals for that coordinate, i.e. the label minus
         the Taylor terms of the other coordinates. The target derivative is
         the Gamma for S (nu = 2) and the Volga for sigma (nu = 2).
  2. Main (N paths). The coordinates are dispersed with the chosen widths, the
     pilot's exercise rule is applied, and the Taylor regression gives the
     Greeks.

Why the pilot. Anything estimated on the same paths as the Greeks biases them:
  * an in-sample exercise rule has foresight, which raises the value most
    where the regression's leverage is largest, the edges of the box. The
    parameters do not diffuse, so this is U-shaped in each parameter and biased
    Volga upwards (+50 % median over the check grid). The uniform design and the
    wider pilot box keep the out-of-sample rule's error flat over the main box;
    an out-of-sample rule only makes the price a lower bound (a small low bias).
  * a width chosen on the same paths is smallest when the noise makes the data
    look curved, which is when the Greek estimate has that noise: in the
    paper's 2-step method this biases the Gamma of in-the-money options by
    +2.6 % (Tables 5-9 pooled). Independent pilot paths remove the correlation.

widths="fixed" skips the selection and uses the placeholder widths
alpha_S = c_S S0 sigma sqrt(T), alpha_sigma = c_sigma sigma, alpha_r, alpha_d.

Price, Delta and Gamma are averaged over the runs; each parameter Greek comes
from the run that disperses its parameter; Theta follows from the PDE identity
(core.theta_pde). With american_t0=True (default), an option whose continuation
value is at or below K - S0 is exercised at t_0: price K - S0, Delta -1 and all
other Greeks 0.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import ndtr
from scipy.stats import qmc

from .core import PutSpec, isd_kernel, theta_pde
from .reference import bs_put
from .selector import ALPHA_CAP, select_alpha

DIMS = ("S", "sigma", "r", "d")
PARAMS = ("sigma", "r", "d")
STYLES = ("bermudan", "european")
EXERCISE_RULES = ("pilot", "insample")
VF_RULES = ("max", "rule")
PARAM_KERNELS = ("uniform", "epanechnikov")
WIDTHS = ("selector", "fixed")
NU = {"S": 2, "sigma": 2}          # derivative each selected width is optimised for
GROUPS = (("S", "sigma"), ("S", "r"), ("S", "d"))
# Vera (d2P / dsigma dr) needs sigma and r in one run: add the group
# ("S", "sigma", "r"), which takes about 40 % of the time of a 4-run label.

# Greek name -> multi-index (i_S, a_sigma, a_r, a_d) of the t = 0 polynomial
GREEKS = {
    "price": (0, 0, 0, 0),
    "delta": (1, 0, 0, 0),
    "gamma": (2, 0, 0, 0),
    "vega": (0, 1, 0, 0),
    "volga": (0, 2, 0, 0),
    "rho": (0, 0, 1, 0),
    "rho_d": (0, 0, 0, 1),
    "vanna": (1, 1, 0, 0),       # d2P / dS dsigma
    "vera": (0, 1, 1, 0),        # d2P / dsigma dr
    "delta_r": (1, 0, 1, 0),     # d2P / dS dr
    "delta_d": (1, 0, 0, 1),     # d2P / dS dd
}
OUTPUTS = tuple(GREEKS) + ("theta",)


@dataclass(frozen=True)
class MultiConfig:
    N: int = 100_000               # main paths per run (one run per group)
    groups: tuple = GROUPS         # dispersed coordinates of each run; each contains "S"
    # widths of the dispersion
    widths: str = "selector"       # "selector" (pilot chooses `select`) or "fixed"
    select: tuple = ("S", "sigma")  # coordinates whose width the selector chooses
    c0_S: float = 0.25             # pilot alpha0_S = c0_S S0
    c0_sigma: float = 0.6          # pilot alpha0_sigma = c0_sigma sigma
    c_S: float = 0.6               # fixed alpha_S = c_S S0 sigma sqrt(T)   (placeholder)
    c_sigma: float = 0.25          # fixed alpha_sigma = c_sigma sigma      (placeholder)
    alpha_r: float = 0.02          # fixed alpha_r                          (placeholder)
    alpha_d: float = 0.02          # fixed alpha_d                          (placeholder)
    param_kernel: str = "uniform"  # ISD map of sigma, r, d (S keeps eq. 25's Epanechnikov)
    # pilot
    exercise_rule: str = "pilot"   # "pilot": fitted on the pilot paths; "insample"
    N_pilot: int | None = None     # pilot paths (default N)
    pilot_widen: float = 1.3       # pilot box >= pilot_widen x main box, for each parameter
    # regressions
    M_tau: int = 9                 # S-order of the exercise and t_1 bases
    M0: int = 9                    # S-order of the t = 0 basis
    weight: int = 3                # exercise-rule basis: terms with i + weight |a| <= M_tau
    weight_vf: int = 1             # same for the t_1 value-function and t = 0 bases
    param_degree: int = 3          # all bases: |a| <= param_degree
    style: str = "bermudan"        # "bermudan" (LSM) or "european" (no early exercise)
    control_variate: bool = True   # European control variate (see module doc)
    vf_rule: str = "rule"          # V(t_1) by the exercise rule (default) or max(Z, C_1); see lsm_multi
    american_t0: bool = True       # exercise at t_0 when optimal (bermudan only)

    def __post_init__(self):
        for dims in self.groups:
            if "S" not in dims or not set(dims) <= set(DIMS):
                raise ValueError(f"each group must contain 'S' and be a subset of {DIMS}")
        for name, value, allowed in (("style", self.style, STYLES),
                                     ("param_kernel", self.param_kernel, PARAM_KERNELS),
                                     ("exercise_rule", self.exercise_rule, EXERCISE_RULES),
                                     ("widths", self.widths, WIDTHS),
                                     ("vf_rule", self.vf_rule, VF_RULES)):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}")
        if not set(self.select) <= set(NU):
            raise ValueError(f"select must be a subset of {tuple(NU)}")
        if self.pilot_widen < 1:
            raise ValueError("pilot_widen must be >= 1")
        if not 0 < self.c0_sigma < 1 or not 0 < self.c_sigma * self.pilot_widen < 1:
            raise ValueError("c0_sigma and c_sigma x pilot_widen must be in (0, 1) "
                             "to keep sigma positive")

    @property
    def uses_pilot(self) -> bool:
        return self.widths == "selector" or (self.style == "bermudan"
                                             and self.exercise_rule == "pilot")


def _center(spec: PutSpec) -> dict:
    return {"S": spec.S0, "sigma": spec.sigma, "r": spec.r, "d": spec.d}


# --------------------------------------------------------------------------
# Widths of the dispersion
# --------------------------------------------------------------------------
def fixed_widths(spec: PutSpec, cfg: MultiConfig, dims) -> dict:
    """Placeholder widths alpha_k (see MultiConfig)."""
    a = {"S": min(cfg.c_S * spec.S0 * spec.sigma * np.sqrt(spec.T), ALPHA_CAP * spec.S0),
         "sigma": cfg.c_sigma * spec.sigma, "r": cfg.alpha_r, "d": cfg.alpha_d}
    return {k: a[k] for k in DIMS if k in dims}


def _selected(cfg: MultiConfig, dims) -> list:
    return [k for k in DIMS if k in dims and k in cfg.select] if cfg.widths == "selector" else []


def pilot_widths(spec: PutSpec, cfg: MultiConfig, dims) -> dict:
    """Pilot widths: alpha0 for the selected coordinates, the fixed widths
    otherwise (parameters widened by pilot_widen; S diffuses and is not)."""
    a0 = {"S": cfg.c0_S * spec.S0, "sigma": cfg.c0_sigma * spec.sigma}
    out = {}
    for k, a in fixed_widths(spec, cfg, dims).items():
        if k in _selected(cfg, dims):
            out[k] = a0[k]
        else:
            out[k] = a if k == "S" else cfg.pilot_widen * a
    return out


def select_widths(spec: PutSpec, cfg: MultiConfig, z: dict, Y: np.ndarray,
                  alphas0: dict) -> dict:
    """Main widths from the pilot data (z, Y) dispersed with alphas0: the
    selector (Appendix A.2, local order nu + 1) on each selected coordinate's
    partial residuals; fixed widths for the other coordinates.

    Caps: alpha_S* <= min(alpha0_S, ALPHA_CAP S0); a parameter's
    alpha* <= alpha0 / pilot_widen, so the pilot's exercise rule covers the
    main box with a margin."""
    dims = tuple(alphas0)
    out = fixed_widths(spec, cfg, dims)
    selected = _selected(cfg, dims)
    if not selected:
        return out
    fit = taylor_fit(spec, cfg, z, Y)
    center = _center(spec)
    for k in selected:
        kernel = "epanechnikov" if k == "S" else cfg.param_kernel
        a = select_alpha(z[k], partial_residual(fit, Y, k), center[k], alphas0[k], kernel,
                         M=NU[k] + 1, nu=NU[k])["alpha_star"]
        cap = min(alphas0[k], ALPHA_CAP * spec.S0) if k == "S" else alphas0[k] / cfg.pilot_widen
        out[k] = float(min(a, cap))
    return out


def rule_widths(cfg: MultiConfig, alphas: dict) -> dict:
    """Widths of the pilot paths the exercise rule is fitted on, given the main
    widths: the parameters' box is pilot_widen times wider (they do not
    diffuse, see module doc); S is not widened."""
    return {k: a if k == "S" else cfg.pilot_widen * a for k, a in alphas.items()}


def rescale(spec: PutSpec, z: dict, alphas_from: dict, alphas_to: dict) -> dict:
    """Shrink the dispersion of z from alphas_from to alphas_to around z0, as in
    the paper's 2-step method (Section 3.4). Paths are rebuilt from W and z, so
    the rescaled states give exact paths for their new values with the same
    shocks."""
    center = _center(spec)
    out = dict(z)
    for k, a in alphas_to.items():
        out[k] = center[k] + (a / alphas_from[k]) * (z[k] - center[k])
    return out


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def isd_multi(spec: PutSpec, cfg: MultiConfig, alphas: dict, rng: np.random.Generator,
              N: int | None = None) -> dict:
    """Initial states z_n on scrambled Sobol points (product design) with the
    widths ``alphas`` (one per dispersed coordinate): S by the Epanechnikov map
    of eq. (25), the parameters by cfg.param_kernel."""
    N = N or cfg.N
    with warnings.catch_warnings():           # N need not be a power of 2 here
        warnings.simplefilter("ignore", UserWarning)
        U = qmc.Sobol(d=len(alphas), scramble=True, seed=rng).random(N)
    U = np.clip(U, 0.5 / N, 1 - 0.5 / N)
    center = _center(spec)
    z = {k: np.full(N, v) for k, v in center.items()}
    for col, (k, a) in enumerate(alphas.items()):
        kernel = "epanechnikov" if k == "S" else cfg.param_kernel
        z[k] = center[k] + a * isd_kernel(U[:, col], kernel)
    return z


def brownian(spec: PutSpec, N: int, rng: np.random.Generator) -> np.ndarray:
    """W_n(t_j), j = 1..J, shape (N, J). Paths for any parameters follow from it."""
    W = rng.standard_normal((N, spec.J)) * np.sqrt(spec.dt)
    return np.cumsum(W, axis=1, out=W)


def stock_at(spec: PutSpec, z: dict, W: np.ndarray, j: int) -> np.ndarray:
    """S_n(t_j) for the per-path parameters z (j >= 1)."""
    t = j * spec.dt
    return z["S"] * np.exp((z["r"] - z["d"] - 0.5 * z["sigma"] ** 2) * t
                           + z["sigma"] * W[:, j - 1])


def european_put(S, K, tau, sigma, r, d):
    """Black-Scholes-Merton European put value, vectorised over paths."""
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - d) * tau) / sq + 0.5 * sq
    return K * np.exp(-r * tau) * ndtr(sq - d1) - S * np.exp(-d * tau) * ndtr(-d1)


# --------------------------------------------------------------------------
# Bases: (S-part of order <= M) x (parameter monomials of degree <= param_degree)
# --------------------------------------------------------------------------
def basis_terms(M: int, n_params: int, weight: int, param_degree: int):
    """Multi-indices (i, a) with i + weight |a| <= M and |a| <= param_degree."""
    terms = []
    for deg in range(min(param_degree, M // weight) + 1):
        for a in itertools.product(range(deg + 1), repeat=n_params):
            if sum(a) == deg:
                terms += [(i, a) for i in range(M - weight * deg + 1)]
    return terms


def monomials(P: np.ndarray, terms) -> tuple[np.ndarray, list]:
    """Parameter monomials prod_k P[:, k]^a_k for the distinct a in terms, as
    columns (Fortran order), and the (i, column) of each term."""
    exps = list(dict.fromkeys(a for _, a in terms))
    mono = np.empty((P.shape[0], len(exps)), order="F")
    for m, a in enumerate(exps):
        mono[:, m] = np.prod(P ** np.array(a), axis=1) if len(a) else 1.0
    col = {a: m for m, a in enumerate(exps)}
    return mono, [(i, col[a]) for i, a in terms]


def design(s_part: np.ndarray, mono: np.ndarray, index) -> np.ndarray:
    """Columns s_part[:, i] * mono[:, m] for (i, m) in index (Fortran order)."""
    s_part = np.asfortranarray(s_part)
    A = np.empty((s_part.shape[0], len(index)), order="F")
    for col, (i, m) in enumerate(index):
        np.multiply(s_part[:, i], mono[:, m], out=A[:, col])
    return A


def _solve(A, y):
    """Least squares via the normal equations (the bases are well conditioned),
    falling back to lstsq."""
    try:
        return cho_solve(cho_factor(A.T @ A, check_finite=False), A.T @ y, check_finite=False)
    except LinAlgError:
        return np.linalg.lstsq(A, y, rcond=None)[0]


def _design_s(S, lo, hi, mono, index, M):
    return design(cheb.chebvander((2.0 * S - (lo + hi)) / (hi - lo), M), mono, index)


def _fitted(S, mono, index, y, M):
    """Fitted values of the regression of y on Chebyshev(S) x parameter monomials."""
    lo, hi = S.min(), max(S.max(), S.min() + 1e-12)
    A = _design_s(S, lo, hi, mono, index, M)
    return A @ _solve(A, y)


def _fit(S, mono, index, y, M):
    """The same regression, returned as (coefficients, lo, hi) for later use."""
    lo, hi = S.min(), max(S.max(), S.min() + 1e-12)
    return _solve(_design_s(S, lo, hi, mono, index, M), y), lo, hi


def _predict(fit, S, mono, index, M):
    """Evaluate a stored fit; S is clipped to the fit's data range."""
    coef, lo, hi = fit
    return _design_s(np.clip(S, lo, hi), lo, hi, mono, index, M) @ coef


def _params(spec, z, scale):
    """The dispersed parameters (the keys of scale besides S), centred and
    divided by scale[k], in PARAMS order."""
    center = _center(spec)
    params = [k for k in PARAMS if k in scale]
    P = np.column_stack([(z[k] - center[k]) / scale[k] for k in params]) \
        if params else np.empty((len(z["S"]), 0))
    return params, P


# --------------------------------------------------------------------------
# LSM and the t = 0 Taylor regression
# --------------------------------------------------------------------------
def lsm_multi(spec: PutSpec, cfg: MultiConfig, z: dict, W: np.ndarray, scale: dict,
              rules: dict | None = None, value_function: bool = True):
    """LSM with per-path parameters. Returns (Y_naive, Y_vf, rules), Y_naive and
    Y_vf discounted to t = 0.

    ``scale`` has one width per dispersed coordinate: the parameters enter the
    bases as (z_k - z0_k) / scale[k]. Stored rules must be applied with the
    scale they were fitted with.
    Exercise rule at t_1..t_{J-1}: in-the-money regression on (S, parameters),
    fitted on these paths if ``rules`` is None, else the stored fits
    ``rules[j]`` are applied (out of sample). ``rules`` returned are the fits used.
    Y_vf = e^{-r_n dt} V(t_1) with V = C_1 ("european") or, for "bermudan",
    V = max(Z, C_1) (cfg.vf_rule="max") or V = Z where the exercise rule at t_1
    exercises and C_1 elsewhere ("rule"); C_1 is fitted on all paths; with cfg.control_variate, Y_vf is
    the premium label e^{-r_n dt} (V(t_1) - E_1) (see module doc). With
    ``value_function=False`` Y_vf is None.
    """
    J = spec.J
    N = len(z["S"])
    params, P = _params(spec, z, scale)
    mono, index = monomials(P, basis_terms(cfg.M_tau, len(params), cfg.weight,
                                           cfg.param_degree))
    mono_vf, index_vf = monomials(P, basis_terms(cfg.M_tau, len(params), cfg.weight_vf,
                                                 cfg.param_degree))
    disc = np.exp(-z["r"] * spec.dt)
    cash = spec.payoff(stock_at(spec, z, W, J))
    euro_payoff = cash.copy()
    if J == 1:
        return disc * cash, disc * cash, {}
    Y_vf = None
    fits = {}
    for j in range(J - 1, 0, -1):
        cash *= disc
        Sj = stock_at(spec, z, W, j)
        ex = spec.payoff(Sj)
        exercise = np.zeros(N, dtype=bool)
        if cfg.style == "bermudan":
            itm = ex > 0.0
            if rules is not None:
                fits[j] = rules.get(j)
            elif itm.sum() > 5 * len(index):
                fits[j] = _fit(Sj[itm], mono[itm], index, cash[itm], cfg.M_tau)
            else:
                fits[j] = None
            if fits[j] is not None and itm.any():
                exercise[itm] = ex[itm] >= _predict(fits[j], Sj[itm], mono[itm], index,
                                                    cfg.M_tau)
        if j == 1 and value_function:
            if cfg.control_variate:
                tau = spec.T - spec.dt
                E1 = european_put(Sj, spec.K, tau, z["sigma"], z["r"], z["d"])
                D1 = np.exp(-z["r"] * tau) * euro_payoff
                prem = _fitted(Sj, mono_vf, index_vf, cash - D1, cfg.M_tau)
                if cfg.style == "european":
                    Y_vf = disc * prem
                elif cfg.vf_rule == "rule":
                    Y_vf = disc * np.where(exercise, ex - E1, prem)
                else:
                    Y_vf = disc * np.maximum(ex - E1, prem)
            else:
                cont = _fitted(Sj, mono_vf, index_vf, cash, cfg.M_tau)
                if cfg.style == "european":
                    Y_vf = disc * cont
                elif cfg.vf_rule == "rule":
                    Y_vf = disc * np.where(exercise, ex, cont)
                else:
                    Y_vf = disc * np.maximum(ex, cont)
        cash = np.where(exercise, ex, cash)
    return disc * cash, Y_vf, fits


@dataclass
class TaylorFit:
    """t = 0 regression of Y on the Taylor monomials of (z - z0) (module doc)."""
    params: list                 # dispersed parameters, in PARAMS order
    terms: list                  # multi-indices (i, a), a over params
    coef: np.ndarray
    h: dict                      # half-width of the data in each dispersed coordinate
    A: np.ndarray                # design matrix


def taylor_fit(spec: PutSpec, cfg: MultiConfig, z: dict, Y: np.ndarray,
               dims=None) -> TaylorFit:
    """Regress Y on the Taylor monomials of the coordinates ``dims`` (default:
    those that vary in z)."""
    center = _center(spec)
    if dims is None:
        dims = [k for k in DIMS if np.ptp(z[k]) > 0]
    h = {k: float(np.max(np.abs(z[k] - center[k]))) for k in DIMS if k in dims}
    params, P = _params(spec, z, h)
    terms = basis_terms(cfg.M0, len(params), cfg.weight_vf, cfg.param_degree)
    mono, mindex = monomials(P, terms)
    A = design(np.vander((z["S"] - spec.S0) / h["S"], cfg.M0 + 1, increasing=True),
               mono, mindex)
    return TaylorFit(params, terms, np.linalg.lstsq(A, Y, rcond=None)[0], h, A)


def greeks_from(fit: TaylorFit, add: dict | None = None) -> dict:
    """The Greeks the fit identifies (others NaN), plus ``add``."""
    index = {t: n for n, t in enumerate(fit.terms)}
    out = {}
    for name, (i, *a_full) in GREEKS.items():
        a = tuple(ak for k, ak in zip(PARAMS, a_full) if k in fit.params)
        if any(ak for k, ak in zip(PARAMS, a_full) if k not in fit.params) or (i, a) not in index:
            out[name] = np.nan
            continue
        scale = factorial(i) * fit.h["S"] ** -i
        for k, ak in zip(fit.params, a):
            scale *= factorial(ak) * fit.h[k] ** -ak
        out[name] = float(fit.coef[index[(i, a)]] * scale) + (add or {}).get(name, 0.0)
    return out


def partial_residual(fit: TaylorFit, Y: np.ndarray, k: str) -> np.ndarray:
    """Y minus the fitted Taylor terms that involve any coordinate other than k."""
    def involves_other(i, a):
        others = [i > 0] if k != "S" else []
        return any(others) or any(ak > 0 for p, ak in zip(fit.params, a) if p != k)
    cols = [n for n, (i, a) in enumerate(fit.terms) if involves_other(i, a)]
    return Y - fit.A[:, cols] @ fit.coef[cols]


def greeks_multi(spec: PutSpec, cfg: MultiConfig, dims, z: dict, Y: np.ndarray,
                 add: dict | None = None) -> dict:
    """t = 0 regression over the coordinates ``dims`` and its Greeks (others NaN)."""
    return greeks_from(taylor_fit(spec, cfg, z, Y, dims), add)


# --------------------------------------------------------------------------
# One run, one label
# --------------------------------------------------------------------------
def run_group(spec: PutSpec, cfg: MultiConfig, dims, seed) -> dict:
    """One run with the coordinates ``dims`` dispersed: pilot, then main (module
    doc). Returns the Greeks this group identifies and the widths it used."""
    rng = np.random.default_rng(seed)
    alphas = fixed_widths(spec, cfg, dims)
    scale, rules = alphas, None
    if cfg.uses_pilot:
        alphas0 = pilot_widths(spec, cfg, dims)
        Np = cfg.N_pilot or cfg.N
        zp = isd_multi(spec, cfg, alphas0, rng, Np)
        Wp = brownian(spec, Np, rng)
        _, Yp, fits = lsm_multi(spec, cfg, zp, Wp, alphas0,
                                value_function=cfg.widths == "selector")
        scale = alphas0
        if cfg.widths == "selector":
            alphas = select_widths(spec, cfg, zp, Yp, alphas0)
            if cfg.style == "bermudan" and cfg.exercise_rule == "pilot":
                scale = rule_widths(cfg, alphas)
                zp = rescale(spec, zp, alphas0, scale)
                fits = lsm_multi(spec, cfg, zp, Wp, scale, value_function=False)[2]
        if cfg.style == "bermudan" and cfg.exercise_rule == "pilot":
            rules = fits
        else:
            scale = alphas
        del zp, Wp, Yp
    z = isd_multi(spec, cfg, alphas, rng)
    _, Y, _ = lsm_multi(spec, cfg, z, brownian(spec, cfg.N, rng), scale, rules)
    out = greeks_multi(spec, cfg, dims, z, Y, bs_put(spec) if cfg.control_variate else None)
    out.update({f"alpha_{k}": float(v) for k, v in alphas.items()})
    return out


def label(spec: PutSpec, cfg: MultiConfig, seed) -> dict:
    """One estimate of the price and the Greeks at z0: one independent run per
    group; S-Greeks averaged over the runs; Theta from the PDE identity. Greeks
    no group identifies (Vera with the default groups) are left out. alpha_k
    is the width used for coordinate k, averaged over the runs that disperse it."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    runs = [run_group(spec, cfg, dims, child)
            for dims, child in zip(cfg.groups, ss.spawn(len(cfg.groups)))]
    out = {}
    for name in GREEKS:                # only the Greeks some group identifies
        vals = [r[name] for r in runs if not np.isnan(r[name])]
        if vals:
            out[name] = float(np.mean(vals))
    out["theta"], out["ex_region"] = theta_pde(spec, out["price"], out["delta"], out["gamma"])
    if cfg.american_t0 and cfg.style == "bermudan" and out["ex_region"]:
        # exercising at t_0 is optimal: V = K - S, so only Delta is non-zero
        out.update({k: 0.0 for k in OUTPUTS if k in out})
        out["price"], out["delta"] = spec.K - spec.S0, -1.0
    for k in DIMS:
        vals = [r[f"alpha_{k}"] for r in runs if f"alpha_{k}" in r]
        if vals:
            out[f"alpha_{k}"] = float(np.mean(vals))
    return out
