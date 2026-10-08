"""Price and Greeks with respect to S, sigma, r and d from a multivariate initial
state dispersion. NOT PART OF THE PAPER: it extends the paper's value-function
method (Section 3.2) from S to the model parameters.

A label is built from independent runs, one per group of dispersed coordinates
(default: (S, sigma), (S, r), (S, d)). In a run, each path n starts from its
own z_n, e.g. for the group (S, sigma)

    z_n = z0 + (alpha_S k(U_n1), alpha_sigma k(U_n2), 0, 0),

with U_n a scrambled Sobol point (product design), k the Epanechnikov map of
eq. (25) for S and, by default, the uniform map k(u) = 2u - 1 for the
parameters (MultiConfig.param_kernel; see "Exercise rule" below for why). The
parameters stay constant along the path,

    S_n(t_j) = S_n exp((r_n - d_n - sigma_n^2 / 2) t_j + sigma_n W_n(t_j)),

and cash flows are discounted with each path's own r_n. The parameters are
part of the Markov state, so the LSM regressions (exercise rule and the t_1
value function) use a basis in (S, dispersed parameters). The value-function
labels Y_n = e^{-r_n dt} max(Z(t_1), C_1(S_n(t_1), theta_n)) carry the
parameter sensitivities through C_1.

At t = 0, Y is regressed on monomials of x_k = (z_k - z0_k) / h_k, a
multivariate Taylor polynomial of the price around z0. The coefficient c of
x_S^i x_sigma^a1 x_r^a2 x_d^a3 gives the derivative

    d^(i+|a|) P / dS^i dsigma^a1 dr^a2 dd^a3 = i! a1! a2! a3! c / (h_S^i h_sigma^a1 h_r^a2 h_d^a3).

The bases keep the terms with i + weight |a| <= M and |a| <= param_degree.

Exercise rule (exercise_rule="pilot", default). The exercise regressions are
fitted on an independent pilot set of paths whose parameter box is
pilot_widen times wider, and the stored rules are applied to the main paths.
Two effects bias the second derivatives in the parameters (Volga) when the
rule is fitted on the main paths themselves ("insample"):
  * the parameters do not diffuse, so at every exercise date the regression
    sees the ISD's parameter density; the Epanechnikov density vanishes at the
    edges of the box, where the fitted rule is then poorest;
  * an in-sample rule has foresight, which raises the value most where the
    regression's leverage is largest, i.e. at the edges of the box.
The second effect is U-shaped in each parameter and biased Volga upwards
(+50 % median over the check grid; the bias shrank with N). An out-of-sample
rule has no foresight: its error only makes the rule suboptimal, which lowers
the value (a small low bias in the price, the usual LSM lower bound). The
uniform design and the wider pilot box keep that error flat over the main box.

European control variate (control_variate=True): C_1 = E_1 + pi_1, with E_1
the European put value at t_1 for the path's own state and parameters
(closed form) and pi_1 the regression of the early-exercise premium cash flow.
The t = 0 regression is run on the premium label
    e^{-r_n dt} max(Z(t_1) - E_1, pi_1) = e^{-r_n dt} (V(t_1) - E_1),
whose conditional mean is P_American(z) - P_European(z), and the closed-form
European Greeks at z0 are added back.

Price, Delta and Gamma are averaged over the runs; each parameter Greek comes
from the run that disperses its parameter; Theta follows from the PDE identity
(core.theta_pde).

The dispersion sizes are PLACEHOLDERS (no selector yet), and each run uses the
whole sample in one pass at those sizes: with a fixed alpha*, the paper's
2-step method reduces to this single pass. The pilot paths only estimate the
exercise rule; they are not the paper's 2-step pilot.
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
from .selector import ALPHA_CAP

DIMS = ("S", "sigma", "r", "d")
PARAMS = ("sigma", "r", "d")
STYLES = ("bermudan", "european")
EXERCISE_RULES = ("pilot", "insample")
PARAM_KERNELS = ("uniform", "epanechnikov")
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
    N: int = 100_000               # paths per run (one run per group)
    groups: tuple = GROUPS         # dispersed coordinates of each run; each contains "S"
    c_S: float = 0.6               # alpha_S = c_S S0 sigma sqrt(T)   (placeholder)
    c_sigma: float = 0.25          # alpha_sigma = c_sigma sigma      (placeholder)
    alpha_r: float = 0.02          # (placeholder)
    alpha_d: float = 0.02          # (placeholder)
    M_tau: int = 9                 # S-order of the exercise and t_1 bases
    M0: int = 9                    # S-order of the t = 0 basis
    weight: int = 3                # exercise-rule basis: terms with i + weight |a| <= M_tau
    weight_vf: int = 1             # same for the t_1 value-function and t = 0 bases
    param_degree: int = 3          # all bases: |a| <= param_degree
    style: str = "bermudan"        # "bermudan" (LSM) or "european" (no early exercise)
    control_variate: bool = True   # European control variate (see module doc)
    american_t0: bool = False      # report American values when exercising at t_0 is optimal
    param_kernel: str = "uniform"  # ISD map of sigma, r, d (S keeps eq. 25's Epanechnikov)
    exercise_rule: str = "pilot"   # "pilot": fitted on independent paths; "insample"
    N_pilot: int | None = None     # pilot paths (default N)
    pilot_widen: float = 1.3       # pilot parameter box = pilot_widen x the run's box

    def __post_init__(self):
        for dims in self.groups:
            if "S" not in dims or not set(dims) <= set(DIMS):
                raise ValueError(f"each group must contain 'S' and be a subset of {DIMS}")
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}")
        if self.param_kernel not in PARAM_KERNELS:
            raise ValueError(f"unknown param_kernel {self.param_kernel!r}")
        if self.exercise_rule not in EXERCISE_RULES:
            raise ValueError(f"unknown exercise_rule {self.exercise_rule!r}")
        if self.pilot_widen < 1:
            raise ValueError("pilot_widen must be >= 1")
        widen = self.pilot_widen if self.exercise_rule == "pilot" else 1.0
        if not 0 < self.c_sigma * widen < 1:
            raise ValueError("c_sigma (x pilot_widen) must be in (0, 1) to keep sigma positive")


def _center(spec: PutSpec) -> dict:
    return {"S": spec.S0, "sigma": spec.sigma, "r": spec.r, "d": spec.d}


def dispersion_sizes(spec: PutSpec, cfg: MultiConfig, dims) -> dict:
    """alpha_k for each dispersed coordinate (placeholders, see MultiConfig)."""
    a = {"S": min(cfg.c_S * spec.S0 * spec.sigma * np.sqrt(spec.T), ALPHA_CAP * spec.S0),
         "sigma": cfg.c_sigma * spec.sigma, "r": cfg.alpha_r, "d": cfg.alpha_d}
    return {k: a[k] for k in DIMS if k in dims}


def isd_multi(spec: PutSpec, cfg: MultiConfig, dims, rng: np.random.Generator,
              N: int | None = None, widen: float = 1.0) -> dict:
    """Initial states z_n on scrambled Sobol points (product design): S by the
    Epanechnikov map of eq. (25), the parameters by cfg.param_kernel. ``widen``
    scales the parameter box (pilot paths)."""
    N = N or cfg.N
    alphas = dispersion_sizes(spec, cfg, dims)
    with warnings.catch_warnings():           # N need not be a power of 2 here
        warnings.simplefilter("ignore", UserWarning)
        U = qmc.Sobol(d=len(alphas), scramble=True, seed=rng).random(N)
    U = np.clip(U, 0.5 / N, 1 - 0.5 / N)
    center = _center(spec)
    z = {k: np.full(N, v) for k, v in center.items()}
    for col, (k, a) in enumerate(alphas.items()):
        if k == "S":
            z[k] = center[k] + a * isd_kernel(U[:, col], "epanechnikov")
        else:
            z[k] = center[k] + widen * a * isd_kernel(U[:, col], cfg.param_kernel)
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


def _params(spec, cfg, dims, z, scale):
    """Dispersed parameters of the group, centred and divided by scale[k]."""
    center = _center(spec)
    params = [k for k in PARAMS if k in dims]
    P = np.column_stack([(z[k] - center[k]) / scale[k] for k in params]) \
        if params else np.empty((len(z["S"]), 0))
    return params, P


# --------------------------------------------------------------------------
# One run: LSM and the t = 0 regression for one group of dispersed coordinates
# --------------------------------------------------------------------------
def lsm_multi(spec: PutSpec, cfg: MultiConfig, dims, z: dict, W: np.ndarray,
              rules: dict | None = None, value_function: bool = True):
    """LSM with per-path parameters. Returns (Y_naive, Y_vf, rules), Y_naive and
    Y_vf discounted to t = 0.

    Exercise rule at t_1..t_{J-1}: in-the-money regression on (S, parameters),
    fitted on these paths if ``rules`` is None, else the stored fits
    ``rules[j]`` are applied (out of sample). ``rules`` returned are the fits used.
    Y_vf = e^{-r_n dt} V(t_1) with V = max(Z, C_1) ("bermudan") or V = C_1
    ("european"), C_1 fitted on all paths; with cfg.control_variate, Y_vf is
    the premium label e^{-r_n dt} (V(t_1) - E_1) (see module doc). With
    ``value_function=False`` (pilot paths) Y_vf is None.
    """
    J = spec.J
    N = len(z["S"])
    params, P = _params(spec, cfg, dims, z, dispersion_sizes(spec, cfg, dims))
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
                Y_vf = disc * (np.maximum(ex - E1, prem) if cfg.style == "bermudan" else prem)
            else:
                cont = _fitted(Sj, mono_vf, index_vf, cash, cfg.M_tau)
                Y_vf = disc * (np.maximum(ex, cont) if cfg.style == "bermudan" else cont)
        cash = np.where(exercise, ex, cash)
    return disc * cash, Y_vf, fits


def greeks_multi(spec: PutSpec, cfg: MultiConfig, dims, z: dict, Y: np.ndarray,
                 add: dict | None = None) -> dict:
    """t = 0 regression of Y on the Taylor monomials of (z - z0); see module doc.
    Returns the Greeks this group identifies (others NaN), plus ``add``."""
    h = {k: float(np.max(np.abs(z[k] - v))) for k, v in _center(spec).items() if k in dims}
    params, P = _params(spec, cfg, dims, z, h)
    terms = basis_terms(cfg.M0, len(params), cfg.weight_vf, cfg.param_degree)
    mono, mindex = monomials(P, terms)
    A = design(np.vander((z["S"] - spec.S0) / h["S"], cfg.M0 + 1, increasing=True),
               mono, mindex)
    c = np.linalg.lstsq(A, Y, rcond=None)[0]
    index = {t: n for n, t in enumerate(terms)}
    out = {}
    for name, (i, *a_full) in GREEKS.items():
        a = tuple(ak for k, ak in zip(PARAMS, a_full) if k in params)
        if any(ak for k, ak in zip(PARAMS, a_full) if k not in params) or (i, a) not in index:
            out[name] = np.nan
            continue
        scale = factorial(i) * h["S"] ** -i
        for k, ak in zip(params, a):
            scale *= factorial(ak) * h[k] ** -ak
        out[name] = float(c[index[(i, a)]] * scale) + (add or {}).get(name, 0.0)
    return out


def exercise_rules(spec: PutSpec, cfg: MultiConfig, dims, rng) -> dict:
    """Exercise fits estimated on independent pilot paths (cfg.N_pilot paths,
    parameter box widened by cfg.pilot_widen)."""
    Np = cfg.N_pilot or cfg.N
    zp = isd_multi(spec, cfg, dims, rng, Np, cfg.pilot_widen)
    Wp = brownian(spec, Np, rng)
    return lsm_multi(spec, cfg, dims, zp, Wp, value_function=False)[2]


def run_group(spec: PutSpec, cfg: MultiConfig, dims, seed) -> dict:
    """One run with the coordinates ``dims`` dispersed."""
    rng = np.random.default_rng(seed)
    rules = None
    if cfg.style == "bermudan" and cfg.exercise_rule == "pilot":
        rules = exercise_rules(spec, cfg, dims, rng)
    z = isd_multi(spec, cfg, dims, rng)
    W = brownian(spec, cfg.N, rng)
    _, Y, _ = lsm_multi(spec, cfg, dims, z, W, rules)
    return greeks_multi(spec, cfg, dims, z, Y, bs_put(spec) if cfg.control_variate else None)


def label(spec: PutSpec, cfg: MultiConfig, seed) -> dict:
    """One estimate of the price and the Greeks at z0: one independent run per
    group; S-Greeks averaged over the runs; Theta from the PDE identity. Greeks
    no group identifies (Vera with the default groups) are left out."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    runs = [run_group(spec, cfg, dims, child)
            for dims, child in zip(cfg.groups, ss.spawn(len(cfg.groups)))]
    out = {}
    for name in GREEKS:                # only the Greeks some group identifies
        vals = [r[name] for r in runs if not np.isnan(r[name])]
        if vals:
            out[name] = float(np.mean(vals))
    out["theta"], out["ex_region"] = theta_pde(spec, out["price"], out["delta"], out["gamma"])
    if cfg.american_t0 and out["ex_region"]:
        # exercising at t_0 is optimal: V = K - S, so only Delta is non-zero
        out.update({k: 0.0 for k in OUTPUTS if k in out})
        out["price"], out["delta"] = spec.K - spec.S0, -1.0
    alphas = {}
    for dims in cfg.groups:
        alphas.update(dispersion_sizes(spec, cfg, dims))
    out.update({f"alpha_{k}": float(v) for k, v in alphas.items()})
    return out
