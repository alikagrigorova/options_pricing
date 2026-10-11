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

Exercise rule. At t_j (j = J-1..1) the continuation value of the in-the-money
paths is a regression on Chebyshev(S) x parameter monomials, and a path is
exercised when Z(t_j) >= C_j. The European control variate is not used here:
regressing the premium cash flow Y_j - D_j (D_j the discounted European payoff)
was tried and removed. On the in-the-money paths, most of which are exercised
soon, Y_j and D_j are weakly correlated, so Y_j - D_j has 1.3-3x the residual
variance of Y_j and the rule lost twice as much value (price -0.23 % instead of
-0.12 %, results/diagnostics/price_bias/rule_cv).

Taylor regression at t = 0. Y is regressed on monomials of x_k = (z_k - z0_k)/h_k.
The coefficient c of x_S^i x_sigma^a1 x_r^a2 x_d^a3 gives
    d^(i+|a|) P / dS^i dsigma^a1 dr^a2 dd^a3 = i! a1! a2! a3! c / (h_S^i h_sigma^a1 h_r^a2 h_d^a3),
with the terms i + weight |a| <= M and |a| <= param_degree.

Each run has two phases on independent paths:

  1. Pilot (N_pilot paths). The coordinates in MultiConfig.select (default S
     and sigma; r and d optional) are dispersed widely: alpha0_S = c0_S S0 (the
     paper's alpha = 10 at S0 = 40), alpha0_sigma = c0_sigma sigma, alpha0_r,
     alpha0_d; the others by pilot_widen times their fixed width. The pilot
       * fits the exercise rule (exercise_rule="pilot"), and
       * chooses each selected width with the paper's selector (Appendix A.2,
         local order M = nu + 1, see methods.SELECTOR_ORDERS) applied to the
         pilot's partial residuals for that coordinate, i.e. the label minus
         the Taylor terms of the other coordinates. The target derivative is
         the Gamma for S (nu = 2), the Volga for sigma (nu = 2) and, when
         selected, Rho and Phi for r and d (nu = nu_rd = 1).
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

import ast
import itertools
import warnings
from dataclasses import dataclass, fields
from math import factorial

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import ndtr
from scipy.stats import qmc

from .core import PutSpec, isd_kernel, theta_pde
from .reference import bs_put
from .selector import ALPHA_CAP, S_REF, select_alpha

DIMS = ("S", "sigma", "r", "d")
PARAMS = ("sigma", "r", "d")
STYLES = ("bermudan", "european")
EXERCISE_RULES = ("pilot", "insample")
VF_RULES = ("max", "rule")
PARAM_KERNELS = ("uniform", "epanechnikov")
WIDTHS = ("selector", "fixed")
NU = {"S": 2, "sigma": 2}          # derivative each selected width is optimised for
#                                    (r, d: MultiConfig.nu_rd)
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
    "phi": (0, 0, 0, 1),         # dP / dd (dividend rho)
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
    alpha0_r: float = 0.05         # pilot alpha0_r (when "r" in select)
    alpha0_d: float = 0.05         # pilot alpha0_d (when "d" in select)
    nu_rd: int = 1                 # derivative the r and d widths are optimised for (Rho, Phi)
    rd_unit: str = "pilot"         # selector units for r, d: "pilot" (alpha0 -> S_REF / 4) or "raw"
    c_S: float = 0.6               # fixed alpha_S = c_S S0 sigma sqrt(T)   (placeholder)
    c_sigma: float = 0.25          # fixed alpha_sigma = c_sigma sigma      (placeholder)
    alpha_r: float = 0.03          # fixed alpha_r (width sweep: 0.03-0.04 best when labels are averaged)
    alpha_d: float = 0.03          # fixed alpha_d (same)
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
    rd_floor: bool = False         # one-sided r, d spread [max(0, x0 - alpha), x0 + alpha] when x0 >= 0
    kink_r: bool = False           # truncated-power basis in r with a knot at r = 0 (see kink_index)
    alpha_r_shrink: bool = True    # fixed alpha_r = min(alpha_r, r0) for r0 > 0, so the window stays in r >= 0
    alpha_d_shrink: str = "r0"     # fixed alpha_d = min(alpha_d, r0) ("r0") or min(alpha_d, d0) ("d0") when > 0; "" off
    european_region: bool = True   # r0 <= 0 and d0 >= 0: exact European label (see label)

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
        if not set(self.select) <= set(DIMS):
            raise ValueError(f"select must be a subset of {DIMS}")
        if self.alpha_d_shrink not in ("", "r0", "d0"):
            raise ValueError(f"unknown alpha_d_shrink {self.alpha_d_shrink!r}")
        if self.rd_unit not in ("pilot", "raw"):
            raise ValueError(f"unknown rd_unit {self.rd_unit!r}")
        if self.pilot_widen < 1:
            raise ValueError("pilot_widen must be >= 1")
        if not 0 < self.c0_sigma < 1 or not 0 < self.c_sigma * self.pilot_widen < 1:
            raise ValueError("c0_sigma and c_sigma x pilot_widen must be in (0, 1) "
                             "to keep sigma positive")

    @property
    def uses_pilot(self) -> bool:
        return self.widths == "selector" or (self.style == "bermudan"
                                             and self.exercise_rule == "pilot")


def config_from(settings=()) -> MultiConfig:
    """MultiConfig with fields set from "name=value" strings, the values read as
    Python literals (else as strings), e.g. ("N_pilot=200_000",
    "control_variate=False"). For command-line scripts."""
    names = {f.name for f in fields(MultiConfig)}
    kw = {}
    for item in settings:
        name, sep, value = item.partition("=")
        if not sep or name.strip() not in names:
            raise ValueError(f"expected name=value with name one of {sorted(names)}, got {item!r}")
        try:
            kw[name.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            kw[name.strip()] = value.strip()
    return MultiConfig(**kw)


def _center(spec: PutSpec) -> dict:
    return {"S": spec.S0, "sigma": spec.sigma, "r": spec.r, "d": spec.d}


# --------------------------------------------------------------------------
# Widths of the dispersion
# --------------------------------------------------------------------------
def fixed_widths(spec: PutSpec, cfg: MultiConfig, dims) -> dict:
    """Placeholder widths alpha_k (see MultiConfig)."""
    a_r = min(cfg.alpha_r, spec.r) if cfg.alpha_r_shrink and spec.r > 0 else cfg.alpha_r
    x = {"r0": spec.r, "d0": spec.d}.get(cfg.alpha_d_shrink, 0.0)
    a_d = min(cfg.alpha_d, x) if x > 0 else cfg.alpha_d
    a = {"S": min(cfg.c_S * spec.S0 * spec.sigma * np.sqrt(spec.T), ALPHA_CAP * spec.S0),
         "sigma": cfg.c_sigma * spec.sigma, "r": a_r, "d": a_d}
    return {k: a[k] for k in DIMS if k in dims}


def _selected(cfg: MultiConfig, dims) -> list:
    return [k for k in DIMS if k in dims and k in cfg.select] if cfg.widths == "selector" else []


def pilot_widths(spec: PutSpec, cfg: MultiConfig, dims) -> dict:
    """Pilot widths: alpha0 for the selected coordinates, the fixed widths
    otherwise (parameters widened by pilot_widen; S diffuses and is not)."""
    a0 = {"S": cfg.c0_S * spec.S0, "sigma": cfg.c0_sigma * spec.sigma,
          "r": cfg.alpha0_r, "d": cfg.alpha0_d}
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
        nu = NU.get(k, cfg.nu_rd)
        sel = select_alpha(z[k], partial_residual(fit, Y, k), center[k], alphas0[k], kernel,
                           M=nu + 1, nu=nu, x_unit=_x_unit(spec, cfg, k, alphas0[k]))
        # the selector's own cap 0.75 x0 keeps S and sigma positive; r and d may
        # be 0 or negative, so they take the uncapped value and only the cap below
        a = sel["alpha_raw"] if k in ("r", "d") else sel["alpha_star"]
        cap = min(alphas0[k], ALPHA_CAP * spec.S0) if k == "S" else alphas0[k] / cfg.pilot_widen
        out[k] = float(min(a, cap))
    return out


def _x_unit(spec: PutSpec, cfg: MultiConfig, k: str, alpha0: float) -> float:
    """Units the selector runs in for coordinate k (selector module doc: the
    printed (A.3) is not unit-free, so h_ROT depends on them). S: the stock
    starts at S_REF. sigma: as is. r, d (rd_unit="pilot"): the pilot width maps
    to the stock's pilot width at S_REF, c0_S S_REF (= 10, the paper's alpha);
    rd_unit="raw": as is."""
    if k == "S":
        return spec.S0 / S_REF
    if k in ("r", "d") and cfg.rd_unit == "pilot":
        return alpha0 / (cfg.c0_S * S_REF)
    return 1.0


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
        u = isd_kernel(U[:, col], kernel)                 # in [-1, 1]
        lo, hi = spread_bounds(cfg, k, center[k], a)
        z[k] = center[k] + a * u if lo == center[k] - a else lo + (hi - lo) * (u + 1) / 2
    return z


def spread_bounds(cfg: MultiConfig, k: str, x0: float, a: float) -> tuple:
    """Interval [lo, hi] the coordinate k is dispersed over: [x0 - a, x0 + a],
    or with cfg.rd_floor the one-sided [max(0, x0 - a), x0 + a] for r and d when
    x0 >= 0, so that no path has a negative rate or dividend yield. The Taylor
    polynomial stays centred at x0."""
    if cfg.rd_floor and k in ("r", "d") and x0 >= 0:
        return max(0.0, x0 - a), x0 + a
    return x0 - a, x0 + a


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
def basis_terms(M: int, n_params: int, weight: int, param_degree: int, kink: int | None = None):
    """Multi-indices (i, a) with i + weight |a| <= M and |a| <= param_degree,
    |a| = sum_k |a_k|.

    kink: index of a parameter (r) that takes a truncated-power basis with a
    knot at 0 instead of monomials: exponents 0 and 1 of (x - x0) and -1, -2,
    -3 for (x)_+, (x)_+^2, (x)_+^3 (degree |a_k|); see monomials and kink_index."""
    def exps(k, deg):
        if k == kink:
            return [e for e in (0, 1, -1, -2, -3) if abs(e) <= deg]
        return range(deg + 1)
    terms = []
    for deg in range(min(param_degree, M // weight) + 1):
        for a in itertools.product(*[exps(k, deg) for k in range(n_params)]):
            if sum(abs(e) for e in a) == deg:
                terms += [(i, a) for i in range(M - weight * deg + 1)]
    return terms


def monomials(P: np.ndarray, terms, Pk: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """Parameter monomials prod_k P[:, k]^a_k for the distinct a in terms, as
    columns (Fortran order), and the (i, column) of each term. A negative a_k
    stands for Pk[:, k]^|a_k| (truncated powers, basis_terms' kink)."""
    exps = list(dict.fromkeys(a for _, a in terms))
    mono = np.empty((P.shape[0], len(exps)), order="F")
    for m, a in enumerate(exps):
        col = np.ones(P.shape[0])
        for k, e in enumerate(a):
            if e > 0:
                col = col * P[:, k] ** e
            elif e < 0:
                col = col * Pk[:, k] ** -e
        mono[:, m] = col
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
    divided by scale[k], in PARAMS order; and Pk, the truncated powers' base
    max(z_k, 0) / scale[k] (used by a kink basis)."""
    center = _center(spec)
    params = [k for k in PARAMS if k in scale]
    if not params:
        return params, np.empty((len(z["S"]), 0)), np.empty((len(z["S"]), 0))
    P = np.column_stack([(z[k] - center[k]) / scale[k] for k in params])
    Pk = np.column_stack([np.maximum(z[k], 0.0) / scale[k] for k in params])
    return params, P, Pk


def kink_index(spec: PutSpec, cfg: MultiConfig, params) -> int | None:
    """Index of r in params when the r basis needs a knot at r = 0, else None.

    With cfg.kink_r and 0 < r0 < alpha_r, the r window crosses 0, where the
    early-exercise premium has a kink (it is 0 for r <= 0, d >= 0). The decision
    depends only on (r0, alpha_r), so the pilot's exercise rule and the main
    paths use the same basis."""
    if not cfg.kink_r or "r" not in params:
        return None
    a_r = fixed_widths(spec, cfg, ("r",))["r"]
    return params.index("r") if 0 < spec.r < a_r else None


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
    params, P, Pk = _params(spec, z, scale)
    kink = kink_index(spec, cfg, params)
    mono, index = monomials(P, basis_terms(cfg.M_tau, len(params), cfg.weight,
                                           cfg.param_degree, kink), Pk)
    mono_vf, index_vf = monomials(P, basis_terms(cfg.M_tau, len(params), cfg.weight_vf,
                                                 cfg.param_degree, kink), Pk)
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
    kink: int | None = None      # index in params of a truncated-power (kink) basis
    center: dict | None = None   # z0


def taylor_fit(spec: PutSpec, cfg: MultiConfig, z: dict, Y: np.ndarray,
               dims=None) -> TaylorFit:
    """Regress Y on the Taylor monomials of the coordinates ``dims`` (default:
    those that vary in z)."""
    center = _center(spec)
    if dims is None:
        dims = [k for k in DIMS if np.ptp(z[k]) > 0]
    h = {k: float(np.max(np.abs(z[k] - center[k]))) for k in DIMS if k in dims}
    params, P, Pk = _params(spec, z, h)
    kink = kink_index(spec, cfg, params)
    terms = basis_terms(cfg.M0, len(params), cfg.weight_vf, cfg.param_degree, kink)
    mono, mindex = monomials(P, terms, Pk)
    A = design(np.vander((z["S"] - spec.S0) / h["S"], cfg.M0 + 1, increasing=True),
               mono, mindex)
    return TaylorFit(params, terms, np.linalg.lstsq(A, Y, rcond=None)[0], h, A, kink, center)


def _feature_derivative(e: int, m: int, x0: float, h: float) -> float:
    """m-th derivative at x0 of a parameter basis function: ((x - x0)/h)^e for
    e >= 0, (max(x, 0)/h)^|e| for e < 0 (x0 > 0: the right branch of the knot)."""
    if e >= 0:
        return factorial(e) / h ** e if m == e else 0.0
    k = -e
    return factorial(k) / factorial(k - m) * x0 ** (k - m) / h ** k if m <= k else 0.0


def greeks_from(fit: TaylorFit, add: dict | None = None) -> dict:
    """The Greeks the fit identifies (others NaN), plus ``add``."""
    index = {t: n for n, t in enumerate(fit.terms)}
    out = {}
    for name, (i, *a_full) in GREEKS.items():
        a = tuple(ak for k, ak in zip(PARAMS, a_full) if k in fit.params)
        if any(ak for k, ak in zip(PARAMS, a_full) if k not in fit.params):
            out[name] = np.nan
            continue
        if fit.kink is not None:
            # derivative of the fitted curve at z0: sum over the kink parameter's
            # basis functions; the other parameters' monomials as usual
            kp = fit.params[fit.kink]
            val, found = 0.0, False
            for n, (ti, ta) in enumerate(fit.terms):
                if ti != i or any(ta[k] != a[k] for k in range(len(a)) if k != fit.kink):
                    continue
                dv = _feature_derivative(ta[fit.kink], a[fit.kink], fit.center[kp], fit.h[kp])
                if dv:
                    sc = factorial(i) * fit.h["S"] ** -i
                    for k, (p, ak) in enumerate(zip(fit.params, a)):
                        if k != fit.kink:
                            sc *= factorial(ak) * fit.h[p] ** -ak
                    val += fit.coef[n] * sc * dv
                    found = True
            out[name] = float(val) + (add or {}).get(name, 0.0) if found else np.nan
            continue
        if (i, a) not in index:
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
        return any(others) or any(ak != 0 for p, ak in zip(fit.params, a) if p != k)
    cols = [n for n, (i, a) in enumerate(fit.terms) if involves_other(i, a)]
    return Y - fit.A[:, cols] @ fit.coef[cols]


def greeks_multi(spec: PutSpec, cfg: MultiConfig, dims, z: dict, Y: np.ndarray,
                 add: dict | None = None) -> dict:
    """t = 0 regression over the coordinates ``dims`` and its Greeks (others NaN)."""
    return greeks_from(taylor_fit(spec, cfg, z, Y, dims), add)


# --------------------------------------------------------------------------
# One run, one label
# --------------------------------------------------------------------------
def pilot(spec: PutSpec, cfg: MultiConfig, dims, rng: np.random.Generator):
    """Pilot phase of a run (module doc). Returns (alphas, scale, rules): the
    widths for the main paths, the scale the exercise rule was fitted with and
    the rule (None: fitted on the main paths)."""
    alphas = fixed_widths(spec, cfg, dims)
    if not cfg.uses_pilot:
        return alphas, alphas, None
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
        return alphas, scale, fits
    return alphas, alphas, None


def run_group(spec: PutSpec, cfg: MultiConfig, dims, seed) -> dict:
    """One run with the coordinates ``dims`` dispersed: pilot, then main (module
    doc). Returns the Greeks this group identifies and the widths it used."""
    rng = np.random.default_rng(seed)
    alphas, scale, rules = pilot(spec, cfg, dims, rng)
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
    if cfg.european_region and cfg.style == "bermudan" and spec.r <= 0 and spec.d >= 0:
        return european_label(spec, cfg)
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


def european_label(spec: PutSpec, cfg: MultiConfig) -> dict:
    """Exact label when r0 <= 0 and d0 >= 0 (MultiConfig.european_region).

    There, exercising a put early is never optimal: the strike earns no interest
    (r <= 0) and selling the stock gives up no positive carry (d >= 0), so the
    Bermudan put equals the European put and so do its price, Delta, Gamma,
    Theta, Vega, Volga and Vanna. Rho and Phi (and delta_r, delta_d) are the
    European ones from the side of the no-exercise region; at r0 = 0 or d0 = 0
    the early-exercise premium on the other side grows faster than linearly
    (only deep in-the-money paths are exercised), so the one-sided derivative
    from that side converges to the same value, but very slowly.
    """
    eu = bs_put(spec)
    out = {k: float(eu[k]) for k in OUTPUTS if k in eu and k != "vera"}
    out["ex_region"] = False
    for dims in cfg.groups:
        for k, v in fixed_widths(spec, cfg, dims).items():
            out[f"alpha_{k}"] = 0.0
    return out
