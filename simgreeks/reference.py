"""Reference pricer for validating the simulated Greeks (not part of the paper).

Crank-Nicolson finite differences for the Black-Scholes-Merton PDE of a put in
x = ln S and time to maturity tau,

    V_tau = 1/2 sigma^2 V_xx + (r - d - 1/2 sigma^2) V_x - r V,

on a uniform x-grid with S0 on a node. Rannacher smoothing (the first two CN
steps replaced by four implicit Euler half-steps) follows the payoff and every
exercise date. Boundaries: V = 0 at S_max; at S_min the put is exercised at the
next exercise date, V = K e^{-r s} - S e^{-d s} with s the time to that date.

style:
  "bermudan" : exercise at t_j = j dt, j = 1..J (dt = T / J, J = spec.J), as in
               the LSM of simgreeks.core: no exercise at t_0. This is the
               setting of the paper's binomial benchmark values.
  "american" : exercise at every time step, t_0 included (implicit Euler with
               projection; first order in time).
  "european" : no early exercise.

Theta is dV/dt at t_0 in calendar time, per year, by a central difference over
(-k, +k), k the time step. The value is smooth in t there because t_0 is not an
exercise date ("bermudan", "european") or is exercised at every step
("american"). Delta and Gamma use central differences on the x-grid.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm

from .core import PutSpec

STYLES = ("bermudan", "american", "european")


@dataclass(frozen=True)
class FDGrid:
    n_x: int = 2001               # x-grid nodes (odd, S0 at the centre)
    width_sd: float = 8.0         # half-width of the grid in units of sigma sqrt(T)
    min_width: float = 0.5        # minimum half-width in x
    steps_per_dt: int = 40        # time steps per exercise interval dt
    half_width: float | None = None   # fixed half-width in x (else from the three above)


def _operator(spec: PutSpec, h: float):
    """Coefficients (l, c, u) of the discretised generator on the x-grid."""
    a = 0.5 * spec.sigma ** 2
    b = spec.r - spec.d - a
    return a / h ** 2 - b / (2 * h), -2 * a / h ** 2 - spec.r, a / h ** 2 + b / (2 * h)


class _Stepper:
    """theta-scheme steps (I - th k L) V_new = (I + (1 - th) k L) V_old with
    Dirichlet boundary values."""

    def __init__(self, coefs, n, k, th):
        l, c, u = coefs
        self.l, self.c, self.u, self.k, self.th = l, c, u, k, th
        ab = np.empty((3, n - 2))
        ab[0, :] = -th * k * u
        ab[1, :] = 1 - th * k * c
        ab[2, :] = -th * k * l
        self.ab = ab

    def __call__(self, V, lo_new, hi_new):
        l, c, u, k, th = self.l, self.c, self.u, self.k, self.th
        e = (1 - th) * k
        rhs = V[1:-1] + e * (l * V[:-2] + c * V[1:-1] + u * V[2:])
        rhs[0] += th * k * l * lo_new
        rhs[-1] += th * k * u * hi_new
        out = np.empty_like(V)
        out[1:-1] = solve_banded((1, 1), self.ab, rhs, overwrite_b=True, check_finite=False)
        out[0], out[-1] = lo_new, hi_new
        return out


def _half_width(spec: PutSpec, grid: FDGrid) -> float:
    return max(grid.width_sd * spec.sigma * np.sqrt(spec.T), grid.min_width,
               1.5 * abs(np.log(spec.K / spec.S0)))


def put_fd(spec: PutSpec, style: str = "bermudan", grid: FDGrid = FDGrid()) -> dict:
    """Price, Delta, Gamma and Theta at (t_0, S0). See the module docstring.

    For "bermudan" these are the continuation value at t_0 and its derivatives;
    ``ex_region`` = (K - S0 >= price) says whether exercising at t_0 would be
    optimal if it were allowed.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    n = grid.n_x | 1
    ic = n // 2
    L = grid.half_width or _half_width(spec, grid)
    h = 2 * L / (n - 1)
    S = spec.S0 * np.exp((np.arange(n) - ic) * h)
    J, dt = spec.J, spec.dt
    m = grid.steps_per_dt
    k = dt / m
    coefs = _operator(spec, h)
    exercise_every_step = style == "american"
    cn = _Stepper(coefs, n, k, 0.5)
    ie_half = _Stepper(coefs, n, k / 2, 1.0)
    ie = _Stepper(coefs, n, k, 1.0)
    payoff = np.maximum(spec.K - S, 0.0)

    def lower(s):                 # deep in the money: exercised s from now
        return spec.K * np.exp(-spec.r * s) - S[0] * np.exp(-spec.d * s)

    V = payoff.copy()
    since_kink = 0.0              # time since payoff / last exercise (in tau)
    steps_since_kink = 0
    rec = {}
    for done in range(1, J * m + 2):   # J intervals, plus one step past t_0
        if exercise_every_step:
            V = np.maximum(ie(V, spec.K - S[0], 0.0), payoff)
        elif steps_since_kink < 2:     # Rannacher: two IE half-steps per CN step
            V = ie_half(V, lower(since_kink + k / 2), 0.0)
            V = ie_half(V, lower(since_kink + k), 0.0)
        else:
            V = cn(V, lower(since_kink + k), 0.0)
        since_kink += k
        steps_since_kink += 1
        if done - J * m in (-1, 0, 1):          # t = +k, 0, -k
            rec[done - J * m] = V[ic - 1:ic + 2].copy()
        if style == "bermudan" and done % m == 0 and done < J * m:
            V = np.maximum(V, payoff)           # exercise date t_{J - done/m}
            since_kink, steps_since_kink = 0.0, 0
    v_m, v_0, v_p = rec[0]
    Vx = (v_p - v_m) / (2 * h)
    Vxx = (v_p - 2 * v_0 + v_m) / h ** 2
    price = float(v_0)
    theta = float((rec[-1][1] - rec[1][1]) / (2 * k))   # V(t=+k) - V(t=-k)
    return dict(price=price, delta=float(Vx / spec.S0),
                gamma=float((Vxx - Vx) / spec.S0 ** 2), theta=theta,
                ex_region=bool(spec.K - spec.S0 >= price))


BUMPS = {"sigma": 0.005, "r": 0.002, "d": 0.002}


def put_fd_greeks(spec: PutSpec, style: str = "bermudan", grid: FDGrid = FDGrid(),
                  bumps: dict = BUMPS, one_sided: tuple = ()) -> dict:
    """put_fd at z0 plus Greeks in sigma, r and d from bumped solves on the same
    x-grid (so the discretisation error cancels), by central differences with
    steps h and h/2 combined by Richardson extrapolation, (4 D(h/2) - D(h)) / 3:

      vega, rho, phi         : (P(+h) - P(-h)) / 2h
      volga                  : (P(+h) - 2 P + P(-h)) / h^2   (sigma only)
      vanna, delta_r, delta_d: (Delta(+h) - Delta(-h)) / 2h
      vera                   : (P(++) - P(+-) - P(-+) + P(--)) / (4 h_sigma h_r)

    one_sided: parameters ("r", "d") differentiated forward instead,
    (P(+h) - P) / h and (Delta(+h) - Delta) / h, combined as 2 D(h/2) - D(h).
    At r = 0 (or d = 0) the price has a kink-like non-smoothness (no early
    exercise on one side), so the forward value depends strongly on h there.
    """
    grid = replace(grid, half_width=grid.half_width or _half_width(spec, grid))
    base = put_fd(spec, style, grid)
    out = dict(base)
    names = {"sigma": ("vega", "vanna"), "r": ("rho", "delta_r"), "d": ("phi", "delta_d")}

    def differences(k, h):
        up = put_fd(replace(spec, **{k: getattr(spec, k) + h}), style, grid)
        first, cross = names[k]
        if k in one_sided:
            return {first: (up["price"] - base["price"]) / h,
                    cross: (up["delta"] - base["delta"]) / h}
        dn = put_fd(replace(spec, **{k: getattr(spec, k) - h}), style, grid)
        D = {first: (up["price"] - dn["price"]) / (2 * h),
             cross: (up["delta"] - dn["delta"]) / (2 * h)}
        if k == "sigma":
            D["volga"] = (up["price"] - 2 * base["price"] + dn["price"]) / h ** 2
        return D

    def cross(hs, hr):
        P = {(a, b): put_fd(replace(spec, sigma=spec.sigma + a * hs, r=spec.r + b * hr),
                            style, grid)["price"] for a in (1, -1) for b in (1, -1)}
        return (P[1, 1] - P[1, -1] - P[-1, 1] + P[-1, -1]) / (4 * hs * hr)

    for k, h in bumps.items():
        coarse, fine = differences(k, h), differences(k, h / 2)
        w = 2 if k in one_sided else 4             # error O(h) forward, O(h^2) central
        out.update({q: (w * fine[q] - coarse[q]) / (w - 1) for q in coarse})
    if "sigma" in bumps and "r" in bumps:
        hs, hr = bumps["sigma"], bumps["r"]
        out["vera"] = (4 * cross(hs / 2, hr / 2) - cross(hs, hr)) / 3
    return out


def bs_put(spec: PutSpec) -> dict:
    """Closed-form Black-Scholes-Merton Greeks of the European put (Theta = dV/dt)."""
    S, K, T, s, r, d = spec.S0, spec.K, spec.T, spec.sigma, spec.r, spec.d
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - d + 0.5 * s * s) * T) / (s * sq)
    d2 = d1 - s * sq
    ed, er, pdf = np.exp(-d * T), np.exp(-r * T), norm.pdf(d1)
    vega = S * ed * pdf * sq
    return dict(price=K * er * norm.cdf(-d2) - S * ed * norm.cdf(-d1),
                delta=-ed * norm.cdf(-d1), gamma=ed * pdf / (S * s * sq),
                theta=(-ed * S * pdf * s / (2 * sq) + r * K * er * norm.cdf(-d2)
                       - d * S * ed * norm.cdf(-d1)),
                vega=vega, volga=vega * d1 * d2 / s,
                rho=-K * T * er * norm.cdf(-d2), phi=S * T * ed * norm.cdf(-d1),
                vanna=-ed * pdf * d2 / s, vera=-vega * d1 * sq / s, delta_r=ed * pdf * sq / s,
                delta_d=T * ed * norm.cdf(-d1) - ed * pdf * sq / s)
