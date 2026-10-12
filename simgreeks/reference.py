"""Reference pricer for validating the simulated Greeks (not part of the paper).

Crank-Nicolson finite differences for the Black-Scholes-Merton PDE of a put
(or a call, spec.kind = "call") in x = ln S and time to maturity tau,

    V_tau = 1/2 sigma^2 V_xx + (r - d - 1/2 sigma^2) V_x - r V,

on a uniform x-grid with S0 on a node. Rannacher smoothing (the first two CN
steps replaced by four implicit Euler half-steps) follows the payoff and every
exercise date. Boundaries: V = 0 far out of the money; far in the money the
option is exercised at the next exercise date, V = sign (S e^{-d s} - K e^{-r s})
with s the time to that date (s = time to maturity where early exercise is never
optimal, spec.no_early_exercise).

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
    steps_per_dt: int = 40        # time steps per exercise interval dt (10 for daily dates)
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


def grid_for(spec: PutSpec) -> FDGrid:
    """Default grid: 40 time steps per exercise interval at 50 dates a year, 10
    at daily dates (time steps of 1/2000 and 1/2520 year)."""
    return FDGrid(steps_per_dt=10) if spec.steps_per_year >= 200 else FDGrid()


def _half_width(spec: PutSpec, grid: FDGrid) -> float:
    return max(grid.width_sd * spec.sigma * np.sqrt(spec.T), grid.min_width,
               1.5 * abs(np.log(spec.K / spec.S0)))


def put_fd(spec: PutSpec, style: str = "bermudan", grid: FDGrid | None = None) -> dict:
    """Price, Delta, Gamma and Theta at (t_0, S0) of a put or a call (spec.kind).
    See the module docstring.

    For "bermudan" these are the continuation value at t_0 and its derivatives;
    ``ex_region`` = (exercise value at S0 >= price) says whether exercising at
    t_0 would be optimal if it were allowed.
    """
    grid = grid or grid_for(spec)
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
    payoff = spec.payoff(S)
    call = spec.kind == "call"
    never = spec.no_early_exercise or style == "european"

    def itm(s, tau):              # deep in the money: exercised s from now (at maturity if never)
        s = tau if never else s
        edge = S[-1] if call else S[0]
        return spec.sign * (edge * np.exp(-spec.d * s) - spec.K * np.exp(-spec.r * s))

    def bounds(s, tau):           # (V at S_min, V at S_max)
        return (0.0, itm(s, tau)) if call else (itm(s, tau), 0.0)

    V = payoff.copy()
    since_kink = 0.0              # time since payoff / last exercise (in tau)
    tau = 0.0
    steps_since_kink = 0
    rec = {}
    for done in range(1, J * m + 2):   # J intervals, plus one step past t_0
        if exercise_every_step:
            V = np.maximum(ie(V, *bounds(0.0, tau + k)), payoff)
        elif steps_since_kink < 2:     # Rannacher: two IE half-steps per CN step
            V = ie_half(V, *bounds(since_kink + k / 2, tau + k / 2))
            V = ie_half(V, *bounds(since_kink + k, tau + k))
        else:
            V = cn(V, *bounds(since_kink + k, tau + k))
        since_kink += k
        tau += k
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
                ex_region=bool(spec.intrinsic >= price))


BUMPS = {"sigma": 0.005, "r": 0.002, "d": 0.002}


def put_fd_greeks(spec: PutSpec, style: str = "bermudan", grid: FDGrid | None = None,
                  bumps: dict = BUMPS, one_sided: tuple = (), vera: bool = True) -> dict:
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
    vera=False skips Vera (8 solves).
    """
    grid = grid or grid_for(spec)
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
    if vera and "sigma" in bumps and "r" in bumps:
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


def bs_call(spec: PutSpec) -> dict:
    """Closed-form Black-Scholes-Merton Greeks of the European call (Theta = dV/dt)."""
    S, K, T, s, r, d = spec.S0, spec.K, spec.T, spec.sigma, spec.r, spec.d
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - d + 0.5 * s * s) * T) / (s * sq)
    d2 = d1 - s * sq
    ed, er, pdf = np.exp(-d * T), np.exp(-r * T), norm.pdf(d1)
    vega = S * ed * pdf * sq
    return dict(price=S * ed * norm.cdf(d1) - K * er * norm.cdf(d2),
                delta=ed * norm.cdf(d1), gamma=ed * pdf / (S * s * sq),
                theta=(-ed * S * pdf * s / (2 * sq) - r * K * er * norm.cdf(d2)
                       + d * S * ed * norm.cdf(d1)),
                vega=vega, volga=vega * d1 * d2 / s,
                rho=K * T * er * norm.cdf(d2), phi=-S * T * ed * norm.cdf(d1),
                vanna=-ed * pdf * d2 / s, vera=-vega * d1 * sq / s, delta_r=ed * pdf * sq / s,
                delta_d=-T * ed * norm.cdf(d1) - ed * pdf * sq / s)


def bs_option(spec: PutSpec) -> dict:
    """bs_put or bs_call by spec.kind."""
    return bs_call(spec) if spec.kind == "call" else bs_put(spec)


LABEL_GREEKS = ("price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi",
                "delta_r", "delta_d", "vera")
FORWARD_GREEKS = ("rho", "phi", "delta_r", "delta_d")


def label_reference(spec: PutSpec, grid: FDGrid | None = None, style: str = "bermudan",
                    vera: bool = True) -> dict:
    """Reference values for a label of simgreeks.multi: style "bermudan" (the
    label's exercise dates, spec.J) or "american" (exercise at every time step of
    the PDE grid, Richardson-extrapolated in the time step: continuous exercise).

    * no early exercise optimal (put: r0 <= 0 <= d0; call: d0 <= 0 <= r0): the
      closed-form European values (the labels use them too,
      MultiConfig.european_region);
    * otherwise put_fd_greeks with central bumps; the bump of the parameter whose
      0 is the edge of the no-exercise region (r for a put, d for a call) is at
      most half its value, so it stays on one side; plus forward bumps for Rho,
      Phi and their Delta cross terms as <greek>_fwd;
    * exercise region (immediate exercise optimal): price = exercise value,
      Delta = sign (-1 put, +1 call), the other Greeks 0, as in the labels
      (MultiConfig.american_t0).
    Returns the LABEL_GREEKS (vera=False: without Vera), the <greek>_fwd values
    and ex_region.
    """
    greeks = [q for q in LABEL_GREEKS if vera or q != "vera"]
    if spec.no_early_exercise:
        eu = bs_option(spec)
        out = {q: float(eu[q]) for q in greeks}
        out.update({f"{q}_fwd": out[q] for q in FORWARD_GREEKS})
        out["ex_region"] = False
        return out
    bumps = dict(BUMPS)
    edge = "d" if spec.kind == "call" else "r"
    if getattr(spec, edge) > 0:
        bumps[edge] = min(BUMPS[edge], getattr(spec, edge) / 2)

    def solve(g):
        c = put_fd_greeks(spec, style, g, bumps, vera=vera)
        f = put_fd_greeks(spec, style, g, {"r": bumps["r"], "d": bumps["d"]},
                          one_sided=("r", "d"))
        res = {q: float(c[q]) for q in greeks}
        res.update({f"{q}_fwd": float(f[q]) for q in FORWARD_GREEKS})
        return res

    if style == "american" and grid is None:
        # implicit Euler with projection is first order in time: Richardson over
        # 20 and 40 steps per exercise interval (agrees with 80/160 to ~1e-6 at
        # daily dates)
        base = grid_for(spec)
        lo, hi = solve(replace(base, steps_per_dt=20)), solve(replace(base, steps_per_dt=40))
        out = {q: 2 * hi[q] - lo[q] for q in hi}
    else:
        out = solve(grid)
    out["ex_region"] = bool(spec.intrinsic >= out["price"])
    if out["ex_region"]:
        for q in list(out):
            if q != "ex_region":
                out[q] = 0.0
        out["price"], out["delta"] = spec.intrinsic, spec.sign
    return out
