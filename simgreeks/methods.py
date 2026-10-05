"""The four estimators of Section 3 and one replication of the 2-step algorithm.

NAIVE      : regress discounted pathwise payoffs Y on the ISD X          (3.1)
NAIVE-VF   : regress the discounted t_1 value function on X              (3.2)
TRUNC-VF   : as NAIVE-VF, but only paths with |X - x0| <= alpha*         (3.3)
2STEP-VF   : rescale all paths so the ISD has size alpha*, apply the pilot's
             estimated stopping rule to them, refit the t_1 regression on
             the rescaled paths and regress the smoothed payoffs on the
             new X                                                       (3.4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandwidth import optimal_alpha
from .core import (PutSpec, greeks_regression, isd_density_at_center,
                   isd_sample, lsm, simulate_growth)

METHODS = ("NAIVE", "NAIVE-VF", "TRUNC-VF", "2STEP-VF")

# How the 2-step method treats the rescaled paths:
#   refit_t1 : re-apply the pilot's stored in-the-money rules at t_{J-1}..t_2,
#              refit the all-path t_1 regression, V = max(Z, C_new)  (default)
#   reuse_t1 : re-apply all pilot fits, including the t_1 curve, V = max(Z, C_pilot)
#   rerun    : re-estimate the whole LSM on the rescaled paths, V by the ITM rule
SECOND_STEP_MODES = ("refit_t1", "reuse_t1", "rerun")


def second_step(spec: PutSpec, cfg: "AlgoConfig", X, G, pilot, alpha0, a_star,
                mode: str = "refit_t1"):
    """Rescale the pilot ISD to size a_star and return (X', smoothed Y')."""
    x0 = spec.S0
    X2 = x0 + (a_star / alpha0) * (X - x0)
    S2 = X2[:, None] * G                  # = S * X'/X: exact under GBM, same shocks
    if mode == "refit_t1":
        stored = {j: f for j, f in pilot.itm_fits.items() if j >= 2}
        res2 = lsm(spec, S2, cfg.M_tau, itm_fits=stored, vf_rule="max")
    elif mode == "reuse_t1":
        res2 = lsm(spec, S2, cfg.M_tau, itm_fits=pilot.itm_fits,
                   t1_fit_all=pilot.t1_fit_all, vf_rule="max")
    elif mode == "rerun":
        res2 = lsm(spec, S2, cfg.M_tau)
    else:
        raise ValueError(f"unknown second_step mode {mode!r}")
    return X2, res2.Y_vf


@dataclass(frozen=True)
class AlgoConfig:
    N: int = 100_000
    alpha: float = 10.0            # initial ISD size
    M_tau: int = 9                 # polynomial order for the exercise strategy
    M0: int = 9                    # polynomial order for the t = 0 regression
    isd_kernel: str = "epanechnikov"
    isd_deterministic: bool = True
    nu_opt: int = 2                # derivative alpha* is optimised for
    second_step: str = "refit_t1"  # see SECOND_STEP_MODES
    alpha_star_fixed: float | None = None  # bypass the selector (diagnostics)
    methods: tuple = field(default=METHODS)


def run_once(spec: PutSpec, cfg: AlgoConfig, seed, M0_list=None, nu_list=None):
    """One independent replication.

    Returns a list of dict rows with keys method, M0, nu, price, delta, gamma,
    alpha_star. ``M0_list`` / ``nu_list`` allow evaluating several t = 0
    polynomial orders / bandwidth targets on the *same* simulated paths (used
    for Tables 7 and 9).
    """
    rng = np.random.default_rng(seed)
    x0 = spec.S0
    M0_list = M0_list or [cfg.M0]
    nu_list = nu_list or [cfg.nu_opt]

    G = simulate_growth(spec, cfg.N, rng)
    X = isd_sample(cfg.N, x0, cfg.alpha, cfg.isd_kernel, cfg.isd_deterministic, rng)
    res = lsm(spec, X[:, None] * G, cfg.M_tau)
    f0 = isd_density_at_center(cfg.alpha, cfg.isd_kernel)

    rows = []
    for M0 in M0_list:
        if "NAIVE" in cfg.methods:
            rows.append(_row("NAIVE", M0, None, greeks_regression(X, res.Y_naive, x0, M0), np.nan))
        if "NAIVE-VF" in cfg.methods:
            rows.append(_row("NAIVE-VF", M0, None, greeks_regression(X, res.Y_vf, x0, M0), np.nan))
        for nu in nu_list:
            need_bw = ("TRUNC-VF" in cfg.methods) or ("2STEP-VF" in cfg.methods)
            if not need_bw:
                continue
            if cfg.alpha_star_fixed is not None:
                a_star = cfg.alpha_star_fixed
            else:
                a_star = optimal_alpha(X, res.Y_vf, x0, cfg.alpha, f0, M0=M0, nu=nu,
                                       alpha_max=0.75 * x0)
            if "TRUNC-VF" in cfg.methods:
                m = np.abs(X - x0) <= a_star
                if m.sum() < 5 * (M0 + 1):
                    est = (np.nan, np.nan, np.nan)
                else:
                    est = greeks_regression(X[m], res.Y_vf[m], x0, M0)
                rows.append(_row("TRUNC-VF", M0, nu, est, a_star))
            if "2STEP-VF" in cfg.methods:
                X2, Y2 = second_step(spec, cfg, X, G, res, cfg.alpha, a_star,
                                     cfg.second_step)
                est = greeks_regression(X2, Y2, x0, M0)
                rows.append(_row("2STEP-VF", M0, nu, est, a_star))
    return rows


def _row(method, M0, nu, est, a_star):
    return dict(method=method, M0=M0, nu=nu, price=est[0], delta=est[1],
                gamma=est[2], alpha_star=a_star)


def run_fixed_alpha(spec: PutSpec, cfg: AlgoConfig, seed, alpha_stars, modes):
    """Diagnostic: one pilot run with ISD size cfg.alpha, then the second step
    for each fixed alpha* and each second-step mode on the same paths."""
    rng = np.random.default_rng(seed)
    x0 = spec.S0
    G = simulate_growth(spec, cfg.N, rng)
    X = isd_sample(cfg.N, x0, cfg.alpha, cfg.isd_kernel, cfg.isd_deterministic, rng)
    pilot = lsm(spec, X[:, None] * G, cfg.M_tau)
    rows = []
    for a_star in alpha_stars:
        for mode in modes:
            X2, Y2 = second_step(spec, cfg, X, G, pilot, cfg.alpha, a_star, mode)
            est = greeks_regression(X2, Y2, x0, cfg.M0)
            r = _row("2STEP-VF", cfg.M0, None, est, a_star)
            r["mode"] = mode
            rows.append(r)
    return rows
