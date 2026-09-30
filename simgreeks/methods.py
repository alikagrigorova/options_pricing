"""The four estimators of Section 3 and one replication of the 2-step algorithm.

NAIVE      : regress discounted pathwise payoffs Y on the ISD X          (3.1)
NAIVE-VF   : regress the discounted t_1 value function on X              (3.2)
TRUNC-VF   : as NAIVE-VF, but only paths with |X - x0| <= alpha*         (3.3)
2STEP-VF   : rescale all paths so the ISD has size alpha*, re-run LSM,
             regress the value-function payoffs on the new X             (3.4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandwidth import optimal_alpha
from .core import (PutSpec, greeks_regression, isd_density_at_center,
                   isd_sample, lsm, simulate_growth)

METHODS = ("NAIVE", "NAIVE-VF", "TRUNC-VF", "2STEP-VF")


@dataclass(frozen=True)
class AlgoConfig:
    N: int = 100_000
    alpha: float = 10.0            # initial ISD size
    M_tau: int = 9                 # polynomial order for the exercise strategy
    M0: int = 9                    # polynomial order for the t = 0 regression
    isd_kernel: str = "epanechnikov"
    isd_deterministic: bool = True
    nu_opt: int = 2                # derivative alpha* is optimised for
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
                # rescale: same shocks G, new initial values within alpha*
                X2 = isd_sample(cfg.N, x0, a_star, cfg.isd_kernel,
                                cfg.isd_deterministic, rng)
                res2 = lsm(spec, X2[:, None] * G, cfg.M_tau)
                est = greeks_regression(X2, res2.Y_vf, x0, M0)
                rows.append(_row("2STEP-VF", M0, nu, est, a_star))
    return rows


def _row(method, M0, nu, est, a_star):
    return dict(method=method, M0=M0, nu=nu, price=est[0], delta=est[1],
                gamma=est[2], alpha_star=a_star)
