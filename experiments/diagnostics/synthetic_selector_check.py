"""Does the paper's alpha* selector recover a known curvature?

Synthetic data: X from the Epanechnikov ISD around x0 = 40 with size alpha0,
Y = V(X) + noise * eps, with V the Black-Scholes European put (K = 40,
vol = 20%, r = 6%, T = 1), whose Taylor coefficients at x0 are known exactly.
For each noise level and alpha0 the selector's stage-2 estimate of beta_10
is compared with the truth, and alpha* with the "oracle" alpha* from (A.1)
using the true beta_10 and noise variance.

    python experiments/diagnostics/synthetic_selector_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from simgreeks.core import isd_sample  # noqa: E402
from simgreeks.selector import constants, select_alpha  # noqa: E402

X0, K, VOL, R, T = 40.0, 40.0, 0.20, 0.06, 1.0
M, NU, N = 9, 2, 100_000
OUT = ROOT / "results" / "diagnostics" / "synthetic"


def bs_put_mp(s):
    d1 = (mp.log(s / K) + (R + VOL ** 2 / 2) * T) / (VOL * mp.sqrt(T))
    d2 = d1 - VOL * mp.sqrt(T)
    return K * mp.e ** (-R * T) * mp.ncdf(-d2) - s * mp.ncdf(-d1)


def bs_put(s):
    d1 = (np.log(s / K) + (R + VOL ** 2 / 2) * T) / (VOL * np.sqrt(T))
    d2 = d1 - VOL * np.sqrt(T)
    return K * np.exp(-R * T) * norm.cdf(-d2) - s * norm.cdf(-d1)


def main(seeds=5):
    mp.mp.dps = 50
    b10 = float(mp.diff(bs_put_mp, mp.mpf(X0), 10) / mp.factorial(10))
    k = constants(M, NU, "literal")
    lines = ["# Synthetic known-curve test of the paper's selector (literal reading)\n",
             f"True beta_10 = V^(10)(40)/10! = {b10:.4e}. Medians over {seeds} seeds "
             "(one for zero noise).\n",
             "| noise | alpha0 | beta_10 estimate | estimate / true | alpha* | oracle alpha* |",
             "|---|---|---|---|---|---|"]
    for sig in (0.0, 1e-9, 1e-6, 1e-3, 0.45):
        for a0 in (5, 10, 25):
            eb, ea = [], []
            for seed in range(1 if sig == 0 else seeds):
                rng = np.random.default_rng(seed)
                X = isd_sample(N, X0, a0, "epanechnikov")
                Y = bs_put(X) + sig * rng.standard_normal(N)
                r = select_alpha(X, Y, X0, a0, "epanechnikov", M, NU, "literal")
                eb.append(r["beta_q"])
                ea.append(r["alpha_raw"])
            oracle = ((2 * NU + 1) * k["a"] * sig ** 2 / (2 * (k["q"] - NU) * k["b"] ** 2
                      * b10 ** 2 * N * 0.75 / a0)) ** (1 / (2 * k["q"] + 1)) if sig > 0 else np.nan
            ob = float(np.median(eb))
            lines.append(f"| {sig:.0e} | {a0} | {ob:.3e} | {ob / b10:.2g} | "
                         f"{np.median(ea):.2f} | {oracle:.2f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
