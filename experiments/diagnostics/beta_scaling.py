"""How do the selector's ingredients scale with the initial ISD size alpha0?

Paper selector (literal reading), Gamma target (M0 = 9, nu = 2), ATM
(S0 = K = 40), alpha0 in {0.5, 1, 2.5, 5, 10, 15, 20, 25}, 100 replications.

Fits log Q = a + b log alpha0 for Q in h_ROT, |beta_10|, sigma^2(x0), f(x0),
alpha*. If beta_10 is noise from a window of width ~alpha0, b(beta) ~ -10, and
with f ~ alpha0^-1 the plug-in (A.1) gives alpha* ~ (alpha0^20 alpha0)^(1/21)
= alpha0. A genuinely estimated curvature would give b(beta) ~ 0.

    python experiments/diagnostics/beta_scaling.py [--reps 100]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec, isd_sample, lsm, simulate_growth  # noqa: E402
from simgreeks.selector import select_alpha  # noqa: E402

X0, M, N = 40.0, 9, 100_000
OUT = ROOT / "results" / "diagnostics" / "beta_scaling"
ALPHAS = (0.5, 1, 2.5, 5, 10, 15, 20, 25)
QUANTS = ("h_rot", "abs_beta_q", "sigma2_local", "f_x0", "alpha_star")


def one(args):
    a0, rep = args
    spec = PutSpec(K=40.0)
    rng = np.random.default_rng(np.random.SeedSequence([1010, int(a0 * 10), rep]))
    G = simulate_growth(spec, N, rng)
    X = isd_sample(N, X0, a0, "epanechnikov")
    Y = lsm(spec, X[:, None] * G, 9).Y_vf
    r = select_alpha(X, Y, X0, a0, "epanechnikov", M, 2, "literal")
    return dict(alpha0=a0, rep=rep, h_rot=r["h_rot"], beta_q=r["beta_q"],
                abs_beta_q=abs(r["beta_q"]), sigma2_local=r["sigma2_local"],
                f_x0=r["f_x0"], alpha_star=r["alpha_star"])


def slope(x, y):
    lx, ly = np.log(x), np.log(y)
    A = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = ly - A @ coef
    se = np.sqrt(r @ r / (len(ly) - 2) * np.linalg.inv(A.T @ A)[1, 1])
    return coef[1], se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, [(a, r) for a in ALPHAS for r in range(args.reps)], chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "raw.csv", index=False)

    med = df.groupby("alpha0")[list(QUANTS)].median()
    neg = df.groupby("alpha0").beta_q.apply(lambda s: (s < 0).mean())
    lines = ["# Scaling of the selector's ingredients with alpha0 (literal, nu = 2, M0 = 9, ATM)\n",
             f"{args.reps} replications per alpha0; medians, then log-log slopes (OLS on all "
             "replications, slope ± standard error).\n",
             "| alpha0 | h_ROT | median abs(beta_10) | share beta_10 < 0 | sigma^2(x0) | f(x0) | "
             "alpha* | alpha*/alpha0 |", "|---|---|---|---|---|---|---|---|"]
    for a in ALPHAS:
        m = med.loc[a]
        lines.append(f"| {a:g} | {m.h_rot:.3f} | {m.abs_beta_q:.3e} | {neg[a]:.2f} | "
                     f"{m.sigma2_local:.4f} | {m.f_x0:.4f} | {m.alpha_star:.3f} | "
                     f"{m.alpha_star / a:.3f} |")
    lines += ["\n| quantity | slope | ± se | if beta_10 is noise |", "|---|---|---|---|"]
    pred = {"h_rot": "1", "abs_beta_q": "-10", "sigma2_local": "~0", "f_x0": "-1 (exact)",
            "alpha_star": "1"}
    for q in QUANTS:
        b, se = slope(df.alpha0.values, df[q].values)
        lines.append(f"| {q} | {b:.3f} | {se:.3f} | {pred[q]} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
