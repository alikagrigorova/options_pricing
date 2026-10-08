"""Diagnostic: exercise-rule basis of the multivariate method (simgreeks.multi).

The full American check (experiments/multi_check.py) shows small but significant
price biases, largest at r = d = 0 (+0.005 to +0.014). Running each group alone
shows the bias in every group, also (S, sigma) where r is not dispersed, and
also in the 1-D NAIVE-VF / 2-step methods with the placeholder alpha*. With
MultiConfig(weight=9), i.e. an exercise rule in S plus a linear parameter
shift, the (S, sigma) run alone was unbiased at r = 0 (+0.0003, t = 0.2),
suggesting in-sample (foresight) bias of the richer exercise regression.

This compares weight = 3 (current default) and weight = 9 on the problem
options (r = 0; r = d = 6%; sigma = 40%) and a few controls, for all Greeks.
Result (results/diagnostics/exercise_basis/summary.md): weight 9 is NOT a fix.
It only shrinks the r = 0 price bias (the (S, r) and (S, d) runs stay biased),
and it biases Vega (t up to 11) and the price of the control options.

    python experiments/diagnostics/exercise_basis_check.py [--reps 60] [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import MultiConfig, label  # noqa: E402
from simgreeks.reference import put_fd_greeks  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

Q = ["price", "delta", "gamma", "vega", "rho", "rho_d", "vanna"]
OUT = ROOT / "results" / "diagnostics" / "exercise_basis"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    specs = [PutSpec(K=K, r=0.0) for K in (36, 40, 44)] + \
            [PutSpec(K=K, d=0.06) for K in (36, 40, 44)] + \
            [PutSpec(K=36, sigma=0.4), PutSpec(K=40, sigma=0.4), PutSpec(K=44),
             PutSpec(K=36, sigma=0.1, T=0.5), PutSpec(K=40)]
    cfgs = {"weight=3": MultiConfig(), "weight=9": MultiConfig(weight=9)}
    configs = [dict(spec=s, cfg=c, fn=label, tags=dict(i=i, w=w))
               for i, s in enumerate(specs) for w, c in cfgs.items()]
    df = run_configs(configs, args.reps, workers=args.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "raw.csv", index=False)
    lines = ["# Exercise-rule basis: weight 3 vs weight 9\n",
             f"{args.reps} replications; bias = mean − reference (put_fd_greeks), "
             "t = bias / (sd/√R).\n",
             "| option | weight | " + " | ".join(Q) + " |",
             "|---|---|" + "---|" * len(Q)]
    for i, s in enumerate(specs):
        ref = put_fd_greeks(s)
        for w in cfgs:
            g = df[(df.i == i) & (df.w == w)]
            cells = [f"{g[q].mean() - ref[q]:+.4f} (t={(g[q].mean() - ref[q]) / (g[q].std() / np.sqrt(len(g))):+.1f})"
                     for q in Q]
            lines.append(f"| K={s.K:g} σ={s.sigma:g} T={s.T:g} r={s.r:g} d={s.d:g} | {w} | "
                         + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
