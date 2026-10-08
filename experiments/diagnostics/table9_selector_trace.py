"""Trace the paper's alpha* selector for nu = 0, 1, 2 (the Table 9 ATM case).

S0 = K = 40, alpha0 = 10, M0 = 9, N = 100,000, Epanechnikov deterministic ISD,
100 replications. Both readings of Appendix A.2's constants:
  literal : as printed (default of simgreeks.selector)
  fg      : Fan & Gijbels' constants
Every replication's intermediate quantities (q, a, b, h_ROT, local beta_q,
local sigma^2, f(x0), alpha*, global pilot coefficients) are written to
results/diagnostics/table9_trace/trace.csv; the 2-step Greeks with each alpha*
are compared with the paper's Table 9 (Epa.D., K = 40).

    python experiments/diagnostics/table9_selector_trace.py [--reps 100]
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

from simgreeks.core import PutSpec, greeks_regression, isd_sample, lsm, simulate_growth  # noqa: E402
from simgreeks.methods import AlgoConfig, second_step  # noqa: E402
from simgreeks.selector import poly_fit, select_alpha  # noqa: E402

X0, ALPHA0, M, N = 40.0, 10.0, 9, 100_000
OUT = ROOT / "results" / "diagnostics" / "table9_trace"


def one_rep(rep):
    spec = PutSpec(K=40.0)
    cfg = AlgoConfig(alpha=ALPHA0)
    rng = np.random.default_rng(np.random.SeedSequence([9009, rep]))
    G = simulate_growth(spec, N, rng)
    X = isd_sample(N, X0, ALPHA0, "epanechnikov")
    pilot = lsm(spec, X[:, None] * G, 9)
    glob = poly_fit(X, pilot.Y_vf, X0, ALPHA0, M + 3)
    rows = []
    for reading in ("literal", "fg"):
        for nu in (0, 1, 2):
            r = select_alpha(X, pilot.Y_vf, X0, ALPHA0, "epanechnikov", M, nu, reading, glob)
            X2, Y2 = second_step(spec, cfg, X, G, pilot, ALPHA0, r["alpha_star"], "refit_t1")
            p, d, g = greeks_regression(X2, Y2, X0, 9)
            coefs = " ".join(f"{c:.6e}" for c in r.pop("global_coefs"))
            rows.append(dict(r, reading=reading, nu=nu, rep=rep, price=p, delta=d, gamma=g,
                             global_coefs=coefs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor() as ex:
        for out in ex.map(one_rep, range(args.reps)):
            rows.extend(out)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trace.csv", index=False)

    paper = pd.read_csv(ROOT / "data" / "paper_tables.csv")
    paper = paper[(paper.table == 9) & (paper.ISD == "Epa.D.") & (paper.K == 40)] \
        .drop_duplicates("nu").set_index("nu")
    lines = ["# Table 9 selector trace (ATM, alpha0 = 10, M0 = 9, N = 100,000)\n",
             f"{args.reps} replications; 2-step Greeks with the selected alpha*. "
             "Paper: Table 9, Epa.D., K = 40. Per-replication trace in trace.csv.\n",
             "| reading | nu | alpha* mean | sd | min | max | price sd | delta sd | "
             "gamma sd | paper price sd | paper delta sd | paper gamma sd |",
             "|" + "---|" * 12]
    for (rd, nu), g in df.groupby(["reading", "nu"], sort=False):
        pr = paper.loc[nu]
        lines.append("| " + " | ".join([
            rd, str(nu), f"{g.alpha_star.mean():.2f}", f"{g.alpha_star.std():.2f}",
            f"{g.alpha_star.min():.2f}", f"{g.alpha_star.max():.2f}",
            f"{g.price.std():.4f}", f"{g.delta.std():.4f}", f"{g.gamma.std():.4f}",
            f"{float(pr.price_paper_sd):.4f}", f"{float(pr.delta_paper_sd):.4f}",
            f"{float(pr.gamma_paper_sd):.4f}"]) + " |")
    const = df.drop_duplicates(["reading", "nu"])[["reading", "nu", "q", "a", "b", "b_rot"]]
    lines.append("\nConstants:\n\n" + const.to_markdown(index=False))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
