"""Validate the simulated Theta (Black-Scholes PDE identity, core.theta_pde)
against a finite-difference Theta from the reference PDE solver.

Grid: the 27 options of Table 5 plus the (r, d) = (0, 0) and (6%, 6%) options
of Table 6. The 2-step method uses the placeholder alpha* = 0.6 S0 sigma
sqrt(T) (not part of the paper) with initial alpha = 10, N = 100,000,
M_tau = M0 = 9.

Reference (simgreeks.reference.put_fd, Bermudan, same exercise grid as LSM):
  theta_C   : dC/dt at t_0 of the continuation value C, by a central difference
              in calendar time
  theta_ref : the convention of core.theta_pde, i.e. 0 if K - S0 >= C (exercise
              region) and theta_C otherwise; the MC Theta is tested against it
  theta_id  : r C - (r - d) S0 Delta - 1/2 sigma^2 S0^2 Gamma from the
              reference's own C, Delta and Gamma (checks the identity itself)
  theta_am  : Theta of the American put (exercise at every time step), for
              information

    python experiments/theta_check.py [--reps 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.methods import AlgoConfig  # noqa: E402
from simgreeks.reference import FDGrid, put_fd  # noqa: E402
from simgreeks.report import Z99  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

RESULTS = ROOT / "results"
KEYS = ["T", "sigma", "r", "d", "K"]


def grid():
    cells = [(T, s, 0.06, 0.0, K) for T in (0.5, 1.0, 2.0) for s in (0.10, 0.20, 0.40)
             for K in (36, 40, 44)]
    cells += [(1.0, 0.20, r, d, K) for r, d in ((0.0, 0.0), (0.06, 0.06)) for K in (36, 40, 44)]
    return cells


def reference(cells):
    rows = []
    for T, s, r, d, K in cells:
        spec = PutSpec(K=K, sigma=s, T=T, r=r, d=d)
        f = put_fd(spec, "bermudan")
        am = put_fd(spec, "american", FDGrid(steps_per_dt=200))
        theta_id = (r * f["price"] - (r - d) * spec.S0 * f["delta"]
                    - 0.5 * s ** 2 * spec.S0 ** 2 * f["gamma"])
        rows.append(dict(T=T, sigma=s, r=r, d=d, K=K, price_ref=f["price"],
                         delta_ref=f["delta"], gamma_ref=f["gamma"], theta_C=f["theta"],
                         theta_id=theta_id, ex_ref=f["ex_region"],
                         theta_ref=0.0 if f["ex_region"] else f["theta"],
                         theta_am=am["theta"]))
    return pd.DataFrame(rows)


def simulate(cells, reps):
    cfg = AlgoConfig(alpha=10, methods=("2STEP-VF",), alpha_star_rule="heuristic",
                     alpha_star_c=0.6)
    configs = [dict(spec=PutSpec(K=K, sigma=s, T=T, r=r, d=d), cfg=cfg,
                    tags=dict(T=T, sigma=s, r=r, d=d, K=K))
               for T, s, r, d, K in cells]
    return run_configs(configs, reps, base_seed=20261008)


def summarise(mc, ref):
    agg = mc.groupby(KEYS).agg(theta_mean=("theta", "mean"), theta_sd=("theta", "std"),
                               n=("theta", "count"), ex_freq=("ex_region", "mean"),
                               alpha_star=("alpha_star", "mean")).reset_index()
    out = ref.merge(agg, on=KEYS)
    out["t"] = (out.theta_mean - out.theta_ref) / (out.theta_sd / np.sqrt(out.n))
    out["sig"] = out.t.abs() > Z99
    return out


def to_markdown(out, reps):
    head = ["T", "σ", "r", "d", "K", "α*", "Θ_C (FD)", "Θ_id − Θ_C", "Θ_ref",
            "MC Θ mean", "MC Θ sd", "t", "ex-region freq", "Θ American"]
    lines = ["# Theta check: PDE identity on the 2-step estimates vs reference FD Theta\n",
             f"{reps} replications per option; 2-step method with the placeholder "
             "alpha* = 0.6 S0 sigma sqrt(T) (not the paper's selector), initial alpha = 10, "
             "N = 100,000, M_tau = M0 = 9. Theta = dV/dt per year (calendar time).\n",
             "Θ_C: reference Theta of the Bermudan continuation value (no exercise at t_0, "
             "as in LSM and the paper's benchmarks). Θ_id − Θ_C: the PDE identity applied "
             "to the reference's own price, Delta and Gamma, minus Θ_C. Θ_ref: 0 in the "
             "exercise region (K − S0 ≥ C), Θ_C otherwise; this is the target of the MC "
             "Theta. t = (MC mean − Θ_ref) / (sd / √R), † if |t| > 2.576. "
             "ex-region freq: share of replications classified in the exercise region. "
             "Θ American: exercise at every time step, for information.\n",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in out.iterrows():
        lines.append("| " + " | ".join([
            f"{r['T']:g}", f"{r.sigma:.0%}", f"{r.r:.0%}", f"{r.d:.0%}", f"{r.K:g}",
            f"{r.alpha_star:.2f}", f"{r.theta_C:+.4f}", f"{r.theta_id - r.theta_C:+.1e}",
            f"{r.theta_ref:+.4f}", f"{r.theta_mean:+.4f}", f"{r.theta_sd:.4f}",
            f"{r.t:+.2f}{'†' if r.sig else ''}", f"{r.ex_freq:.2f}", f"{r.theta_am:+.4f}"])
            + " |")
    lines.append(f"\nSignificant at 1%: {int(out.sig.sum())} of {len(out)}.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    cells = grid()
    ref = reference(cells)
    mc = simulate(cells, args.reps)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    mc.to_csv(RESULTS / "raw" / "theta_check.csv", index=False)
    out = summarise(mc, ref)
    out.to_csv(RESULTS / "theta_check_summary.csv", index=False)
    (RESULTS / "theta_check.md").write_text(to_markdown(out, args.reps))
    print(to_markdown(out, args.reps))


if __name__ == "__main__":
    main()
