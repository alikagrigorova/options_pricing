"""Diagnostic: exercise rule and parameter design of the multivariate ISD
(simgreeks.multi, not part of the paper).

The first American check showed Volga biased upwards on 21 of 33 options
(+50 % median). This locates the cause on the (S, sigma) run alone:

  insample-epa : exercise rule fitted on the same paths, Epanechnikov sigma
                 design (the first design)
  insample-uni : same paths, uniform sigma design
  pilot-epa    : rule fitted on independent pilot paths, Epanechnikov, same box
  pilot-uni    : independent pilot paths, uniform design, pilot box 1.3x wider
                 (the design that followed; the current one also chooses the
                 widths on the pilot and decides exercise at t_1 by the rule)

Seven options; references from reference.put_fd_greeks.

    python experiments/diagnostics/multi_design_check.py [--reps 40] [--workers W] [--N 100000]

Writes results/diagnostics/multi_design/summary.md and raw.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import MultiConfig, label  # noqa: E402
from simgreeks.reference import put_fd_greeks  # noqa: E402
from simgreeks.report import Z99  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "multi_design"
SPECS = [PutSpec(K=36), PutSpec(K=40), PutSpec(K=44), PutSpec(K=36, sigma=0.1),
         PutSpec(K=40, sigma=0.4, T=0.5), PutSpec(K=40, r=0.0), PutSpec(K=44, d=0.06)]
# Fixed widths, max(Z, C) at t_1 and no exercise at t_0, so that only the
# exercise rule and the sigma design differ between the variants.
BASE = dict(groups=(("S", "sigma"),), widths="fixed", vf_rule="max", american_t0=False)
VARIANTS = {
    "insample-epa": MultiConfig(**BASE, exercise_rule="insample", param_kernel="epanechnikov"),
    "insample-uni": MultiConfig(**BASE, exercise_rule="insample"),
    "pilot-epa": MultiConfig(**BASE, param_kernel="epanechnikov", pilot_widen=1.0),
    "pilot-uni": MultiConfig(**BASE),
}
QUANTS = ["price", "delta", "gamma", "vega", "volga", "vanna"]


def timed_label(spec, cfg, seed):
    t0 = time.process_time()
    out = label(spec, cfg, seed)
    out["cpu_s"] = time.process_time() - t0
    return out


def option_name(s):
    return f"K={s.K:g} σ={s.sigma:g} T={s.T:g} r={s.r:g} d={s.d:g}"


def summarise(df, variants, quants, refs):
    rows = []
    for v in variants:
        for i, ref in enumerate(refs):
            g = df[(df.variant == v) & (df.i == i)]
            for q in quants:
                m, sd, n = g[q].mean(), g[q].std(), g[q].count()
                rows.append(dict(variant=v, i=i, q=q, ref=ref[q], bias=m - ref[q], sd=sd,
                                 t=(m - ref[q]) / (sd / np.sqrt(n))))
    return pd.DataFrame(rows)


def tables(out, df, variants, quants):
    lines = ["Per variant: options flagged (|t| > 2.576) / median |bias| / |ref| / "
             "median sd of one label / |ref| (medians over options with |ref| above 1 % "
             "of the largest |ref| for that Greek). CPU s: mean CPU seconds per label "
             "(one process).\n",
             "| variant | CPU s | " + " | ".join(quants) + " |",
             "|---|---|" + "---|" * len(quants)]
    for v in variants:
        cells = []
        for q in quants:
            g = out[(out.variant == v) & (out.q == q)]
            big = g[g.ref.abs() > 0.01 * g.ref.abs().max()]
            cells.append(f"{int((g.t.abs() > Z99).sum())} / {(big.bias.abs() / big.ref.abs()).median():.1%}"
                         f" / {(big.sd / big.ref.abs()).median():.0%}")
        lines.append(f"| {v} | {df[df.variant == v].cpu_s.mean():.2f} | " + " | ".join(cells) + " |")
    lines += ["", "t = bias / (sd/√R) by option:\n",
              "| option | variant | " + " | ".join(quants) + " |",
              "|---|---|" + "---|" * len(quants)]
    for i, s in enumerate(SPECS):
        for v in variants:
            g = out[(out.variant == v) & (out.i == i)].set_index("q")
            lines.append(f"| {option_name(s)} | {v} | "
                         + " | ".join(f"{g.t[q]:+.1f}{'†' if abs(g.t[q]) > Z99 else ''}"
                                      for q in quants) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--N", type=int, default=100_000, help="paths per run")
    args = ap.parse_args()
    variants = {v: replace(c, N=args.N) for v, c in VARIANTS.items()}
    configs = [dict(spec=s, cfg=c, fn=timed_label, tags=dict(i=i, variant=v))
               for i, s in enumerate(SPECS) for v, c in variants.items()]
    df = run_configs(configs, args.reps, base_seed=7003, workers=args.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "raw.csv", index=False)
    refs = [put_fd_greeks(s) for s in SPECS]
    lines = ["# Multivariate ISD: exercise rule and σ design, (S, σ) run\n",
             f"{args.reps} replications per option and variant, {args.N:,} paths per run "
             "(and per pilot); references from "
             "reference.put_fd_greeks. See the script's docstring for the variants.\n"]
    lines += tables(summarise(df, VARIANTS, QUANTS, refs), df, VARIANTS, QUANTS)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
