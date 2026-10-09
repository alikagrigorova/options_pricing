"""Diagnostic: why the generator's price is low (simgreeks.multi, not part of
the paper).

The American check flags 24 of 33 prices, all low by about 0.1 %, while the
paper flags none. Each label's price comes from an exercise rule fitted on the
pilot paths. Here, for six of the most-flagged options and for a run of each
group alone (S only, as in the paper, and (S, sigma), (S, r), (S, d)), we
record

  rule  : the value at z0 of the run's exercise rule, on 100,000 paths that all
          start at z0 (no regression at t = 0; European control variate with
          coefficient 1). The four rules share these paths.
  label : the run's price estimate (t_1 value function, t = 0 regression).

rule - ref is how far the estimated rule is from the optimal one (a lower bound,
so <= 0 up to Monte Carlo noise); label - rule is what the t_1 value function
and the t = 0 regression add. The default label averages the (S, sigma),
(S, r) and (S, d) runs.

    python experiments/diagnostics/price_bias_check.py [--reps 50] [--workers W]
    python experiments/diagnostics/price_bias_check.py --report   # tables from raw.csv

Writes results/diagnostics/price_bias/summary.md and raw.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import (MultiConfig, _center, brownian, greeks_multi, isd_multi,  # noqa: E402
                             lsm_multi, pilot, stock_at)
from simgreeks.reference import bs_put, put_fd_greeks  # noqa: E402
from simgreeks.report import Z99  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "price_bias"
KEYS = ["T", "sigma", "r", "d", "K"]
OPTIONS = [(0.5, 0.20, 0.06, 0.0, 36), (1.0, 0.40, 0.06, 0.0, 40), (2.0, 0.20, 0.06, 0.0, 40),
           (1.0, 0.20, 0.06, 0.06, 36), (1.0, 0.20, 0.0, 0.0, 36), (0.5, 0.40, 0.06, 0.0, 44)]
GROUPS = {"S": ("S",), "S,sigma": ("S", "sigma"), "S,r": ("S", "r"), "S,d": ("S", "d")}
CFG = MultiConfig()
N_Z0 = 100_000                  # paths starting at z0 that value each rule


def spec_of(option):
    T, s, r, d, K = option
    return PutSpec(K=K, sigma=s, T=T, r=r, d=d)


def run_and_rule(spec, dims, rng):
    """simgreeks.multi.run_group's price, and the exercise rule it used."""
    alphas, scale, rules = pilot(spec, CFG, dims, rng)
    z = isd_multi(spec, CFG, alphas, rng)
    _, Y, _ = lsm_multi(spec, CFG, z, brownian(spec, CFG.N, rng), scale, rules)
    return greeks_multi(spec, CFG, dims, z, Y, bs_put(spec))["price"], rules, scale


def replication(spec, _cfg, seed):
    seeds = seed.spawn(len(GROUPS) + 1)
    z0 = {k: np.full(N_Z0, v) for k, v in _center(spec).items()}
    W = brownian(spec, N_Z0, np.random.default_rng(seeds[0]))
    euro = np.exp(-spec.r * spec.T) * spec.payoff(stock_at(spec, z0, W, spec.J))
    out = {}
    for (name, dims), s in zip(GROUPS.items(), seeds[1:]):
        price, rules, scale = run_and_rule(spec, dims, np.random.default_rng(s))
        Y = lsm_multi(spec, CFG, z0, W, scale, rules, value_function=False)[0]
        out[f"label|{name}"] = price
        out[f"rule|{name}"] = float((Y - euro).mean() + bs_put(spec)["price"])
    return out


def report(mc):
    mc = mc.copy()
    mc["label|default"] = mc[[f"label|{g}" for g in list(GROUPS)[1:]]].mean(axis=1)
    cols = {"rule": list(GROUPS), "label": list(GROUPS) + ["default"]}
    rows = []
    for option in OPTIONS:
        g = mc
        for k, v in zip(KEYS, option):
            g = g[np.isclose(g[k], v)]
        ref = put_fd_greeks(spec_of(option))["price"]
        for what, names in cols.items():
            for name in names:
                x = g[f"{what}|{name}"] - ref
                rows.append(dict(zip(KEYS, option), ref=ref, what=what, group=name,
                                 bias=x.mean(), t=x.mean() / (x.std() / np.sqrt(len(x)))))
    out = pd.DataFrame(rows)
    lines = ["# Why the generator's price is low\n",
             f"{len(mc) // len(OPTIONS)} replications per option, {CFG.N:,} pilot and "
             f"{CFG.N:,} main paths per run, default MultiConfig; each rule valued on "
             f"{N_Z0:,} paths from z0. Reference: reference.put_fd_greeks (Bermudan). "
             "See the script's docstring.\n"]
    for what, title in (("rule", "Value of the exercise rule at z0 − reference"),
                        ("label", "Label price − reference")):
        names = cols[what]
        lines += [f"## {title}\n", "bias (t); † if |t| > 2.576\n",
                  "| T | σ | r | d | K | " + " | ".join(names) + " |",
                  "|---|---|---|---|---|" + "---|" * len(names)]
        o = out[out.what == what]
        for option in OPTIONS:
            T, s, r, d, K = option
            g = o[np.isclose(o["T"], T) & np.isclose(o.sigma, s) & np.isclose(o.r, r)
                  & np.isclose(o.d, d) & np.isclose(o.K, K)].set_index("group")
            lines.append(f"| {T:g} | {s:.0%} | {r:.0%} | {d:.0%} | {K:g} | " + " | ".join(
                f"{g.bias[n]:+.4f} ({g.t[n]:+.1f}{'†' if abs(g.t[n]) > Z99 else ''})"
                for n in names) + " |")
        mean = o.groupby("group").bias.mean()
        rel = (o.bias / o.ref).groupby(o.group).mean()
        lines += ["| mean | | | | | " + " | ".join(f"{mean[n]:+.4f} ({rel[n]:+.2%})"
                                                for n in names) + " |", ""]
    lines += ["## Paired differences\n", "| | mean | se |", "|---|---|---|"]
    multi = mc[[f"rule|{g}" for g in list(GROUPS)[1:]]].mean(axis=1)
    for name, x in [("rule of the multivariate runs − rule of S only", multi - mc["rule|S"])] + \
            [(f"label − rule, {g}", mc[f"label|{g}"] - mc[f"rule|{g}"]) for g in GROUPS]:
        lines.append(f"| {name} | {x.mean():+.4f} | {x.std() / np.sqrt(len(x)):.4f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="only rebuild summary.md from raw.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        mc = pd.read_csv(OUT / "raw.csv")
    else:
        configs = [dict(spec=spec_of(o), cfg=CFG, fn=replication, tags=dict(zip(KEYS, o)))
                   for o in OPTIONS]
        mc = run_configs(configs, args.reps, base_seed=4242, workers=args.workers)
        mc.to_csv(OUT / "raw.csv", index=False)
    md = report(mc)
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
