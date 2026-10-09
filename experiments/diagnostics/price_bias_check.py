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
                         [--set name=value ...] [--name NAME]
    python experiments/diagnostics/price_bias_check.py --report [--name NAME]
    python experiments/diagnostics/price_bias_check.py --compare default NAME ...

--set changes a MultiConfig field (multi.config_from), e.g. --set N_pilot=200_000;
the seeds do not depend on it, so variants share the paths from z0. Writes
results/diagnostics/price_bias/[NAME/]summary.md, raw.csv and settings.json
(--report rebuilds summary.md from raw.csv). --compare writes comparison.md:
each variant's mean rule and label bias, paired against the first variant, and
CPU seconds per run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import (_center, brownian, config_from, greeks_multi, isd_multi,  # noqa: E402
                             lsm_multi, pilot, stock_at)
from simgreeks.reference import bs_put, put_fd_greeks  # noqa: E402
from simgreeks.report import Z99  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "price_bias"
KEYS = ["T", "sigma", "r", "d", "K"]
OPTIONS = [(0.5, 0.20, 0.06, 0.0, 36), (1.0, 0.40, 0.06, 0.0, 40), (2.0, 0.20, 0.06, 0.0, 40),
           (1.0, 0.20, 0.06, 0.06, 36), (1.0, 0.20, 0.0, 0.0, 36), (0.5, 0.40, 0.06, 0.0, 44)]
GROUPS = {"S": ("S",), "S,sigma": ("S", "sigma"), "S,r": ("S", "r"), "S,d": ("S", "d")}
N_Z0 = 100_000                  # paths starting at z0 that value each rule


def spec_of(option):
    T, s, r, d, K = option
    return PutSpec(K=K, sigma=s, T=T, r=r, d=d)


def run_and_rule(spec, cfg, dims, rng):
    """simgreeks.multi.run_group's price, and the exercise rule it used."""
    alphas, scale, rules = pilot(spec, cfg, dims, rng)
    z = isd_multi(spec, cfg, alphas, rng)
    _, Y, _ = lsm_multi(spec, cfg, z, brownian(spec, cfg.N, rng), scale, rules)
    return greeks_multi(spec, cfg, dims, z, Y, bs_put(spec))["price"], rules, scale


def replication(spec, cfg, seed):
    seeds = seed.spawn(len(GROUPS) + 1)
    z0 = {k: np.full(N_Z0, v) for k, v in _center(spec).items()}
    W = brownian(spec, N_Z0, np.random.default_rng(seeds[0]))
    euro = np.exp(-spec.r * spec.T) * spec.payoff(stock_at(spec, z0, W, spec.J))
    out = {}
    for (name, dims), s in zip(GROUPS.items(), seeds[1:]):
        t0 = time.process_time()
        price, rules, scale = run_and_rule(spec, cfg, dims, np.random.default_rng(s))
        out[f"cpu|{name}"] = time.process_time() - t0
        Y = lsm_multi(spec, cfg, z0, W, scale, rules, value_function=False)[0]
        out[f"label|{name}"] = price
        out[f"rule|{name}"] = float((Y - euro).mean() + bs_put(spec)["price"])
    return out


def out_dir(name):
    return OUT if name in (None, "", "default") else OUT / name


def with_default_label(mc):
    mc = mc.copy()
    mc["label|default"] = mc[[f"label|{g}" for g in list(GROUPS)[1:]]].mean(axis=1)
    return mc


def report(mc, settings):
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
             f"{len(mc) // len(OPTIONS)} replications per option; MultiConfig "
             f"{'with ' + ', '.join(settings) if settings else 'default'}; each rule "
             f"valued on {N_Z0:,} paths from z0. Reference: reference.put_fd_greeks "
             "(Bermudan). See the script's docstring.\n"]
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
    if f"cpu|{list(GROUPS)[1]}" in mc:
        lines += ["", "CPU seconds per run (one process), mean: " + ", ".join(
            f"{g} {mc[f'cpu|{g}'].mean():.2f}" for g in GROUPS)]
    return "\n".join(lines) + "\n"


def compare(names):
    """Mean bias over the options of the rule value and the label, for each
    variant, and the paired difference to the first variant (same seeds), over the
    (option, replication) pairs every variant has."""
    refs = {o: put_fd_greeks(spec_of(o))["price"] for o in OPTIONS}
    data = {n: pd.read_csv(out_dir(n) / "raw.csv") for n in names}
    common = None
    for mc in data.values():
        keys = mc[KEYS + ["rep"]]
        common = keys if common is None else common.merge(keys)
    data = {n: with_default_label(mc.merge(common).sort_values(KEYS + ["rep"],
                                                                ignore_index=True))
            for n, mc in data.items()}
    settings = {n: json.loads((out_dir(n) / "settings.json").read_text())
                if (out_dir(n) / "settings.json").exists() else [] for n in names}
    ref = pd.Series([refs[tuple(r)] for r in data[names[0]][KEYS].itertuples(index=False)])
    lines = ["# Price bias by variant\n",
             "Mean over the six options of (value − reference) / reference, ± standard "
             "error; rule: value of the exercise rule at z0; label: the run's price. "
             "S: the run with S alone dispersed; multivariate: mean of the (S, σ), "
             "(S, r), (S, d) runs; label default: the default label (their mean). Δ vs "
             f"{names[0]}: paired difference (same seeds). CPU: mean seconds per "
             "multivariate run (one process). Replications: those every variant has.\n",
             "| variant | settings | rule, S | rule, multivariate | label default "
             f"| Δ rule multivariate vs {names[0]} | Δ label vs {names[0]} | CPU s |",
             "|---|---|---|---|---|---|---|---|"]
    base = data[names[0]]

    def stat(x):
        x = x / ref
        return f"{x.mean():+.3%} ± {x.std() / np.sqrt(len(x)):.3%}"

    for n in names:
        mc = data[n]
        multi = mc[[f"rule|{g}" for g in list(GROUPS)[1:]]].mean(axis=1)
        base_multi = base[[f"rule|{g}" for g in list(GROUPS)[1:]]].mean(axis=1)
        cpu = (mc[[f"cpu|{g}" for g in list(GROUPS)[1:]]].to_numpy().mean()
               if f"cpu|{list(GROUPS)[1]}" in mc else np.nan)
        lines.append(f"| {n} | {', '.join(settings[n]) or 'default'} | "
                     f"{stat(mc['rule|S'] - ref)} | {stat(multi - ref)} | "
                     f"{stat(mc['label|default'] - ref)} | {stat(multi - base_multi)} | "
                     f"{stat(mc['label|default'] - base['label|default'])} | {cpu:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="MultiConfig field to change (repeatable)")
    ap.add_argument("--name", default="default", help="output subdirectory of the variant")
    ap.add_argument("--report", action="store_true", help="only rebuild summary.md from raw.csv")
    ap.add_argument("--compare", nargs="+", metavar="NAME", help="compare finished variants")
    args = ap.parse_args()
    if args.compare:
        md = compare(args.compare)
        (OUT / "comparison.md").write_text(md)
        print(md)
        return
    out = out_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)
    if args.report:
        mc = pd.read_csv(out / "raw.csv")
        settings = json.loads((out / "settings.json").read_text()) \
            if (out / "settings.json").exists() else []
    else:
        cfg, settings = config_from(args.set), args.set
        configs = [dict(spec=spec_of(o), cfg=cfg, fn=replication, tags=dict(zip(KEYS, o)))
                   for o in OPTIONS]
        mc = run_configs(configs, args.reps, base_seed=4242, workers=args.workers)
        mc.to_csv(out / "raw.csv", index=False)
        (out / "settings.json").write_text(json.dumps(settings))
    md = report(mc, settings)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
