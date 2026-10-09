"""Validate the multivariate-ISD Greeks (simgreeks.multi, not part of the paper).

  european : no early exercise and no control variate, against the closed-form
             Black-Scholes-Merton Greeks; checks the multivariate regressions.
  american : Bermudan put (paper's exercise grid), default MultiConfig, on the
             27 options of Table 5 plus the (r, d) = (0, 0) and (6%, 6%)
             options of Table 6, against the reference PDE pricer: Greeks in
             sigma, r, d by bumping (reference.put_fd_greeks).
  timing   : single-process seconds per label for T = 0.5, 1, 2.

    python experiments/multi_check.py european american timing [--reps 100] [--workers N]

Writes results/multi_check_<exp>.md and results/raw/multi_check_<exp>.csv.
Labels are saved after each option (results/raw/multi_check_<exp>.partial.csv),
and a rerun resumes from there.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import MultiConfig, label  # noqa: E402
from simgreeks.reference import bs_put, put_fd_greeks  # noqa: E402
from simgreeks.report import Z99  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402

RESULTS = ROOT / "results"
GREEK_LETTERS = {"sigma": "σ"}
KEYS = ["T", "sigma", "r", "d", "K"]
QUANTS = ["price", "delta", "gamma", "theta", "vega", "volga", "rho", "phi", "vanna",
          "delta_r", "delta_d"]


def american_grid():
    cells = [(T, s, 0.06, 0.0, K) for T in (0.5, 1.0, 2.0) for s in (0.10, 0.20, 0.40)
             for K in (36, 40, 44)]
    return cells + [(1.0, 0.20, r, d, K) for r, d in ((0.0, 0.0), (0.06, 0.06))
                    for K in (36, 40, 44)]


def european_grid():
    return [(1.0, 0.20, 0.06, 0.0, K) for K in (36, 40, 44)] + \
           [(2.0, 0.40, 0.06, 0.0, 40), (0.5, 0.10, 0.06, 0.0, 40), (1.5, 0.30, 0.03, 0.02, 40)]


def spec_of(cell):
    T, s, r, d, K = cell
    return PutSpec(K=K, sigma=s, T=T, r=r, d=d)


def reference(cells, style):
    rows = []
    for cell in cells:
        spec = spec_of(cell)
        if style == "european":
            f = bs_put(spec)
        else:
            f = put_fd_greeks(spec)
            if f["ex_region"]:      # exercise at t_0 (MultiConfig.american_t0)
                f.update({q: 0.0 for q in QUANTS})
                f["price"], f["delta"] = spec.K - spec.S0, -1.0
        rows.append(dict(zip(KEYS, cell), **{q: f[q] for q in QUANTS}))
    return pd.DataFrame(rows)


def rows_of(df, tags):
    for k, v in tags.items():
        df = df[np.isclose(df[k], v)]
    return df


def simulate(cells, cfg, reps, seed, partial: Path, workers=None):
    """Labels for each option. Each option's labels are appended to ``partial``
    (a CSV) as soon as they are done, and options already there are skipped, so
    an interrupted check resumes where it stopped. Seeds depend only on the
    option and the replication, so a resumed check gives the same labels.
    Delete ``partial`` after changing the configuration."""
    done = pd.read_csv(partial) if partial.exists() else None
    if done is not None:
        print(f"   resuming from {partial.name}", flush=True)
    for n, c in enumerate(cells):
        tags = dict(zip(KEYS, c))
        if done is not None and len(rows_of(done, tags)):
            if len(rows_of(done, tags)) != reps:
                raise RuntimeError(f"{partial} has another number of replications; delete it")
            continue
        mc = run_configs([dict(spec=spec_of(c), cfg=cfg, fn=label, tags=tags)], reps,
                         base_seed=seed, workers=workers, progress=False)
        mc.to_csv(partial, mode="a", header=not partial.exists(), index=False)
        print(f"   option {n + 1}/{len(cells)} done", flush=True)
    return pd.read_csv(partial)


def summarise(mc, ref):
    rows = []
    for _, r in ref.iterrows():
        g = mc
        for k in KEYS:
            g = g[np.isclose(g[k], r[k])]
        for q in QUANTS:
            m, sd, n = g[q].mean(), g[q].std(), g[q].count()
            t = (m - r[q]) / (sd / np.sqrt(n)) if sd > 0 else 0.0
            rows.append(dict({k: r[k] for k in KEYS}, greek=q, ref=r[q], mean=m, sd=sd,
                             t=t, sig=abs(t) > Z99))
    return pd.DataFrame(rows)


def to_markdown(out, title, intro):
    lines = [f"# {title}\n", intro, "",
             "## Summary by Greek\n",
             "flagged: options where |mean − ref| / (sd/√R) > 2.576. Relative bias and "
             "label noise (sd of one label) are medians of |mean − ref| / |ref| and "
             "sd / |ref| over options with |ref| above 1 % of the largest |ref| for "
             "that Greek.\n",
             "| Greek | flagged | median relative bias | median label noise |",
             "|---|---|---|---|"]
    for q in QUANTS:
        g = out[out.greek == q]
        big = g[g.ref.abs() > 0.01 * g.ref.abs().max()]
        rb = ((big["mean"] - big.ref).abs() / big.ref.abs()).median()
        rn = (big.sd / big.ref.abs()).median()
        lines.append(f"| {q} | {int(g.sig.sum())} / {len(g)} | {rb:.1%} | {rn:.0%} |")
    lines += ["", "## By option\n",
              "| T | σ | r | d | K | Greek | ref | mean | sd (one label) | t |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r['T']:g} | {r.sigma:.0%} | {r.r:.0%} | {r.d:.0%} | {r.K:g} | "
                     f"{r.greek} | {r.ref:+.4f} | {r['mean']:+.4f} | {r.sd:.4f} | "
                     f"{r.t:+.2f}{'†' if r.sig else ''} |")
    return "\n".join(lines) + "\n"


def check(name, cells, cfg, reps, seed, title, intro, workers=None):
    ref = reference(cells, cfg.style)
    t0 = time.time()
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    partial = RESULTS / "raw" / f"multi_check_{name}.partial.csv"
    mc = simulate(cells, cfg, reps, seed, partial, workers)
    print(f"   {name}: {len(mc)} labels in {time.time() - t0:.0f}s", flush=True)
    mc.to_csv(RESULTS / "raw" / f"multi_check_{name}.csv", index=False)
    partial.unlink()
    out = summarise(mc, ref)
    md = to_markdown(out, title, intro)
    (RESULTS / f"multi_check_{name}.md").write_text(md)
    print(md.split("## By option")[0])


def timing():
    cfg = MultiConfig()
    lines = [f"# Seconds per label (one process, default MultiConfig: {len(cfg.groups)} runs "
             "x (100,000 pilot + 100,000 main paths))\n",
             "| T | seconds |", "|---|---|"]
    for T in (0.5, 1.0, 2.0):
        spec = PutSpec(K=40, T=T)
        label(spec, cfg, 0)
        t0 = time.time()
        for s in range(3):
            label(spec, cfg, s + 1)
        lines.append(f"| {T:g} | {(time.time() - t0) / 3:.2f} |")
    (RESULTS / "multi_check_timing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=["european", "american", "timing"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes (default: all cores; ~0.7 GB each)")
    args = ap.parse_args()
    groups = ", ".join("(" + ", ".join(GREEK_LETTERS.get(k, k) for k in g) + ")"
                       for g in MultiConfig().groups)
    common = (f"{args.reps} replications per option; one label = independent runs with "
              f"{groups} dispersed, each with 100,000 pilot and 100,000 main paths. "
              "Widths: α_S (Epanechnikov) and α_σ (uniform) chosen per run on the pilot "
              "paths by the paper's selector (local order ν + 1, targets Gamma and Volga); "
              "α_r = α_d = 0.02 (uniform, placeholders). Not part of the paper.")
    for e in args.experiments:
        if e == "european":
            check("european", european_grid(),
                  MultiConfig(style="european", control_variate=False), args.reps, 7001,
                  "Multivariate ISD, European put vs closed form",
                  common + " No early exercise and no control variate: this checks the "
                  "multivariate regressions alone.", args.workers)
        elif e == "american":
            check("american", american_grid(), MultiConfig(), args.reps, 7002,
                  "Multivariate ISD, Bermudan put vs reference PDE pricer",
                  common + " Exercise rule fitted on the pilot paths rescaled to the "
                  "chosen widths (parameter box 1.3× wider) and also used at t_1 in the "
                  "value-function label. European control variate on. Reference: "
                  "Crank-Nicolson (simgreeks.reference), Greeks in σ, r, d by bumping with "
                  "Richardson extrapolation; in the exercise region the American values "
                  "(price K − S0, Delta −1, other Greeks 0), as the labels.",
                  args.workers)
        else:
            timing()


if __name__ == "__main__":
    main()
