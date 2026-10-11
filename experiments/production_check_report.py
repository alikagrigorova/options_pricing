"""Report of the production validation check (experiments/tournament.py, part
production_check): the label generator with its production settings,
MultiConfig() = design D2, out-of-sample exercise rule, 100,000 main + 100,000
pilot paths per run, r and d windows min(0.03, r0), exact European labels at
r0 <= 0, d0 >= 0; 100 replications of each of the 33 American check options plus
two supplementary options with r0 = d0 = 1 % (K = 40, 44). Not part of the paper.

    python experiments/production_check_report.py --out OUT [--tournament DIR]

OUT is the tournament.py output directory with production_check/. Writes
results/production_check/report.md and raw/labels.csv, raw/reference.csv.
--tournament: a tournament output directory with american/ and
reference_american.parquet, for the side-by-side with the tournament's D2-oos.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.tournament import EXTRA_CELLS, options, spec_of  # noqa: E402
from simgreeks.reference import label_reference, put_fd_greeks  # noqa: E402

GREEKS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi"]
KEYS = ["T", "sigma", "r", "d", "K"]
DEST = ROOT / "results" / "production_check"
N_MAIN = 33                       # options 0..32: the American check; 33, 34: supplementary


def load_labels(out: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(out) / "production_check" / "task_*" / "chunk_*.parquet")))
    if not files:
        raise SystemExit(f"no production_check output in {out}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def references(opts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, o in opts.iterrows():
        ref = label_reference(spec_of(o))
        rows.append(dict(id=int(o.id), **{k: o[k] for k in KEYS},
                         **{q: ref[q] for q in GREEKS}, ex_region=ref["ex_region"],
                         rho_fwd=ref["rho_fwd"], phi_fwd=ref["phi_fwd"]))
    return pd.DataFrame(rows).set_index("id")


def per_option(lab: pd.DataFrame, ref: pd.DataFrame, key="id") -> pd.DataFrame:
    """Per option and Greek: n, bias, se, t, noise (absolute) and relative versions."""
    rows = []
    for i, g in lab.groupby(key):
        for q in GREEKS:
            x = g[q].dropna()
            r = float(ref.loc[i, q])
            n, m, sd = len(x), x.mean(), x.std(ddof=1)
            b = m - r
            exact = sd <= 1e-9 * max(1.0, abs(r))      # exact labels (European region): rounding only
            if exact:
                sd = 0.0
            se = sd / np.sqrt(n)
            t = b / se if se > 0 else (0.0 if abs(b) <= 1e-9 * max(1.0, abs(r)) else np.inf)
            rows.append(dict(id=i, greek=q, n=n, ref=r, mean=m, bias=b, se=se, t=t, sd=sd))
    df = pd.DataFrame(rows)
    big = df.groupby("greek").ref.transform(lambda s: s.abs().max())
    df["use_rel"] = df.ref.abs() > 0.01 * big          # relative measures only where ref is not ~0
    df["rel_bias"] = np.where(df.use_rel, df.bias / df.ref.abs(), np.nan)
    df["rel_se"] = np.where(df.use_rel, df.se / df.ref.abs(), np.nan)
    df["rel_noise"] = np.where(df.use_rel, df.sd / df.ref.abs(), np.nan)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in GREEKS:
        d = df[df.greek == q]
        fin = d[np.isfinite(d.t)]
        sig = d[d.t.abs() > 3]
        rows.append(dict(
            greek=q, options=len(d), t2=int((d.t.abs() > 2).sum()), t3=int((d.t.abs() > 3).sum()),
            t3_pos=int((sig.t > 0).sum()), t3_neg=int((sig.t < 0).sum()),
            pos=int((fin.bias > 0).sum()), neg=int((fin.bias < 0).sum()),
            med_rel_bias=100 * d.rel_bias.median(), max_abs_rel_bias=100 * d.rel_bias.abs().max(),
            med_rel_noise=100 * d.rel_noise.median(),
            mean_t=fin.t.mean() if len(fin) else np.nan))
    return pd.DataFrame(rows).set_index("greek")


def fmt(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–" if x is None or np.isnan(x) else ("+inf" if x > 0 else "-inf")
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def readme_run(ref_old: pd.DataFrame) -> pd.DataFrame:
    """The earlier 33-option check (results/raw/multi_check_american.csv: alpha_r = alpha_d =
    0.02, no European region, 100 reps), against the reference it was reported with."""
    raw = pd.read_csv(ROOT / "results" / "raw" / "multi_check_american.csv")
    opts = options("american").copy()
    key = {tuple(np.round([o["T"], o.sigma, o.r, o.d, o.K], 6)): int(o.id) for _, o in opts.iterrows()}
    raw["id"] = [key[tuple(np.round(v, 6))] for v in raw[KEYS].values]
    return per_option(raw, ref_old)


def old_reference(opts: pd.DataFrame) -> pd.DataFrame:
    """The reference of results/multi_check_american.md: put_fd_greeks with central bumps,
    exercise values in the exercise region."""
    rows = []
    for _, o in opts.iterrows():
        f = put_fd_greeks(spec_of(o))
        res = {q: float(f[q]) for q in GREEKS}
        if f["ex_region"]:
            res = {q: 0.0 for q in GREEKS}
            res["price"], res["delta"] = o.K - o.S0, -1.0
        rows.append(dict(id=int(o.id), **res))
    return pd.DataFrame(rows).set_index("id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tournament", default=None)
    ap.add_argument("--variant", default="D2-prod", help="D2-prod (report.md) or D3o3-prod (report_D3.md)")
    a = ap.parse_args()
    lab = load_labels(a.out)
    lab = lab[lab.variant == a.variant]
    suffix = "" if a.variant == "D2-prod" else "_D3"
    raw_dir = DEST / f"raw{suffix}"
    opts = options("production_check")
    ref = references(opts)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "rep", "seed_entropy"] + KEYS + GREEKS + ["delta_r", "delta_d", "ex_region",
                                                             "european_region", "alpha_S", "alpha_sigma",
                                                             "alpha_r", "alpha_d", "seconds"]
    lab.sort_values(["id", "rep"])[cols].to_csv(raw_dir / "labels.csv", index=False)
    ref.reset_index().to_csv(raw_dir / "reference.csv", index=False)

    po = per_option(lab, ref)
    main_ = po[po.id < N_MAIN]
    supp = po[po.id >= N_MAIN]
    S = summary(main_)
    # side-by-side
    old = summary(readme_run(old_reference(options("american"))))
    tour = None
    if a.tournament:
        tl = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(a.tournament) / "american" / "*" / "*.parquet"))])
        tl = tl[tl.variant == "D2-oos"]
        tref = pd.read_parquet(Path(a.tournament) / "reference_american.parquet").set_index("id")
        tour = summary(per_option(tl, tref))

    L = []
    w = L.append
    w(f"# Production check: the label generator with production settings ({a.variant})\n")
    w("Not part of the paper. `experiments/tournament.py --part production_check`, "
      "`experiments/production_check_report.py`.\n")
    w("**Settings.** `MultiConfig()`: design D2 (runs (S, σ), (S, r), (S, d); price, Δ, Γ averaged over the "
      "runs), exercise rule fitted out of sample on the pilot paths, 100,000 main + 100,000 pilot paths per "
      "run, α_S and α_σ by the paper's selector (M = ν + 1) on the pilot, α_r = α_d = min(0.03, r0) "
      "(0.03 for every option here; the d0 rule min(0.03, d0) gives the same windows for these options), "
      "exact European labels at r0 ≤ 0, d0 ≥ 0. 100 replications per option; seeds "
      "`job_seed(8004, {id}, rep)`, a fresh SeedSequence per label, so every label is reproducible. "
      "Reference: `reference.label_reference` (Crank–Nicolson, central bumps; European values at r0 ≤ 0, "
      "d0 ≥ 0; exercise values in the exercise region).\n")
    w(f"Options: the 33 of `results/multi_check_american.md` (ids 0–32) and, as a supplement, r0 = d0 = 1 %, "
      f"σ = 20 %, T = 1, K = 40, 44 (ids 33, 34). Labels: {len(lab)}; mean "
      f"{lab.seconds.mean():.1f} s per label.\n")
    w("Bias = mean of the 100 labels − reference; se = sd over the replications / √100; t = bias / se; "
      "noise = sd over the replications (one label). Relative values in % of |reference|, shown only where "
      "|reference| ≥ 1 % of the largest |reference| of that Greek.\n")

    # pass/fail
    w("## Pass criteria\n")
    crit = []
    for q in GREEKS:
        s = S.loc[q]
        worst = max(s.t3_pos, s.t3_neg)
        crit.append((q, worst, "pass" if worst <= 2 else "FAIL"))
    w("1. No Greek has a consistent-sign bias with |t| > 3 on more than about 2 options (33 main options):\n")
    w("| Greek | options with t > 3 | options with t < −3 | verdict |")
    w("|---|---|---|---|")
    for q, worst, v in crit:
        w(f"| {q} | {int(S.loc[q].t3_pos)} | {int(S.loc[q].t3_neg)} | {v} |")
    pb = main_[main_.greek == "price"].rel_bias.abs()
    db = main_[main_.greek == "delta"].rel_bias.abs()
    w("")
    w(f"2. Price bias within about 0.2 %: median {100*pb.median():.3f} %, largest {100*pb.max():.3f} % → "
      f"**{'pass' if pb.max() <= 0.002 else 'FAIL'}**.  ")
    w(f"   Δ bias within about 0.3 %: median {100*db.median():.3f} %, largest {100*db.max():.3f} % → "
      f"**{'pass' if db.max() <= 0.003 else 'FAIL'}**.\n")

    w("## Summary per Greek (33 main options)\n")
    w("| Greek | |t| > 2 | |t| > 3 (+ / −) | bias sign (+ / −) | median signed rel. bias % | largest abs rel. bias % | median rel. noise % |")
    w("|---|---|---|---|---|---|---|")
    for q in GREEKS:
        s = S.loc[q]
        w(f"| {q} | {s.t2} | {s.t3} ({s.t3_pos} / {s.t3_neg}) | {s.pos} / {s.neg} | "
          f"{fmt(s.med_rel_bias, 2, True)} | {fmt(s.max_abs_rel_bias)} | {fmt(s.med_rel_noise, 1)} |")
    w("\nThe 3 options with r0 = d0 = 0 are exact European labels (noise 0, bias 0); the sign counts "
      "include only options with nonzero noise.\n")

    w("## Side by side with the earlier 33-option checks\n")
    w("Same statistics; *README run*: `results/multi_check_american.md` (α_r = α_d = 0.02, no European "
      "region, 100 reps, central-bump reference); *tournament D2-oos*: part A of the tournament (133,333 "
      "paths per run, 50 reps). Cells: |t| > 3 options · median signed rel. bias % · median rel. noise %.\n")
    hdr = "| Greek | production check | README run |" + (" tournament D2-oos |" if tour is not None else "")
    w(hdr)
    w("|---|---|---|" + ("---|" if tour is not None else ""))
    for q in GREEKS:
        row = [q] + [f"{X.loc[q].t3} · {fmt(X.loc[q].med_rel_bias, 2, True)} · {fmt(X.loc[q].med_rel_noise, 1)}"
                     for X in ([S, old] + ([tour] if tour is not None else []))]
        w("| " + " | ".join(row) + " |")

    w("\n## Volga and Vanna\n")
    for q in ("volga", "vanna"):
        d = main_[(main_.greek == q) & np.isfinite(main_.t) & (main_.sd > 0)]
        z = d.t.values
        w(f"- **{q}**: {int((z > 0).sum())} positive / {int((z < 0).sum())} negative biases over {len(z)} "
          f"options; mean t = {z.mean():+.2f} (pure noise: 0 ± {1/np.sqrt(len(z)):.2f}); "
          f"sd of t = {z.std(ddof=1):.2f} (pure noise: about 1); |t| > 3 on {int((np.abs(z) > 3).sum())}; "
          f"median signed rel. bias {100*d.rel_bias.median():+.1f} %.")
    w("")

    w("## Supplement: r0 = d0 = 1 % (K = 40, 44), the known small-rate case\n")
    w("| K | Greek | ref | bias ± se | rel. bias % | t | rel. noise % |")
    w("|---|---|---|---|---|---|---|")
    for _, r in supp.iterrows():
        K = int(ref.loc[r.id, "K"])
        w(f"| {K} | {r.greek} | {r.ref:+.4f} | {r.bias:+.4f} ± {r.se:.4f} | {fmt(100*r.rel_bias, 2, True)} | "
          f"{fmt(r.t, 1, True)} | {fmt(100*r.rel_noise, 1)} |")
    w("")

    w("## Price and Δ: the out-of-sample lower bound\n")
    for q in ("price", "delta"):
        d = main_[(main_.greek == q) & (main_.sd > 0)]
        w(f"- **{q}**: {int((d.bias < 0).sum())} of {len(d)} biases negative; median {100*d.rel_bias.median():+.3f} %, "
          f"mean {100*d.rel_bias.mean():+.3f} %, range {100*d.rel_bias.min():+.3f} to {100*d.rel_bias.max():+.3f} %; "
          f"|t| > 3 on {int((d.t.abs() > 3).sum())}.")
    w("")

    w("## Every option and Greek\n")
    w("| id | T | σ | r | d | K | Greek | ref | bias ± se | rel. bias % | t | rel. noise % |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in po.iterrows():
        o = ref.loc[r.id]
        w(f"| {r.id} | {o['T']:g} | {o.sigma:.0%} | {o.r:.0%} | {o.d:.0%} | {int(o.K)} | {r.greek} | "
          f"{r.ref:+.4f} | {r.bias:+.4f} ± {r.se:.4f} | {fmt(100*r.rel_bias, 2, True)} | {fmt(r.t, 1, True)} | "
          f"{fmt(100*r.rel_noise, 1)} |")
    (DEST / f"report{suffix}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
