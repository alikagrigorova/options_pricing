"""Report of Stage A of the American (daily-exercise) labels, puts and calls
(experiments/tournament.py, parts amd_check, amd_check_ref, amd_random,
amd_random_ref). Not part of the paper.

    python experiments/american_daily_report.py --out OUT

Writes results/american_daily/report.md and raw/ (labels and references as
Parquet). Bias is against the continuous American PDE (the target) and against
the daily-Bermudan PDE (the labels' own exercise dates); their difference is the
discreteness gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.tournament import AMD_VARIANTS, load  # noqa: E402

DEST = ROOT / "results" / "american_daily"
GREEKS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi", "delta_r", "delta_d"]
MODES = ("oos", "ins", "avg")
KINDS = ("put", "call")
REFS = {"American": "", "daily Bermudan": "berm_"}
VNAME = {"D2-daily": "value function at t1", "D2-daily-vf50": "value function at 1/50 year"}


def ref_frame(ref: pd.DataFrame, prefix: str) -> pd.DataFrame:
    r = ref.drop_duplicates("id").set_index("id")
    out = pd.DataFrame({q: r[prefix + q] for q in GREEKS})
    out["ex_region"] = r[prefix + "ex_region"].astype(bool)
    return out


def per_option(lab: pd.DataFrame, ref: pd.DataFrame, col_of=lambda q: q) -> pd.DataFrame:
    """Per option and Greek: n, ref, bias, se, t, sd and relative versions."""
    rows = []
    for i, g in lab.groupby("id"):
        for q in GREEKS:
            x = g[col_of(q)].dropna()
            r = float(ref.loc[i, q])
            n, m, sd = len(x), x.mean(), x.std(ddof=1)
            b = m - r
            tol = 1e-9 * max(1.0, abs(r))
            if sd <= tol:                      # exact label (European region, exercise region)
                sd = 0.0
            se = sd / np.sqrt(n)
            t = b / se if se > 0 else (0.0 if abs(b) <= tol else np.sign(b) * np.inf)
            rows.append(dict(id=i, greek=q, n=n, ref=r, mean=m, bias=b, se=se, t=t, sd=sd))
    df = pd.DataFrame(rows)
    big = df.groupby("greek").ref.transform(lambda s: s.abs().max())
    df["use_rel"] = df.ref.abs() > 0.01 * big
    df["rel_bias"] = np.where(df.use_rel, df.bias / df.ref.abs(), np.nan)
    df["rel_noise"] = np.where(df.use_rel, df.sd / df.ref.abs(), np.nan)
    return df


def summary_cells(po: pd.DataFrame, q: str) -> str:
    d = po[po.greek == q]
    fin = d[np.isfinite(d.t)]
    t3p, t3n = int((d.t > 3).sum()), int((d.t < -3).sum())
    scaled = d.bias.sum() / d.ref.abs().sum()
    scaled_se = np.sqrt((d.se ** 2).sum()) / d.ref.abs().sum()
    st = scaled / scaled_se if scaled_se > 0 else 0.0
    return (f"{t3p}/{t3n} · {100 * d.rel_bias.median():+.2f}% · {fin.t.mean():+.2f} · "
            f"{100 * scaled:+.2f}% ({st:+.1f}) · {100 * d.rel_noise.median():.1f}%")


def point_errors(df: pd.DataFrame, q: str, col: str, ref_col: str):
    """Relative errors where |ref| >= 1 % of the largest, absolute otherwise; scaled."""
    r = df[ref_col]
    big = r.abs() >= 0.01 * r.abs().max()
    e = df[col] - r
    return (e / r.abs())[big], e[~big], e.sum() / r.abs().sum(), e.std(ddof=1) * np.sqrt(len(e)) / r.abs().sum()


def stat(e: pd.Series, scale=100.0, nd=3) -> str:
    if len(e) < 2:
        return "–"
    b, se = e.mean(), e.std(ddof=1) / np.sqrt(len(e))
    t = b / se if se > 0 else 0.0
    s = f"{scale * b:+.{nd}f} ± {scale * se:.{nd}f} ({t:+.1f})"
    return f"**{s}**" if abs(t) > 3 else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "raw").mkdir(exist_ok=True)
    L = []
    w = L.append
    w("# American options with daily exercise, puts and calls: Stage A validation\n")
    w("Not part of the paper. `experiments/tournament.py --part amd_check / amd_check_ref / amd_random / "
      "amd_random_ref`, `experiments/american_daily_report.py`.\n")
    w("**Labels.** Design D2 hybrid (`MultiConfig()`): average of the out-of-sample and in-sample labels for "
      "price, Δ, Γ, Θ, Vega, Rho, Phi; out-of-sample for Volga, Vanna, ∂Δ/∂r, ∂Δ/∂d; 100,000 main + 100,000 "
      "pilot paths per run; exercise every business day, J = round(252 T) dates. Calls: payoff max(S − K, 0), "
      "European call as control variate, exact European label at d0 ≤ 0 ≤ r0, exercise-region values S0 − K, "
      "Δ = +1. Two variants, paired seeds: **" + "**, **".join(f"{v}** ({VNAME[v]})" for v in AMD_VARIANTS)
      + ". The second (`MultiConfig(vf_time=0.02)`) takes the value-function label at the exercise date nearest "
      "1/50 year (the 5th daily date) instead of t1, with the exercise rule applied at the dates before it; at "
      "50 dates a year it is the current label exactly.\n")
    w("**References.** Crank–Nicolson PDE (`reference.label_reference`): *American* = exercise at every time "
      "step, Richardson-extrapolated over 20 and 40 steps a day (continuous exercise, the target); *daily "
      "Bermudan* = exercise on the labels' dates (10 steps a day). Greeks in σ, r, d by bumping.\n")

    # ---------------- check set ----------------
    lab = load("amd_check", a.out)
    ref = load("amd_check_ref", a.out)
    opts = ref.drop_duplicates("id").set_index("id")
    lab.to_parquet(DEST / "raw" / "check_labels.parquet", index=False)
    ref.to_parquet(DEST / "raw" / "check_reference.parquet", index=False)
    refs = {name: ref_frame(ref, p) for name, p in REFS.items()}

    w("## Time and memory\n")
    w("| kind | T | labels | seconds per label (mean, by variant) | p95 |")
    w("|---|---|---|---|---|")
    for k in KINDS:
        for T, g in lab[lab.kind == k].groupby("T"):
            per = ", ".join(f"{v} {g[g.variant == v].seconds.mean():.0f}" for v in AMD_VARIANTS)
            w(f"| {k} | {T:g} | {len(g)} | {per} | {g.seconds.quantile(.95):.0f} |")
    t2 = lab[lab["T"] == 2.0]
    w(f"\nPeak memory (process RSS, maximum over the tasks running T = 2 labels): {t2.rss_mb.max():.0f} MB. "
      f"PDE reference per option: {ref.seconds.mean():.0f} s mean, {ref.seconds.max():.0f} s max.\n")

    w("## Check sets: summary per Greek (default label)\n")
    w("35 put options (the production check's) and 35 call options (T 0.5/1/2 × σ 10/20/40 % × K 36/40/44 at "
      "r = 6 %, d = 4 %, plus r = 0, d = 4 %; r = 6 %, d = 0 (exact European); r = 6 %, d = 2 %), 100 labels each, "
      "seeds `job_seed(8010, {id}, rep)`. Cells: options with t > 3 / t < −3 · median signed relative bias · "
      "mean t · scaled bias Σ bias / Σ |ref| (t) · median noise of one label.\n")
    POS = {}
    for k in KINDS:
        ids = opts.index[opts.kind == k]
        w(f"### {k}s\n")
        cols = [(v, rn) for v in AMD_VARIANTS for rn in REFS]
        w("| Greek | " + " | ".join(f"{v} vs {rn}" for v, rn in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for v in AMD_VARIANTS:
            for rn in REFS:
                POS[(k, v, rn)] = per_option(lab[(lab.variant == v) & lab.id.isin(ids)], refs[rn])
        for q in GREEKS:
            w(f"| {q} | " + " | ".join(summary_cells(POS[(k, v, rn)], q) for v, rn in cols) + " |")
        w("")

    w("## Check sets: out-of-sample, in-sample and averaged labels (vs American)\n")
    w("Cells as above.\n")
    for k in KINDS:
        ids = opts.index[opts.kind == k]
        for v in AMD_VARIANTS:
            w(f"### {k}s, {v}\n")
            w("| Greek | default | oos | ins | avg |")
            w("|---|---|---|---|---|")
            g = lab[(lab.variant == v) & lab.id.isin(ids)]
            pm = {m: per_option(g, refs["American"], lambda q, m=m: f"{q}_{m}") for m in MODES}
            for q in GREEKS:
                w(f"| {q} | " + " | ".join([summary_cells(POS[(k, v, "American")], q)]
                                          + [summary_cells(pm[m], q) for m in MODES]) + " |")
            w("")

    w("## Daily Bermudan vs American gap per check option\n")
    w("Gap = American − daily Bermudan (PDE). Relative to |American| in brackets; 'in se' = gap / se of the "
      "mean of 100 labels (default label, first variant): above about 1 the 100-label check can see it.\n")
    w("| id | kind | T | σ | r | d | K | price gap | Δ gap | Γ gap | Θ gap | Rho gap | Phi gap | price gap in se |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    A, B = refs["American"], refs["daily Bermudan"]
    po0 = pd.concat([POS[(k, AMD_VARIANTS[0], "American")] for k in KINDS]).set_index(["id", "greek"])
    for i, o in opts.iterrows():
        cells = []
        for q in ("price", "delta", "gamma", "theta", "rho", "phi"):
            gap = A.loc[i, q] - B.loc[i, q]
            rel = f" [{100 * gap / abs(A.loc[i, q]):+.3f}%]" if abs(A.loc[i, q]) > 1e-12 else ""
            cells.append(f"{gap:+.2e}{rel}")
        se = po0.loc[(i, "price"), "se"]
        ins = f"{(A.loc[i, 'price'] - B.loc[i, 'price']) / se:+.1f}" if se > 0 else "exact"
        w(f"| {i} | {o.kind} | {o['T']:g} | {o.sigma:.0%} | {o.r:.0%} | {o.d:.0%} | {o.K:g} | "
          + " | ".join(cells) + f" | {ins} |")
    w("")

    # ---------------- random points ----------------
    try:
        rl, rr = load("amd_random", a.out), load("amd_random_ref", a.out)
    except SystemExit:
        rl = None
    if rl is not None:
        rl.to_parquet(DEST / "raw" / "random_labels.parquet", index=False)
        rr.to_parquet(DEST / "raw" / "random_reference.parquet", index=False)
        R = rr.drop_duplicates("id").set_index("id")
        df = rl.join(R[[c for c in R.columns if c in GREEKS or c.startswith("berm_") or c == "ex_region"]]
                     .add_prefix("ref_"), on="id")
        df["exact"] = df.european_region.astype(bool) | (df.ex_region.astype(bool) & df.ref_ex_region.astype(bool))
        w("## 1,000 random points per type (one label each)\n")
        w("Points 20,000–20,999 (puts) and 21,000–21,999 (calls) of the scrambled Sobol sequence of the "
          "tournament / Stage 2 (S0/K 0.85–1.20, σ 10–40 %, T 0.25–2, d 0–4 %, r 0–6 % with 2 % at 0 and 25 % in "
          "(0, 3 %)); seeds `job_seed(8011, {id}, 0)`. Points with exact labels (no early exercise, or exercise "
          "at t0 in label and reference) left out. Bias = mean of (label − PDE)/|PDE| over the points with |PDE| "
          "≥ 1 % of the largest (± se, t); scaled = Σ (label − PDE) / Σ |PDE| over all points (t); typical error "
          "= median |relative error|.\n")
        for k in KINDS:
            d0 = df[(df.kind == k) & ~df.exact]
            w(f"### {k}s ({d0.id.nunique()} points with non-exact labels, "
              f"{int(df[(df.kind == k) & df.exact].id.nunique())} exact)\n")
            cols = [(v, rn) for v in AMD_VARIANTS for rn in REFS]
            w("| Greek | " + " | ".join(f"{v} vs {rn}: bias % (t) · scaled % (t) · typical" for v, rn in cols) + " |")
            w("|---|" + "---|" * len(cols))
            for q in GREEKS:
                cells = []
                for v, rn in cols:
                    d = d0[d0.variant == v]
                    er, _, sc, scse = point_errors(d, q, q, f"ref_{REFS[rn]}{q}")
                    cells.append(f"{stat(er)} · {100 * sc:+.3f} ({sc / scse:+.1f}) · {100 * er.abs().median():.2f}%")
                w(f"| {q} | " + " | ".join(cells) + " |")
            w("")
            for v in AMD_VARIANTS:
                d = d0[d0.variant == v]
                w(f"{k}s, {v}, vs American — oos / ins / avg relative bias % (t):\n")
                w("| Greek | default | oos | ins | avg |")
                w("|---|---|---|---|---|")
                for q in GREEKS:
                    w(f"| {q} | " + " | ".join(stat(point_errors(d, q, c, f"ref_{q}")[0])
                                              for c in [q] + [f"{q}_{m}" for m in MODES]) + " |")
                w("")

    w("## Every check option and Greek (default label, vs American)\n")
    w("Cells: bias ± se (t) [relative bias %, relative noise %], per variant.\n")
    w("| id | kind | T | σ | r | d | K | Greek | American | daily Bermudan | " + " | ".join(AMD_VARIANTS) + " |")
    w("|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(AMD_VARIANTS))
    P = {v: pd.concat([POS[(k, v, "American")] for k in KINDS]).set_index(["id", "greek"]) for v in AMD_VARIANTS}
    for i, o in opts.iterrows():
        for q in GREEKS:
            cells = []
            for v in AMD_VARIANTS:
                r = P[v].loc[(i, q)]
                rb = f"{100 * r.rel_bias:+.2f}" if np.isfinite(r.rel_bias) else "–"
                rn = f"{100 * r.rel_noise:.1f}" if np.isfinite(r.rel_noise) else "–"
                cells.append(f"{r.bias:+.4f} ± {r.se:.4f} ({r.t:+.1f}) [{rb}, {rn}]")
            w(f"| {i} | {o.kind} | {o['T']:g} | {o.sigma:.0%} | {o.r:.0%} | {o.d:.0%} | {o.K:g} | {q} | "
              f"{A.loc[i, q]:+.4f} | {B.loc[i, q]:+.4f} | " + " | ".join(cells) + " |")
    (DEST / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[: L.index("## Every check option and Greek (default label, vs American)\n")]))


if __name__ == "__main__":
    main()
