"""Report of Stage 2: the default label (MultiConfig(), D2, hybrid of the
out-of-sample and in-sample labels) on N_STAGE2 Sobol options, one label per
option, against the PDE reference; noise check on 200 options x 30 labels.
Not part of the paper. Runs: experiments/tournament.py, parts stage2,
stage2_ref, stage2_noise.

    python experiments/stage2_report.py --out OUT [--pilot]

--pilot: the 500-label pilot (stage2 and stage2_ref with --limit 500): time per
label, projected SCC hours, and a preliminary bias table; writes
results/stage2/report_pilot.md. Without it: the full report,
results/stage2/report.md, and results/stage2/labels.parquet (inputs, default
label, oos/ins/avg labels, reference as ref_<greek>), noise.parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.tournament import N_STAGE2, NOISE_STEP, REPS, load  # noqa: E402

DEST = ROOT / "results" / "stage2"
GREEKS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi", "delta_r", "delta_d"]
MODES = ("oos", "ins", "avg")
INPUTS = ["id", "S0", "K", "sigma", "r", "d", "T"]
RANGES = """\
Inputs: scrambled Sobol sequence (5 dimensions, seed 20261011; `experiments/tournament.py`, `inputs`), the
tournament's part B ranges; the first 2,500 points are the tournament options.

| input | range |
|---|---|
| S0 | 40 (labels scale exactly with S0 and K) |
| moneyness S0/K | uniform on [0.85, 1.20]; K = S0 / m rounded to 0.01 |
| σ | uniform on [0.10, 0.40], rounded to 0.001 |
| T | uniform on [0.25, 2.00] years, rounded to 0.001 |
| d | uniform on [0, 0.04], rounded to 0.0001 |
| r | 2 % of the points at r = 0; 25 % uniform on [0.0025, 0.0275]; 73 % uniform on [0.03, 0.06]; rounded to 0.0025 |
"""


def regions(o: pd.DataFrame) -> dict:
    m = o.S0 / o.K
    return {
        "all": np.ones(len(o), bool),
        "S0/K < 0.90 (deep ITM)": m < 0.90,
        "S0/K 0.90–1.00 (ITM)": (m >= 0.90) & (m < 1.00),
        "S0/K 1.00–1.10 (OTM)": (m >= 1.00) & (m < 1.10),
        "S0/K ≥ 1.10 (deep OTM)": m >= 1.10,
        "T < 0.75": o["T"] < 0.75,
        "T 0.75–1.25": (o["T"] >= 0.75) & (o["T"] < 1.25),
        "T ≥ 1.25": o["T"] >= 1.25,
        "σ < 0.20": o.sigma < 0.20,
        "σ 0.20–0.30": (o.sigma >= 0.20) & (o.sigma < 0.30),
        "σ ≥ 0.30": o.sigma >= 0.30,
        "0 < r < 1 %": (o.r > 0) & (o.r < 0.01),
        "1 % ≤ r < 2 %": (o.r >= 0.01) & (o.r < 0.02),
        "2 % ≤ r < 3 %": (o.r >= 0.02) & (o.r < 0.03),
        "r ≥ 3 %": o.r >= 0.03,
        "r < 2 % (r > 0)": (o.r > 0) & (o.r < 0.02),
        "d < 1 %": o.d < 0.01,
        "1 % ≤ d < 2 %": (o.d >= 0.01) & (o.d < 0.02),
        "d ≥ 2 %": o.d >= 0.02,
        "d < 2 %": o.d < 0.02,
        "r < 2 % and d < 2 % (r > 0)": (o.r > 0) & (o.r < 0.02) & (o.d < 0.02),
    }


def merge(lab: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Labels with the reference as ref_<greek>; drops the points whose label is
    exact by construction (r0 = 0: European closed form; exercise at t0 in both
    the label and the reference: K - S0, -1, 0)."""
    ref = ref.drop_duplicates("id").set_index("id")
    df = lab.join(ref[GREEKS + ["ex_region"]].add_prefix("ref_"), on="id")
    df["exact"] = df.european_region.astype(bool) | (df.ex_region.astype(bool) & df.ref_ex_region.astype(bool))
    return df


def errors(df: pd.DataFrame, q: str, col: str, thresh: float):
    """(relative errors on the points with |ref| >= thresh, absolute errors on the others)."""
    ref = df[f"ref_{q}"]
    big = ref.abs() >= thresh
    e = df[col] - ref
    return (e / ref.abs())[big], e[~big]


def stat(e: pd.Series, scale: float = 1.0, digits: int = 2):
    n = len(e)
    if n < 2:
        return "–", np.nan
    b, se = e.mean(), e.std(ddof=1) / np.sqrt(n)
    t = b / se if se > 0 else 0.0
    flag = "**" if abs(t) > 3 else ""
    return f"{flag}{scale * b:+.{digits}f} ± {scale * se:.{digits}f} ({t:+.1f}){flag}", t


def fmt_abs(x: float) -> str:
    return f"{x:.1e}" if x != 0 else "0"


def bias_tables(df: pd.DataFrame, thresholds: dict, L: list, flags: list, col_of=lambda q: q):
    w = L.append
    regs = regions(df)
    scales = {q: df[f"ref_{q}"].abs().mean() for q in GREEKS}
    for q in GREEKS:
        w(f"### {q}\n")
        w(f"Relative where |PDE| ≥ {fmt_abs(thresholds[q])} (1 % of the largest); absolute otherwise. "
          f"Median rel. error: over the same points as the relative bias. Scaled bias: mean of "
          f"(MC − PDE) over all points / mean |PDE| ({fmt_abs(scales[q])}), robust where the Greek "
          f"crosses zero.\n")
        w("| region | n rel. | rel. bias % ± se (t) | median rel. error % | n abs. | abs. bias ± se (t) | "
          "scaled bias % ± se (t) |")
        w("|---|---|---|---|---|---|---|")
        for name, sel in regs.items():
            d = df[np.asarray(sel)]
            er, ea = errors(d, q, col_of(q), thresholds[q])
            sr, tr = stat(er, 100, 3)
            if len(ea) >= 2:
                b, se = ea.mean(), ea.std(ddof=1) / np.sqrt(len(ea))
                ta = b / se if se > 0 else 0.0
                sa = f"{b:+.2e} ± {se:.1e} ({ta:+.1f})"
                if abs(ta) > 3:
                    sa = f"**{sa}**"
            else:
                sa, ta = "–", np.nan
            for kind, t in (("rel.", tr), ("abs.", ta)):
                if np.isfinite(t) and abs(t) > 3:
                    flags.append((q, name, kind, t))
            med = f"{100 * er.median():+.3f}" if len(er) else "–"
            sc, ts = stat((d[col_of(q)] - d[f"ref_{q}"]) / scales[q], 100, 3)
            if np.isfinite(ts) and abs(ts) > 3:
                flags.append((q, name, "scaled", ts))
            w(f"| {name} | {len(er)} | {sr} | {med} | {len(ea)} | {sa} | {sc} |")
        w("")


def noise_tables(df: pd.DataFrame, noise: pd.DataFrame | None, thresholds: dict, L: list):
    """Relative noise of one label by region: (a) sd of the relative errors over
    the stage2 points (noise plus the spread of the bias over the points), (b)
    the 30-label check: root mean square over the points of the per-point sd."""
    w = L.append
    regs = regions(df)
    nregs = regions(noise.drop_duplicates("id")) if noise is not None else None
    w("Relative noise of one label, % of |PDE| (points with |PDE| ≥ 1 % of the largest). (a) sd of "
      "(MC − PDE)/|PDE| over the points: one label per point, so it holds the noise plus the spread "
      "of the bias between points; (b) from the 30-label check: √(mean over its points of the "
      "per-point variance)/|PDE|; ratio (a)/(b) ≈ 1 when the bias varies little between points. "
      "Cells: (a) · (b) [points in (b)] · ratio.\n")
    w("| region | " + " | ".join(GREEKS) + " |")
    w("|---|" + "---|" * len(GREEKS))
    per_point = None
    if noise is not None:
        g = noise.groupby("id")
        per_point = pd.DataFrame({q: g[q].var(ddof=1) for q in GREEKS})
        per_point = per_point.join(noise.drop_duplicates("id").set_index("id")[[f"ref_{q}" for q in GREEKS]
                                                                              + ["exact"]])
    for name, sel in regs.items():
        d = df[np.asarray(sel)]
        cells = []
        for q in GREEKS:
            er, _ = errors(d, q, q, thresholds[q])
            a = 100 * er.std(ddof=1) if len(er) > 1 else np.nan
            cell = f"{a:.2f}" if np.isfinite(a) else "–"
            if per_point is not None:
                ids = set(noise.drop_duplicates("id").id[np.asarray(nregs[name])])
                pp = per_point[per_point.index.isin(ids) & ~per_point.exact.astype(bool)]
                pp = pp[pp[f"ref_{q}"].abs() >= thresholds[q]]
                if len(pp):
                    b = 100 * np.sqrt((pp[q] / pp[f"ref_{q}"] ** 2).mean())
                    cell += f" · {b:.2f} [{len(pp)}] · {a / b:.2f}"
                else:
                    cell += " · –"
            cells.append(cell)
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")


def timing(lab: pd.DataFrame, ref: pd.DataFrame, L: list, ntasks=(100, 200)):
    w = L.append
    tl, tr = lab.seconds, ref.seconds
    n_lab, n_noise = N_STAGE2, len(range(0, N_STAGE2, NOISE_STEP)) * REPS["stage2_noise"]
    w("| | labels | mean s | median s | p95 s | max s |")
    w("|---|---|---|---|---|---|")
    w(f"| label (D2-hybrid) | {len(tl)} | {tl.mean():.1f} | {tl.median():.1f} | {tl.quantile(.95):.1f} | {tl.max():.1f} |")
    w(f"| PDE reference | {len(tr)} | {tr.mean():.1f} | {tr.median():.1f} | {tr.quantile(.95):.1f} | {tr.max():.1f} |")
    w("")
    parts = [("stage2", n_lab, tl.mean()), ("stage2_ref", N_STAGE2, tr.mean()), ("stage2_noise", n_noise, tl.mean())]
    tot = sum(n * s for _, n, s in parts) / 3600
    w("Projected SCC time (1 core per task, mean seconds above):\n")
    w("| part | jobs | CPU hours | " + " | ".join(f"hours per task, {k} tasks" for k in ntasks) + " |")
    w("|---|---|---|" + "---|" * len(ntasks))
    for name, n, s in parts:
        w(f"| {name} | {n} | {n * s / 3600:.1f} | " + " | ".join(f"{n * s / 3600 / k:.2f}" for k in ntasks) + " |")
    w(f"| total | | {tot:.1f} | " + " | ".join(f"{tot / k:.2f}" for k in ntasks) + " |")
    w("")
    w(f"The pilot's {len(tl)} labels count towards stage2 (same seeds); the remaining "
      f"{N_STAGE2 - len(tl)} labels take {(N_STAGE2 - len(tl)) * tl.mean() / 3600:.1f} CPU hours.\n")


def mode_table(df: pd.DataFrame, thresholds: dict, L: list):
    w = L.append
    w("Relative bias % ± se (t), all non-exact points with |PDE| ≥ 1 % of the largest. **default** is the "
      "label in the plain columns (`default_source`). Median rel. error: same points. Scaled bias: mean of "
      "(MC − PDE) over all non-exact points / mean |PDE|, ± se (t); it is not dominated by the points where "
      "the Greek is near zero (Volga, Vanna, delta_r, delta_d cross zero in the input range).\n")
    w("| Greek | default (source) | oos | ins | avg | default: median rel. error % | default: scaled bias % |")
    w("|---|---|---|---|---|---|---|")
    src = df.default_source.dropna().iloc[0] if df.default_source.notna().any() else ""
    oos_set = src.split("oos:")[1].split() if "oos:" in src else []
    for q in GREEKS:
        cells = [stat(errors(df, q, q, thresholds[q])[0], 100, 3)[0] + f" ({'oos' if q in oos_set else 'avg'})"]
        cells += [stat(errors(df, q, f"{q}_{m}", thresholds[q])[0], 100, 3)[0] for m in MODES]
        cells.append(f"{100 * errors(df, q, q, thresholds[q])[0].median():+.3f}")
        cells.append(stat((df[q] - df[f"ref_{q}"]) / df[f"ref_{q}"].abs().mean(), 100, 3)[0])
        w(f"| {q} | " + " | ".join(cells) + " |")
    w("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    lab, ref = load("stage2", a.out), load("stage2_ref", a.out)
    df = merge(lab, ref)
    use = df[~df.exact]
    thresholds = {q: 0.01 * df[f"ref_{q}"].abs().max() for q in GREEKS}
    DEST.mkdir(parents=True, exist_ok=True)
    noise = None
    if not a.pilot:
        try:
            noise = merge(load("stage2_noise", a.out), ref)
        except SystemExit:
            noise = None

    L, flags = [], []
    w = L.append
    title = "Stage 2 pilot" if a.pilot else "Stage 2"
    w(f"# {title}: default label (D2, hybrid) against the PDE reference\n")
    w("Not part of the paper. `experiments/tournament.py --part stage2 / stage2_ref / stage2_noise`, "
      "`experiments/stage2_report.py`.\n")
    w(RANGES)
    w("**Label.** `MultiConfig()`: design D2, 100,000 main + 100,000 pilot paths per run, α_S and α_σ by "
      "the selector, α_r = min(0.03, r0), α_d = min(0.03, d0), exact European labels at r0 ≤ 0 and d0 ≥ 0. "
      "Default label (plain columns): the average of the out-of-sample and in-sample labels for price, Δ, Γ, "
      "Θ, Vega, Rho, Phi; the out-of-sample label for Volga, Vanna, delta_r, delta_d. The oos, ins and avg "
      "labels are stored as `<greek>_oos`, `_ins`, `_avg`; `default_source` names the source of each "
      "plain column. One label per point, seeds `job_seed(8006, {id}, 0)`; noise check seeds "
      "`job_seed(8007, {id}, rep)`. Reference: `reference.label_reference` (Crank–Nicolson).\n")
    n_ex = int(df.exact.sum())
    w(f"Points: {len(df)} labelled, {len(ref.drop_duplicates('id'))} references; {n_ex} left out of the "
      f"statistics because the label is exact by construction ({int(df.european_region.sum())} at r0 = 0; "
      f"{n_ex - int(df.european_region.sum())} exercised at t0 in label and reference). "
      f"Exercise-region disagreements (label ≠ reference): "
      f"{int((df.ex_region.astype(bool) != df.ref_ex_region.astype(bool)).sum())}.\n")
    r0 = df[df.european_region.astype(bool)]
    if len(r0):
        w("Check at r0 = 0 (exact European labels against the PDE): largest |MC − PDE| " + ", ".join(
            f"{q} {fmt_abs((r0[q] - r0[f'ref_{q}']).abs().max())}" for q in GREEKS) + ".\n")
    w("Bias = mean of (MC − PDE) over the points; se = sd / √n over the points (it holds the label noise "
      "and the spread of the bias between points); t = bias / se. **Bold**: |t| > 3.\n")

    if a.pilot:
        w("## Time per label and projected SCC hours\n")
        timing(lab, ref, L)
    w("## Default label against oos, ins and avg (all points)\n")
    mode_table(use, thresholds, L)
    w("## Bias by region (default label)\n")
    bias_tables(use, thresholds, L, flags)
    w("## Noise by region (default label)\n")
    noise_tables(use, noise, thresholds, L)
    if noise is not None:
        w("## Noise check: per-point bias over 30 labels\n")
        w("Per point: bias = mean of 30 labels − PDE, t = bias / (sd/√30). Cells: points with t > 3 / t < −3, "
          "of the non-exact points with |PDE| ≥ 1 % of the largest.\n")
        w("| Greek | points | t > 3 | t < −3 | mean t |")
        w("|---|---|---|---|---|")
        nz = noise[~noise.exact]
        for q in GREEKS:
            g = nz.groupby("id")
            b = g[q].mean() - g[f"ref_{q}"].first()
            se = g[q].std(ddof=1) / np.sqrt(g.size())
            big = g[f"ref_{q}"].first().abs() >= thresholds[q]
            t = (b / se)[big & (se > 0)]
            w(f"| {q} | {len(t)} | {int((t > 3).sum())} | {int((t < -3).sum())} | {t.mean():+.2f} |")
        w("")
    w("## Regions with |t| > 3 (default label)\n")
    if flags:
        w("| Greek | region | kind | t |")
        w("|---|---|---|---|")
        for q, name, kind, t in flags:
            w(f"| {q} | {name} | {kind} | {t:+.1f} |")
    else:
        w("None.")
    w("")
    name = "report_pilot.md" if a.pilot else "report.md"
    (DEST / name).write_text("\n".join(L) + "\n")
    keep = (INPUTS + GREEKS + [f"{q}_{m}" for q in GREEKS for m in MODES] + [f"ref_{q}" for q in GREEKS]
            + ["default_source", "ex_region", "ex_region_ins", "ref_ex_region", "european_region", "alpha_S",
               "alpha_sigma", "alpha_r", "alpha_d", "seed_entropy", "seconds"])
    df.sort_values("id")[[c for c in keep if c in df]].to_parquet(
        DEST / ("labels_pilot.parquet" if a.pilot else "labels.parquet"), index=False)
    if noise is not None:
        noise.sort_values(["id", "rep"])[[c for c in keep + ["rep"] if c in noise]].to_parquet(
            DEST / "noise.parquet", index=False)
    print("\n".join(L[: 60]))
    print(f"... wrote {DEST / name}")


if __name__ == "__main__":
    main()
