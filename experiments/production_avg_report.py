"""Report of the D2-avg production check (experiments/tournament.py, part
production_avg): the production label generator (MultiConfig(), design D2) that
also returns the in-sample label on the same main paths, and the average of
the out-of-sample (oos) and in-sample (ins) labels. Same 35 options as the
production check (33 American check options + r0 = d0 = 1 %, K = 40, 44),
100 replications, seeds job_seed(8005, {id}, rep). Not part of the paper.

    python experiments/production_avg_report.py --out OUT

Writes results/production_check/report_avg.md and raw_avg/labels.csv,
raw_avg/reference.csv. Compares with D2-prod (results/production_check/raw/)
and the README run (results/raw/multi_check_american.csv).
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

from experiments.production_check_report import (GREEKS, KEYS, N_MAIN, fmt, old_reference,  # noqa: E402
                                                 per_option, readme_run, references, summary)
from experiments.tournament import options  # noqa: E402

DEST = ROOT / "results" / "production_check"
MODES = {"oos": "", "ins": "_ins", "avg": "_avg"}


def mode_frame(lab: pd.DataFrame, mode: str) -> pd.DataFrame:
    """The labels of one mode with the plain Greek column names."""
    sfx = MODES[mode]
    df = lab[["id", "rep"]].copy()
    for q in GREEKS:
        df[q] = lab[q + sfx]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.out) / "production_avg" / "task_*" / "chunk_*.parquet")))
    lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lab = lab[lab.variant == "D2-avg"]
    opts = options("production_avg")
    ref = references(opts)
    (DEST / "raw_avg").mkdir(parents=True, exist_ok=True)
    cols = (["id", "rep", "seed_entropy"] + KEYS + [q + s for s in MODES.values() for q in GREEKS]
            + ["ex_region", "ex_region_ins", "european_region", "alpha_S", "alpha_sigma", "alpha_r",
               "alpha_d", "seconds"])
    lab.sort_values(["id", "rep"])[[c for c in cols if c in lab]].to_csv(DEST / "raw_avg" / "labels.csv", index=False)
    ref.reset_index().to_csv(DEST / "raw_avg" / "reference.csv", index=False)

    po = {m: per_option(mode_frame(lab, m), ref) for m in MODES}
    main_ = {m: d[d.id < N_MAIN] for m, d in po.items()}
    S = {m: summary(main_[m]) for m in MODES}
    prod_lab = pd.read_csv(DEST / "raw" / "labels.csv")
    prod_ref = pd.read_csv(DEST / "raw" / "reference.csv").set_index("id")
    S["D2-prod"] = summary(per_option(prod_lab[prod_lab.id < N_MAIN], prod_ref))
    S["README run"] = summary(readme_run(old_reference(options("american"))))
    prod_sec = prod_lab.seconds.mean()

    L = []
    w = L.append
    w("# Production check of D2-avg: out-of-sample, in-sample and averaged labels\n")
    w("Not part of the paper. `experiments/tournament.py --part production_avg`, "
      "`experiments/production_avg_report.py`.\n")
    w("**Settings.** `MultiConfig(insample_too=True)`: D2-prod (design D2, 100,000 main + 100,000 pilot "
      "paths per run, α_S and α_σ by the selector on the pilot, α_r = min(0.03, r0), α_d = min(0.03, d0), "
      "exact European labels at r0 ≤ 0, d0 ≥ 0) whose runs also refit the exercise rule on their main paths "
      "and apply it to them. **oos**: the D2-prod label (rule fitted on the pilot); **ins**: the in-sample "
      "label on the same main paths and widths; **avg** = (oos + ins) / 2. The oos output equals D2-prod for "
      "the same seeds (checked to 2e-10); this run uses new seeds, `job_seed(8005, {id}, rep)`, 100 "
      "replications per option, BU SCC. Reference: `reference.label_reference`.\n")
    w(f"Labels: {len(lab)}. Seconds per label: D2-avg {lab.seconds.mean():.1f}, D2-prod "
      f"{prod_sec:.1f} (production check, same machines), ratio {lab.seconds.mean() / prod_sec:.2f}.\n")
    w("Bias = mean of 100 labels − reference; se = sd / √100; t = bias / se; noise = sd of one label. "
      "Relative values in % of |reference| where |reference| ≥ 1 % of the largest |reference| of that Greek. "
      "The 3 options with r0 = d0 = 0 are exact European labels (t = 0).\n")

    # pass criteria
    w("## Pass criteria (33 main options)\n")
    w("1. No consistent-sign bias with |t| > 3 on more than about 2 options. Cells: options with t > 3 / t < −3.\n")
    w("| Greek | oos | ins | avg |")
    w("|---|---|---|---|")
    for q in GREEKS:
        cells = []
        for m in MODES:
            s = S[m].loc[q]
            v = "pass" if max(s.t3_pos, s.t3_neg) <= 2 else "**FAIL**"
            cells.append(f"{int(s.t3_pos)} / {int(s.t3_neg)} {v}")
        w(f"| {q} | " + " | ".join(cells) + " |")
    w("\n2. Price within about 0.2 %, Δ within about 0.3 % (median / largest abs relative bias over the options):\n")
    w("| Greek | oos | ins | avg |")
    w("|---|---|---|---|")
    for q, lim in (("price", 0.2), ("delta", 0.3)):
        cells = []
        for m in MODES:
            b = 100 * main_[m][main_[m].greek == q].rel_bias.abs()
            cells.append(f"{b.median():.3f} / {b.max():.3f} % {'pass' if b.max() <= lim else '**FAIL**'}")
        w(f"| {q} | " + " | ".join(cells) + " |")
    w("")

    # summary
    w("## Summary per Greek (33 main options)\n")
    w("Cells: |t| > 2 · |t| > 3 (+/−) · bias signs +/− · median signed rel. bias % · median rel. noise % · mean t.\n")
    w("| Greek | oos | ins | avg |")
    w("|---|---|---|---|")
    for q in GREEKS:
        cells = []
        for m in MODES:
            s = S[m].loc[q]
            cells.append(f"{s.t2} · {s.t3} ({s.t3_pos}/{s.t3_neg}) · {s.pos}/{s.neg} · "
                         f"{fmt(s.med_rel_bias, 2, True)} · {fmt(s.med_rel_noise, 1)} · {fmt(s.mean_t, 2, True)}")
        w(f"| {q} | " + " | ".join(cells) + " |")
    w("")
    w("## Side by side with D2-prod and the README run\n")
    w("Cells: |t| > 3 options · median signed rel. bias % · median rel. noise %. D2-prod: the production check "
      "(other seeds); README run: `results/multi_check_american.md` (α_r = α_d = 0.02, no European region).\n")
    names = ["oos", "ins", "avg", "D2-prod", "README run"]
    w("| Greek | " + " | ".join(names) + " |")
    w("|---|" + "---|" * len(names))
    for q in GREEKS:
        w(f"| {q} | " + " | ".join(f"{S[n].loc[q].t3} · {fmt(S[n].loc[q].med_rel_bias, 2, True)} · "
                                   f"{fmt(S[n].loc[q].med_rel_noise, 1)}" for n in names) + " |")
    w("")

    # correlation of oos and ins errors
    w("## Correlation between the oos and ins errors\n")
    w("Per option: correlation over the 100 replications of (oos − ref) and (ins − ref); median (and range) over "
      "the options with nonzero noise. The noise of avg relative to oos follows: sd(avg)/sd(oos) = "
      "√((1 + 2ρk + k²)/4) with k = sd(ins)/sd(oos).\n")
    w("| Greek | median ρ | range | median sd(avg) / sd(oos) |")
    w("|---|---|---|---|")
    for q in GREEKS:
        rhos, ratios = [], []
        for i, g in lab[lab.id < N_MAIN].groupby("id"):
            a, b, c = g[q], g[q + "_ins"], g[q + "_avg"]
            if a.std() > 1e-9 * max(1.0, abs(a.mean())) and b.std() > 0:
                rhos.append(np.corrcoef(a, b)[0, 1])
                ratios.append(c.std() / a.std())
        if rhos:
            w(f"| {q} | {np.median(rhos):+.2f} | {min(rhos):+.2f} to {max(rhos):+.2f} | {np.median(ratios):.2f} |")
    w("")

    # supplement
    w("## Supplement: r0 = d0 = 1 % (K = 40, 44)\n")
    w("| K | Greek | ref | oos: bias ± se (t) | ins: bias ± se (t) | avg: bias ± se (t) | avg rel. bias % |")
    w("|---|---|---|---|---|---|---|")
    for i in (N_MAIN, N_MAIN + 1):
        for q in GREEKS:
            rows = {m: po[m][(po[m].id == i) & (po[m].greek == q)].iloc[0] for m in MODES}
            r = rows["oos"]
            w(f"| {int(ref.loc[i, 'K'])} | {q} | {r.ref:+.4f} | " + " | ".join(
                f"{rows[m].bias:+.4f} ± {rows[m].se:.4f} ({fmt(rows[m].t, 1, True)})" for m in MODES)
              + f" | {fmt(100 * rows['avg'].rel_bias, 2, True)} |")
    w("")

    # per option
    w("## Every option and Greek\n")
    w("Cells: bias ± se (t) [rel. bias %, rel. noise %].\n")
    w("| id | T | σ | r | d | K | Greek | ref | oos | ins | avg |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for (i, q), r0 in po["oos"].set_index(["id", "greek"]).iterrows():
        o = ref.loc[i]
        cells = []
        for m in MODES:
            r = po[m].set_index(["id", "greek"]).loc[(i, q)]
            cells.append(f"{r.bias:+.4f} ± {r.se:.4f} ({fmt(r.t, 1, True)}) "
                         f"[{fmt(100 * r.rel_bias, 2, True)}, {fmt(100 * r.rel_noise, 1)}]")
        w(f"| {i} | {o['T']:g} | {o.sigma:.0%} | {o.r:.0%} | {o.d:.0%} | {int(o.K)} | {q} | {r0.ref:+.4f} | "
          + " | ".join(cells) + " |")
    (DEST / "report_avg.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[: L.index("## Every option and Greek\n")]))


if __name__ == "__main__":
    main()
