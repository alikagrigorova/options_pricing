"""Compare two runs of experiments/multi_check.py american, label by label
(simgreeks.multi, not part of the paper).

Seeds depend only on the option and the replication, so the labels of two
variants with the same replication number share their random numbers. For each
Greek, over the options outside the exercise region, this reports

  noise ratio : sd of one label in variant B / sd in variant A (median over options)
  bias A, B   : median |mean - ref| / |ref|, and options flagged at 1 %
  paired diff : median over options of mean(B - A) / |ref|, and options where the
                paired difference is significant at 1 %

on the replications both runs have.

    python experiments/diagnostics/compare_multi_checks.py american american_no_cv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.multi_check import KEYS, QUANTS, american_grid, reference  # noqa: E402
from simgreeks.report import Z99  # noqa: E402

RAW = ROOT / "results" / "raw"


def compare(a: str, b: str) -> str:
    A = pd.read_csv(RAW / f"multi_check_{a}.csv")
    B = pd.read_csv(RAW / f"multi_check_{b}.csv")
    m = A.merge(B, on=KEYS + ["rep"], suffixes=("_a", "_b"))
    ref = reference(american_grid(), "bermudan").rename(columns={q: f"{q}_ref" for q in QUANTS})
    m = m.merge(ref, on=KEYS)
    m = m[~m.ex_region_a.astype(bool)]
    reps = m.groupby(KEYS).size()
    lines = [f"# multi_check {b} vs {a}, paired\n",
             f"{len(reps)} options outside the exercise region, {reps.min()}-{reps.max()} "
             "replications each (those both runs have; same seeds). Medians over options "
             "with |ref| above 1 % of the largest |ref| for that Greek. Flagged: |t| > 2.576 "
             "(bias against the reference, or paired difference B − A).\n",
             f"| Greek | noise ratio {b}/{a} | bias {a} | flagged {a} | bias {b} | flagged {b} "
             f"| median paired diff ({b} − {a}) / ref | paired diffs flagged |",
             "|---|---|---|---|---|---|---|---|"]
    for q in QUANTS:
        rows = []
        for key, g in m.groupby(KEYS):
            r = g[f"{q}_ref"].iloc[0]
            n = len(g)
            sa, sb = g[f"{q}_a"].std(), g[f"{q}_b"].std()
            diff = g[f"{q}_b"] - g[f"{q}_a"]
            ta = (g[f"{q}_a"].mean() - r) / (sa / np.sqrt(n)) if sa > 0 else 0.0
            tb = (g[f"{q}_b"].mean() - r) / (sb / np.sqrt(n)) if sb > 0 else 0.0
            td = diff.mean() / (diff.std() / np.sqrt(n)) if diff.std() > 0 else 0.0
            rows.append(dict(ref=r, ratio=sb / sa if sa > 0 else np.nan,
                             ba=abs(g[f"{q}_a"].mean() - r), bb=abs(g[f"{q}_b"].mean() - r),
                             fa=abs(ta) > Z99, fb=abs(tb) > Z99,
                             d=diff.mean(), fd=abs(td) > Z99))
        t = pd.DataFrame(rows)
        big = t[t.ref.abs() > 0.01 * t.ref.abs().max()]
        rel = lambda x: (x / big.ref.abs()).median()  # noqa: E731
        lines.append(f"| {q} | {big.ratio.median():.2f} | {rel(big.ba):.2%} | "
                     f"{int(t.fa.sum())} / {len(t)} | {rel(big.bb):.2%} | "
                     f"{int(t.fb.sum())} / {len(t)} | {(big.d / big.ref.abs()).median():+.2%} | "
                     f"{int(t.fd.sum())} / {len(t)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    a, b = sys.argv[1:3]
    md = compare(a, b)
    (ROOT / "results" / f"multi_check_{b}_vs_{a}.md").write_text(md)
    print(md)
