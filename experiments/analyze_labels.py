"""Analyse label CSVs written by experiments/generate_labels.py. Not part of the
paper.

    python experiments/analyze_labels.py results/labels_1000.csv [more.csv ...]
                                         [--plots DIR] [--md REPORT.md]

Always: what the files contain and sanity checks on the labels (missing
values, Delta outside [-1, 0], price below the exercise value or above K,
negative Gamma or Vega).

With reference values (files made with --exact), for each Greek:
  * error of one label e = (label - exact) / |exact|, in %; options where
    exercising now is optimal are exact by construction and left out, and so
    are options whose |exact| is below --floor times the largest |exact| of
    that Greek (relative errors are meaningless near zero);
  * typical error: median |e| (mostly simulation noise);
  * bias: mean e with its standard error, flagged when |mean / se| > 2.576;
  * heavy tails: share of labels with |e| above 3x the typical error (about
    4 % if the noise were normal);
  * where the bias is: mean e in thirds of each input (and inside / outside
    the range validated in results/multi_check_american.md), listing the
    regions where it is significant;
  * the labels with the largest errors.
--plots DIR saves errors against each input (errors_vs_inputs.png).
--md saves the tables as Markdown.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GREEKS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi",
          "delta_r", "delta_d"]
INPUTS = ["moneyness", "sigma", "T", "r", "d"]
Z99 = 2.576


def in_validated_range(df: pd.DataFrame) -> pd.Series:
    """S0/K 0.91-1.11, sigma 10-40 %, T 0.5-2: the grid of the American check."""
    return (df.moneyness.between(0.905, 1.115) & df.sigma.between(0.095, 0.405)
            & df["T"].between(0.495, 2.005))


def load(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        f = pd.read_csv(p)
        f.insert(0, "file", Path(p).name)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    if "moneyness" not in df:
        df["moneyness"] = df.S0 / df.K
    return df


def describe(df: pd.DataFrame) -> list[str]:
    out = [f"{len(df)} options from {df.file.nunique()} file(s); "
           f"{int(df.ex_region.sum())} where exercising now is optimal (label exact)"]
    rng = {k: (df[k].min(), df[k].max()) for k in INPUTS}
    out.append("inputs: " + ", ".join(f"{k} {lo:g}-{hi:g}" for k, (lo, hi) in rng.items()))
    out.append(f"inside the validated range: {int(in_validated_range(df).sum())} of {len(df)}")
    live = df[~df.ex_region]
    if len(live):
        out.append(f"chosen spreads: alpha_S {100 * (live.alpha_S / live.S0).min():.1f}-"
                   f"{100 * (live.alpha_S / live.S0).max():.1f} % of S0, alpha_sigma "
                   f"{(live.alpha_sigma / live.sigma).min():.2f}-"
                   f"{(live.alpha_sigma / live.sigma).max():.2f} x sigma")
    return out


def sanity(df: pd.DataFrame) -> pd.DataFrame:
    tol = 1e-3 * df.K
    checks = {
        "missing or infinite value": ~np.isfinite(df[GREEKS].astype(float)).all(axis=1),
        "Delta outside [-1, 0]": (df.delta < -1 - 1e-9) | (df.delta > 1e-9),
        "price below the exercise value K - S0": df.price < (df.K - df.S0).clip(lower=0) - tol,
        "price above K": df.price > df.K + tol,
        "negative Gamma": df.gamma < 0,
        "negative Vega": df.vega < 0,
    }
    rows = [dict(check=k, labels=int(m.sum()),
                 ids=", ".join(f"{f}:{i}" if df.file.nunique() > 1 else str(i)
                               for f, i in df.loc[m, ["file", "id"]].head(8).values))
            for k, m in checks.items()]
    return pd.DataFrame(rows)


def errors(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Relative error (%) of each label, NaN where it is not meaningful."""
    live = ~df.ex_region.astype(bool)
    E = pd.DataFrame(index=df.index)
    for g in GREEKS:
        ref = df[f"exact_{g}"]
        ok = live & (ref.abs() >= floor * ref[live].abs().max())
        E[g] = np.where(ok, 100 * (df[g] - ref) / ref.abs(), np.nan)
    return E


def accuracy(E: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GREEKS:
        e = E[g].dropna()
        if e.empty:
            continue
        typ = e.abs().median()
        mean, se = e.mean(), e.std() / np.sqrt(len(e))
        rows.append(dict(greek=g, n=len(e), typical_error=typ, bias=mean, bias_se=se,
                         significant="yes" if abs(mean) > Z99 * se else "",
                         median_error=e.median(),
                         heavy_tail=100 * (e.abs() > 3 * typ).mean(),
                         worst=e.loc[e.abs().idxmax()], worst_id=int(e.abs().idxmax())))
    return pd.DataFrame(rows)


def where_biased(df: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    """Regions (thirds of each input, validated range) with a significant mean error."""
    groups = {"validated range": in_validated_range(df).map({True: "inside", False: "outside"})}
    for k in INPUTS:
        if df[k].nunique() > 3:
            groups[k] = pd.qcut(df[k], 3, duplicates="drop").astype(str)
    rows = []
    for g in GREEKS:
        for name, lab in groups.items():
            for level, e in E[g].groupby(lab):
                e = e.dropna()
                if len(e) < 10:
                    continue
                mean, se = e.mean(), e.std() / np.sqrt(len(e))
                if abs(mean) > Z99 * se:
                    rows.append(dict(greek=g, input=name, region=level, n=len(e), bias=mean,
                                     bias_se=se))
    return pd.DataFrame(rows)


def worst(df: pd.DataFrame, E: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Labels whose largest error is furthest from the typical error, in units of it."""
    typ = E.abs().median()
    Z = (E.abs() / typ).fillna(0)
    top = Z.max(axis=1).sort_values(ascending=False).head(k).index
    out = df.loc[top, ["file", "id", "moneyness", "sigma", "T", "r", "d"]].copy()
    out["greek"] = Z.loc[top].idxmax(axis=1)
    out["error_%"] = [E.loc[i, g] for i, g in zip(top, out.greek)]
    out["x_typical"] = Z.loc[top].max(axis=1)
    return out


def plot(df: pd.DataFrame, E: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    greeks = [g for g in GREEKS if E[g].notna().sum() >= 10][:9]
    fig, axes = plt.subplots(len(greeks), len(INPUTS), figsize=(3.0 * len(INPUTS), 2.1 * len(greeks)),
                             sharex="col", squeeze=False)
    for i, g in enumerate(greeks):
        e = E[g]
        lim = 6 * e.abs().median()
        for j, k in enumerate(INPUTS):
            ax = axes[i, j]
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.scatter(df[k], e.clip(-lim, lim), s=6, color="#2a78d6", alpha=0.45, lw=0)
            m = e.notna()
            if m.sum() >= 20:
                bins = pd.qcut(df.loc[m, k], 8, duplicates="drop")
                med = e[m].groupby(bins, observed=True).median()
                ax.plot([b.mid for b in med.index], med.values, color="#eb6834", lw=2)
            ax.set_ylim(-lim * 1.05, lim * 1.05)
            ax.grid(True, color="#e1e0d9", lw=0.6)
            ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{g}\nerror %", fontsize=8)
            if i == len(greeks) - 1:
                ax.set_xlabel(k, fontsize=8)
    fig.suptitle("Error of one label vs the reference, by input (orange: binned median; "
                 "points beyond 6x the typical error drawn at the edge)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--plots", type=Path, help="directory for errors_vs_inputs.png")
    ap.add_argument("--md", type=Path, help="also write the tables as Markdown")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="skip |exact| below this fraction of the Greek's largest |exact|")
    args = ap.parse_args()

    df = load(args.csv)
    pd.set_option("display.width", 200)
    md = ["# Label analysis\n", ", ".join(str(p) for p in args.csv), ""]

    def show(title, table_or_lines, fmt=None):
        print(f"\n== {title} ==")
        md.append(f"## {title}\n")
        if isinstance(table_or_lines, list):
            print("\n".join(table_or_lines))
            md.extend(f"- {s}" for s in table_or_lines)
        elif table_or_lines.empty:
            print("none")
            md.append("none")
        else:
            t = table_or_lines.round(fmt or 2)
            print(t.to_string(index=False))
            md.append(t.to_markdown(index=False))
        md.append("")

    show("Data", describe(df))
    show("Sanity checks (labels)", sanity(df))
    if not all(f"exact_{g}" in df for g in GREEKS):
        print("\nNo reference values in these files: make them with "
              "generate_labels.py --exact to measure accuracy.")
    else:
        lab_ex = df.ex_region.astype(bool)
        ref_ex = (df.exact_delta == -1) & (df.exact_gamma == 0)
        show("Exercise now?", [
            f"label and reference agree on {int((lab_ex == ref_ex).sum())} of {len(df)} options",
            f"label exercises, reference does not: {int((lab_ex & ~ref_ex).sum())}",
            f"reference exercises, label does not: {int((~lab_ex & ref_ex).sum())}"])
        E = errors(df, args.floor)
        acc = accuracy(E)
        acc["worst_id"] = df.loc[acc.worst_id, "id"].values
        show("Accuracy of one label (errors in %)", acc)
        show("Where the bias is significant (mean error in %, 99 % level)", where_biased(df, E))
        show("Largest errors (in units of the typical error)", worst(df, E))
        if args.plots:
            args.plots.mkdir(parents=True, exist_ok=True)
            plot(df, E, args.plots / "errors_vs_inputs.png")
            print(f"\nwrote {args.plots / 'errors_vs_inputs.png'}")
    if args.md:
        args.md.write_text("\n".join(md) + "\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    sys.exit(main())
