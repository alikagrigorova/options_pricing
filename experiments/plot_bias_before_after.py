"""Bias of the labels before and after averaging the out-of-sample and in-sample
estimates, on the Stage 2 data (10,000 options, results/stage2/labels.parquet).
Not part of the paper.

Before: the out-of-sample label (<greek>_oos), the earlier default, whose exercise
rule is fitted on the pilot paths (a lower bound: price low, |Delta| small).
Now: the default label (<greek>; the average of the out-of-sample and in-sample
labels for price, Delta, Gamma, Theta, Vega, Rho and Phi). Both come from the
same paths, so the comparison is paired. Volga, Vanna and the cross terms are the
out-of-sample label in both, so they are not shown.

Error = (label - PDE) / |PDE| in %, over the options whose label is not exact by
construction (r0 = 0, or exercise at t0 in label and reference) and with |PDE| at
least 1 % of the Greek's largest |PDE|, as in experiments/stage2_report.py.

    python experiments/plot_bias_before_after.py [--out results/bias_before_after]

Writes
  bias_overall.png          mean error per Greek with 99 % intervals, before vs now
  bias_by_input.png         mean error in eight bins of each input, 99 % bands
  errors_vs_inputs_before.png, errors_vs_inputs_now.png
                            the error of every label and its binned median, in the
                            style of results/labels_1000_analysis/errors_vs_inputs.png,
                            with the same axes in both
  summary.md                the numbers behind bias_overall.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GREEKS = ["price", "delta", "gamma", "theta", "vega", "rho", "phi"]
NAMES = {"price": "Price", "delta": "Δ", "gamma": "Γ", "theta": "Θ", "vega": "Vega",
         "rho": "Rho", "phi": "Phi"}
INPUTS = ["moneyness", "sigma", "T", "r", "d"]
XLABEL = {"moneyness": "moneyness S0/K", "sigma": "σ", "T": "T (years)", "r": "r", "d": "d"}
Z99 = 2.576
BEFORE, NOW = "#eb6834", "#2a78d6"          # categorical slots 2 and 1, validated as a pair
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SURFACE = "#fcfcfb"


def load(path: Path):
    df = pd.read_parquet(path)
    df["moneyness"] = df.S0 / df.K
    exact = df.european_region.astype(bool) | (df.ex_region.astype(bool)
                                               & df.ref_ex_region.astype(bool))
    before, now = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)
    for g in GREEKS:
        ref = df[f"ref_{g}"]
        ok = ~exact & (ref.abs() >= 0.01 * ref.abs().max())
        before[g] = np.where(ok, 100 * (df[f"{g}_oos"] - ref) / ref.abs(), np.nan)
        now[g] = np.where(ok, 100 * (df[g] - ref) / ref.abs(), np.nan)
    return df, before, now, int(exact.sum())


def mean_ci(e: pd.Series):
    e = e.dropna()
    m, se = e.mean(), e.std() / np.sqrt(len(e))
    return m, se, len(e)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def legend(fig, y=0.985):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BEFORE, marker="o", lw=2, ms=6,
                      label="before: out-of-sample label"),
               Line2D([], [], color=NOW, marker="o", lw=2, ms=6,
                      label="now: average of out-of-sample and in-sample labels")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, y), labelcolor=INK)


def plot_overall(before, now, path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(GREEKS), figsize=(2.0 * len(GREEKS), 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, g in zip(axes, GREEKS):
        style(ax)
        ax.axhline(0, color=INK2, lw=0.9)
        vals = []
        for x, (E, c) in enumerate(((before, BEFORE), (now, NOW))):
            m, se, _ = mean_ci(E[g])
            ax.errorbar(x, m, yerr=Z99 * se, fmt="o", color=c, ms=8, lw=2, capsize=0,
                        mec=SURFACE, mew=1.5)
            vals.append((m, se))
        span = max(abs(m) + Z99 * se for m, se in vals) * 1.6
        for x, (m, se) in enumerate(vals):
            ax.annotate(f"{m:+.3f}%" if abs(m) < 0.05 else f"{m:+.2f}%", (x, m + np.sign(m or 1) * (Z99 * se + 0.08 * span)),
                        ha="center", va="bottom" if m >= 0 else "top", fontsize=7.5, color=INK)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-span, span)
        ax.set_xticks([0, 1], ["before", "now"])
        ax.set_title(NAMES[g], fontsize=10, color=INK)
        if g == GREEKS[0]:
            ax.set_ylabel("mean error of one label (%)", fontsize=8, color=INK2)
    legend(fig, 1.0)
    fig.text(0.5, 0.01, "10,000 random options (Stage 2), same paths before and now; bars: 99 % "
             "interval of the mean. An interval that crosses 0 means no significant bias.",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def binned(df, E, k, bins=8):
    m = E.notna()
    b = pd.qcut(df.loc[m, k], bins, duplicates="drop")
    g = E[m].groupby(b, observed=True)
    out = g.agg(["mean", "std", "count"])
    out["x"] = df.loc[m, k].groupby(b, observed=True).mean()
    out["half"] = Z99 * out["std"] / np.sqrt(out["count"])
    return out


def plot_by_input(df, before, now, path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(GREEKS), len(INPUTS), figsize=(3.0 * len(INPUTS), 2.0 * len(GREEKS)),
                             sharex="col", sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, g in enumerate(GREEKS):
        for j, k in enumerate(INPUTS):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=INK2, lw=0.9)
            for E, c in ((before, BEFORE), (now, NOW)):
                t = binned(df, E[g], k)
                ax.fill_between(t.x, t["mean"] - t.half, t["mean"] + t.half, color=c,
                                alpha=0.18, lw=0)
                ax.plot(t.x, t["mean"], color=c, lw=2, marker="o", ms=3.5)
            if j == 0:
                ax.set_ylabel(f"{NAMES[g]}\nmean error %", fontsize=8, color=INK2)
            if i == len(GREEKS) - 1:
                ax.set_xlabel(XLABEL[k], fontsize=8, color=INK2)
    fig.suptitle("Mean error of one label in eight bins of each input (bands: 99 % interval)",
                 fontsize=10, color=INK, y=0.995)
    legend(fig, 0.982)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_scatter(df, E, limits, title, color, path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(GREEKS), len(INPUTS), figsize=(3.0 * len(INPUTS), 2.0 * len(GREEKS)),
                             sharex="col", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, g in enumerate(GREEKS):
        e, lim = E[g], limits[g]
        for j, k in enumerate(INPUTS):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=INK2, lw=0.9)
            ax.scatter(df[k], e.clip(-lim, lim), s=3, color="#9bb9df", alpha=0.35, lw=0,
                       rasterized=True)
            m = e.notna()
            b = pd.qcut(df.loc[m, k], 8, duplicates="drop")
            med = e[m].groupby(b, observed=True).median()
            x = df.loc[m, k].groupby(b, observed=True).mean()
            ax.plot(x, med.values, color=color, lw=2.2)
            ax.set_ylim(-lim * 1.05, lim * 1.05)
            if j == 0:
                ax.set_ylabel(f"{NAMES[g]}\nerror %", fontsize=8, color=INK2)
            if i == len(GREEKS) - 1:
                ax.set_xlabel(XLABEL[k], fontsize=8, color=INK2)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def summary(before, now, n_exact) -> str:
    lines = ["# Label bias before and after averaging (Stage 2, 10,000 options)\n",
             f"Mean relative error of one label (%) ± se, t in brackets; {n_exact} options with "
             "exact labels left out, and points with |PDE| < 1 % of the Greek's largest. "
             "Before: out-of-sample label; now: the default (averaged) label; same paths. "
             "† |t| > 2.576.\n",
             "| Greek | n | before | now |", "|---|---|---|---|"]
    for g in GREEKS:
        cells = []
        for E in (before, now):
            m, se, n = mean_ci(E[g])
            cells.append(f"{m:+.3f} ± {se:.3f} ({m / se:+.1f}){'†' if abs(m / se) > Z99 else ''}")
        lines.append(f"| {NAMES[g]} | {n} | {cells[0]} | {cells[1]} |")
    lines += ["", "Volga, Vanna, ∂Δ/∂r and ∂Δ/∂d are the out-of-sample label in both, so they "
              "do not change. Figures: bias_overall.png, bias_by_input.png, "
              "errors_vs_inputs_before.png, errors_vs_inputs_now.png."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "stage2" / "labels.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bias_before_after")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    df, before, now, n_exact = load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_overall(before, now, args.out / "bias_overall.png")
    plot_by_input(df, before, now, args.out / "bias_by_input.png")
    limits = {g: 6 * now[g].abs().median() for g in GREEKS}     # same axes in both
    note = "points beyond 6x the median |error| drawn at the edge"
    plot_scatter(df, before, limits, "Before (out-of-sample label): error of one label vs the PDE, "
                 f"by input; line: binned median; {note}", BEFORE,
                 args.out / "errors_vs_inputs_before.png")
    plot_scatter(df, now, limits, "Now (averaged label): error of one label vs the PDE, by input; "
                 f"line: binned median; {note}", NOW, args.out / "errors_vs_inputs_now.png")
    md = summary(before, now, n_exact)
    (args.out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
