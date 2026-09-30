"""Replicate the experiments of Letourneau & Stentoft (2019) that showcase the
proposed method (Sections 3 and 4.1-4.3).

    python experiments/run.py section3            # Tables 1-4 and Figure 3
    python experiments/run.py fig2                # Figure 2
    python experiments/run.py table5 table6 table7 table8 table9
    python experiments/run.py all --reps 100

Per-replication estimates are written to results/raw/<exp>.csv, summary
tables (with the paper's numbers side by side) to results/<exp>.md and figures
to results/figures/. Benchmark (BM) values are taken from the paper; they are
not recomputed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simgreeks  # noqa: E402,F401  (sets single-threaded BLAS before numpy)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simgreeks.core import PutSpec, isd_sample, lsm, simulate_growth  # noqa: E402
from simgreeks.methods import AlgoConfig  # noqa: E402
from simgreeks.runner import run_configs  # noqa: E402
from simgreeks.report import summary_table, write_markdown  # noqa: E402

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FIGS = RESULTS / "figures"
PAPER = pd.read_csv(ROOT / "data" / "paper_tables.csv")

STRIKES = (36, 40, 44)
SEC3_ALPHAS = (0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40)


def _save_raw(df, name):
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / f"{name}.csv", index=False)


# --------------------------------------------------------------------------
# Section 3: Tables 1-4 and Figure 3
# --------------------------------------------------------------------------
def section3(reps, N):
    configs = [dict(spec=PutSpec(K=K), cfg=AlgoConfig(N=N, alpha=a),
                    tags=dict(K=K, alpha=a))
               for K in STRIKES for a in SEC3_ALPHAS]
    df = run_configs(configs, reps)
    _save_raw(df, "section3")
    return df


def report_section3(df):
    titles = {1: ("NAIVE", "Table 1: naive method"),
              2: ("NAIVE-VF", "Table 2: value function method"),
              3: ("TRUNC-VF", "Table 3: value function + truncation at optimal alpha"),
              4: ("2STEP-VF", "Table 4: proposed 2-step method (rescaling to optimal alpha)")}
    sections = []
    for t, (method, title) in titles.items():
        sub = df[(df.method == method) & df.alpha.isin([0.5, 5, 25])]
        paper = PAPER[PAPER.table == t].copy()
        paper["alpha"] = paper.alpha.astype(float)
        tab = summary_table(sub, ["alpha", "K"], paper, ["alpha", "K"])
        extra = ""
        if method in ("TRUNC-VF", "2STEP-VF"):
            a = sub.groupby(["alpha", "K"]).alpha_star.mean().round(2)
            extra = "\n\nMean estimated optimal alpha*: " + ", ".join(
                f"(alpha={i[0]:g}, K={i[1]}) {v}" for i, v in a.items())
        sections.append((title, tab, extra))
    write_markdown(RESULTS / "section3_tables1-4.md",
                   "Section 3: Tables 1-4 (N = 100,000, M_tau = M0 = 9)", sections)
    plot_fig3(df)


def plot_fig3(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    bm = PAPER[PAPER.table == 5]
    bm = bm[(bm.sigma.astype(float) == 20) & (bm["T"].astype(float) == 1.0)]
    styles = {"NAIVE": ("tab:gray", "o"), "NAIVE-VF": ("tab:red", "s"),
              "TRUNC-VF": ("tab:orange", "D"), "2STEP-VF": ("tab:blue", "v")}
    labels = {36: "OTM", 40: "ATM", 44: "ITM"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    z = 2.576
    for i, K in enumerate(STRIKES):
        bK = bm[bm.K.astype(int) == K].iloc[0]
        for j, q in enumerate(("price", "delta", "gamma")):
            ax = axes[i, j]
            ref = float(bK[f"{q}_bm"])
            for m, (c, mk) in styles.items():
                g = df[(df.K == K) & (df.method == m)].groupby("alpha")[q]
                mu, sd, n = g.mean(), g.std(), g.count()
                ci = z * sd / np.sqrt(n)
                ax.plot(mu.index, mu.values, color=c, marker=mk, ms=4, label=m)
                ax.fill_between(mu.index, mu - ci, mu + ci, color=c, alpha=0.15)
            ax.axhline(ref, color="k", lw=1.2, label="BMRK")
            span = {"price": 0.03, "delta": 0.02, "gamma": 0.012}[q]
            ax.set_ylim(ref - span, ref + span)
            ax.set_title(f"{q.capitalize()}, {labels[K]} (K={K})")
            ax.set_xlabel("initial alpha")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Figure 3: estimates with 99% confidence intervals across initial alpha")
    fig.tight_layout()
    fig.savefig(FIGS / "figure3.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------------------
def fig2(N=100_000, seed=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    spec = PutSpec(K=40)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, a in enumerate((0.5, 25)):
        rng = np.random.default_rng(seed)
        G = simulate_growth(spec, N, rng)
        X = isd_sample(N, spec.S0, a)
        res = lsm(spec, X[:, None] * G, 9)
        for row, (Y, lab) in enumerate(((res.Y_naive, "discounted payoffs from t = tau"),
                                        (res.Y_vf, "value function from t = 1"))):
            ax = axes[row, col]
            ax.scatter(X, Y, s=0.3, alpha=0.4, rasterized=True)
            ax.set_title(f"Using {lab}, alpha = {a:g}")
            ax.set_ylim(0, 10 if a < 1 else 30)
    fig.suptitle("Figure 2: data used in the regression at t = 0 (K = 40)")
    fig.tight_layout()
    fig.savefig(FIGS / "figure2.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Section 4 robustness tables
# --------------------------------------------------------------------------
TWO_STEP = ("2STEP-VF",)


def table5(reps, N):
    configs = []
    for T in (0.5, 1.0, 2.0):
        for sig in (0.10, 0.20, 0.40):
            for K in STRIKES:
                configs.append(dict(spec=PutSpec(K=K, sigma=sig, T=T),
                                    cfg=AlgoConfig(N=N, alpha=10, methods=TWO_STEP),
                                    tags=dict(K=K, sigma=int(round(sig * 100)), T=T)))
    df = run_configs(configs, reps)
    _save_raw(df, "table5")
    paper = PAPER[PAPER.table == 5].copy()
    paper["sigma"] = paper.sigma.astype(int)
    paper["T"] = paper["T"].astype(float)
    tab = summary_table(df, ["T", "sigma", "K"], paper, ["T", "sigma", "K"])
    write_markdown(RESULTS / "table5.md", "Table 5: 2-step method for a large sample of options",
                   [("alpha = 10, N = 100,000, M_tau = M0 = 9", tab, "")])


def table6(reps, N):
    configs = []
    for r, d in ((0.0, 0.0), (0.06, 0.0), (0.06, 0.06)):
        for K in STRIKES:
            configs.append(dict(spec=PutSpec(K=K, r=r, d=d),
                                cfg=AlgoConfig(N=N, alpha=10, methods=TWO_STEP),
                                tags=dict(K=K, r=int(round(r * 100)), d=int(round(d * 100)))))
    df = run_configs(configs, reps)
    _save_raw(df, "table6")
    paper = PAPER[PAPER.table == 6].copy()
    paper["r"] = paper.r.astype(int)
    paper["d"] = paper.d.astype(int)
    tab = summary_table(df, ["r", "d", "K"], paper, ["r", "d", "K"])
    write_markdown(RESULTS / "table6.md", "Table 6: across interest rates and dividend yields",
                   [("sigma = 20%, T = 1, alpha = 10", tab, "")])


def table7(reps, N_unused=None):
    configs = []
    for Nk in (50, 100, 200):
        for K in STRIKES:
            configs.append(dict(spec=PutSpec(K=K),
                                cfg=AlgoConfig(N=Nk * 1000, alpha=10, methods=TWO_STEP),
                                M0_list=[5, 9, 15],
                                tags=dict(K=K, N_thousands=Nk)))
    df = run_configs(configs, reps)
    _save_raw(df, "table7")
    paper = PAPER[PAPER.table == 7].copy()
    paper["N_thousands"] = paper.N_thousands.astype(int)
    paper["M0"] = paper.M0.astype(int)
    tab = summary_table(df, ["M0", "N_thousands", "K"], paper, ["M0", "N_thousands", "K"])
    write_markdown(RESULTS / "table7.md",
                   "Table 7: across number of paths N and t = 0 polynomial order M0",
                   [("alpha = 10, M_tau = 9", tab, "")])


def table8(reps, N):
    configs = [dict(spec=PutSpec(K=K),
                    cfg=AlgoConfig(N=N, alpha=10, M_tau=Mt, methods=TWO_STEP),
                    tags=dict(K=K, M_tau=Mt))
               for Mt in (5, 9, 15) for K in STRIKES]
    df = run_configs(configs, reps)
    _save_raw(df, "table8")
    paper = PAPER[PAPER.table == 8].copy()
    paper["M_tau"] = paper.M_tau.astype(int)
    tab = summary_table(df, ["M_tau", "K"], paper, ["M_tau", "K"])
    write_markdown(RESULTS / "table8.md",
                   "Table 8: across stopping-time polynomial order M_tau",
                   [("alpha = 10, M0 = 9", tab, "")])


def table9(reps, N):
    isd = {"Uni.D.": ("uniform", True, [2]),
           "Epa.D.": ("epanechnikov", True, [0, 1, 2]),
           "Epa.R.": ("epanechnikov", False, [2])}
    configs = []
    for name, (kern, det, nus) in isd.items():
        for K in STRIKES:
            configs.append(dict(spec=PutSpec(K=K),
                                cfg=AlgoConfig(N=N, alpha=10, isd_kernel=kern,
                                               isd_deterministic=det, methods=TWO_STEP),
                                nu_list=nus, tags=dict(K=K, ISD=name)))
    df = run_configs(configs, reps)
    _save_raw(df, "table9")
    df = df.copy()
    df["nu"] = df.nu.astype(int)
    paper = PAPER[PAPER.table == 9].drop_duplicates(["ISD", "nu", "K"]).copy()
    paper["nu"] = paper.nu.astype(int)
    tab = summary_table(df, ["ISD", "nu", "K"], paper, ["ISD", "nu", "K"])
    a = df.groupby(["ISD", "nu", "K"]).alpha_star.mean().round(2)
    extra = "\n\nMean estimated optimal alpha*: " + ", ".join(
        f"({i[0]}, nu={i[1]}, K={i[2]}) {v}" for i, v in a.items())
    write_markdown(RESULTS / "table9.md",
                   "Table 9: across ISD kernels and the derivative alpha* is optimised for",
                   [("alpha = 10, M_tau = M0 = 9 (nu = derivative order targeted by alpha*)",
                     tab, extra)])


EXPERIMENTS = ["section3", "fig2", "table5", "table6", "table7", "table8", "table9"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=EXPERIMENTS + ["all", "report"])
    ap.add_argument("--reps", type=int, default=100, help="independent replications")
    ap.add_argument("--N", type=int, default=100_000, help="simulated paths")
    args = ap.parse_args()
    exps = EXPERIMENTS if "all" in args.experiments else args.experiments
    for e in exps:
        t0 = time.time()
        print(f"== {e} ==", flush=True)
        if e == "fig2":
            fig2(args.N)
        elif e == "section3":
            report_section3(section3(args.reps, args.N))
        elif e == "report":
            report_section3(pd.read_csv(RAW / "section3.csv"))
        else:
            globals()[e](args.reps, args.N)
        print(f"   {e} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
