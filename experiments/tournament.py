"""Label-design tournament (simgreeks.multi, not part of the paper).

Eight variants: designs D1-D4 (simgreeks.multi.DESIGNS) x exercise rule fitted
out of sample on the pilot paths ("oos") or in sample on the main paths ("ins"),
all with 400,000 main + 400,000 pilot paths per label (design_config).

Parts (one job = one label = option x replication x variant):
  european   : D3, D4 on the 6 European check options (no early exercise, no
               control variate), 100 reps; reference: closed form
  american   : the 33 options of multi_check american, 50 reps, 8 variants
  tournament : N_OPTIONS options on scrambled Sobol points (inputs()), 1 label
               per option, 8 variants
  reference  : reference.label_reference for the tournament options

Seeds depend only on (part, option, replication), never on the variant, so
variants are paired. Task i of n runs every n-th job; rows are written to
<out>/<part>/task_<i>/chunk_<k>.parquet every --chunk labels, and a restarted
task skips the jobs already written (resumable).

    python experiments/tournament.py run --part american --task 0 --ntasks 200 --out OUT
    python experiments/tournament.py local --part european --workers 8 --out OUT
    python experiments/tournament.py report --part american --out OUT
    python experiments/tournament.py report --part timing --out OUT   (after a run with --limit)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")          # before numpy; inherited by the workers

import argparse  # noqa: E402
import glob  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import DESIGNS, GREEKS, MultiConfig, design_config, label  # noqa: E402
from simgreeks.reference import (FORWARD_GREEKS, LABEL_GREEKS, bs_put,  # noqa: E402
                                 label_reference)
from simgreeks.runner import job_seed  # noqa: E402

S0 = 40.0
N_OPTIONS = 2500
INPUT_SEED = 20261011
BASE_SEED = {"european": 8001, "american": 8002, "tournament": 8003, "production_check": 8004}
REPS = {"european": 100, "american": 50, "tournament": 1, "production_check": 100}
# production check: the 33 American check options plus r0 = d0 = 1 %, K = 40, 44 (sigma 20 %, T = 1)
EXTRA_CELLS = [(1.0, 0.20, 0.01, 0.01, 40), (1.0, 0.20, 0.01, 0.01, 44)]
PROD_VARIANT = "D2-prod"     # MultiConfig() defaults: D2, out-of-sample rule, 100,000 paths per run
# Extra variants, run on request (--variants) and included in the reports:
#   D3o3-oos  : D3 with order 3 in sigma, r and d at t = 0 (instead of 2 in r, d), tournament budget
#   D3o3-prod : the same with production settings, 100,000 main + 100,000 pilot paths per run
ORDER3 = (("sigma", 3), ("r", 3), ("d", 3))
EXTRA_VARIANTS = ["D3o3-oos", "D3o3-prod", PROD_VARIANT]
VARIANTS = [f"{d}-{rule}" for d in DESIGNS for rule in ("oos", "ins")]
EU_VARIANTS = ["D3-eu", "D4-eu"]
INPUT_COLS = ["S0", "K", "sigma", "r", "d", "T"]
OUT_GREEKS = list(GREEKS) + ["theta"]
EXTRA = ["ex_region", "european_region", "alpha_S", "alpha_sigma", "alpha_r", "alpha_d"]


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
def inputs(n: int = N_OPTIONS) -> pd.DataFrame:
    """Tournament options: S0 = 40; S0/K in [0.85, 1.20], sigma in [0.10, 0.40],
    T in [0.25, 2], d in [0, 0.04]; r: 2 % at r = 0, 25 % in (0, 0.03), the rest
    in [0.03, 0.06]. r is rounded to 0.25 % steps, so the smallest positive r0 is
    0.0025 and the shrinking window alpha_r = min(0.03, r0) stays >= 0.0025."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        U = qmc.Sobol(d=5, scramble=True, seed=INPUT_SEED).random(n)
    m = 0.85 + 0.35 * U[:, 0]
    sigma = np.round(0.10 + 0.30 * U[:, 1], 3)
    T = np.round(0.25 + 1.75 * U[:, 2], 3)
    d = np.round(0.04 * U[:, 3], 4)
    u = U[:, 4]
    r = np.where(u < 0.02, 0.0,
                 np.where(u < 0.27, 0.0025 + 0.0250 * (u - 0.02) / 0.25,
                          0.03 + 0.03 * (u - 0.27) / 0.73))
    r = np.round(r / 0.0025) * 0.0025
    df = pd.DataFrame({"id": np.arange(n), "S0": S0, "K": np.round(S0 / m, 2),
                       "sigma": sigma, "r": np.round(r, 6), "d": d, "T": T})
    return df


def options(part: str) -> pd.DataFrame:
    if part == "tournament" or part == "reference":
        return inputs()
    from experiments.multi_check import american_grid, european_grid
    cells = european_grid() if part == "european" else american_grid()
    if part == "production_check":
        cells = cells + EXTRA_CELLS
    rows = [dict(id=i, S0=S0, K=float(K), sigma=s, r=r, d=d, T=T)
            for i, (T, s, r, d, K) in enumerate(cells)]
    return pd.DataFrame(rows)


def spec_of(row) -> PutSpec:
    return PutSpec(S0=float(row["S0"]), K=float(row["K"]), sigma=float(row["sigma"]),
                   r=float(row["r"]), d=float(row["d"]), T=float(row["T"]))


def config_of(variant: str) -> MultiConfig:
    if variant == PROD_VARIANT:
        return MultiConfig()
    if variant == "D3o3-oos":
        # the tournament's alpha_d rule (min(0.03, r0)), so only the t = 0 order differs from D3-oos
        return design_config("D3", "pilot", t0_max_order=ORDER3, alpha_d_shrink="r0")
    if variant == "D3o3-prod":
        return design_config("D3", "pilot", budget=300_000, t0_max_order=ORDER3)
    design, rule = variant.split("-")
    if rule == "eu":
        return design_config(design, style="european", control_variate=False)
    return design_config(design, "pilot" if rule == "oos" else "insample")


def jobs_of(part: str, limit: int | None = None, variants=None):
    opts = options(part)
    if limit:
        opts = opts.iloc[:limit]
    if part == "reference":
        return [(int(i), 0, "reference") for i in opts.id]
    vs = variants or (EU_VARIANTS if part == "european" else
                      [PROD_VARIANT] if part == "production_check" else VARIANTS)
    return [(int(i), rep, v) for i in opts.id for rep in range(REPS[part]) for v in vs]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def _done(task_dir: Path) -> set:
    done = set()
    for f in sorted(task_dir.glob("chunk_*.parquet")):
        df = pd.read_parquet(f, columns=["id", "rep", "variant"])
        done.update(zip(df.id, df.rep, df.variant))
    return done


def one(part: str, opts: pd.DataFrame, job, mem: bool) -> dict:
    i, rep, variant = job
    row = opts.loc[opts.id == i].iloc[0]
    spec = spec_of(row)
    out = dict(part=part, id=i, rep=rep, variant=variant, **{c: float(row[c]) for c in INPUT_COLS})
    if mem:
        tracemalloc.start()
    t = time.perf_counter()
    if part == "reference":
        res = label_reference(spec)
        out.update({q: res[q] for q in LABEL_GREEKS})
        out.update({f"{q}_fwd": res[f"{q}_fwd"] for q in FORWARD_GREEKS})
        out["ex_region"] = res["ex_region"]
    else:
        seed = job_seed(BASE_SEED[part], {"id": i}, rep)     # fresh SeedSequence per label
        out["seed_entropy"] = str(seed.entropy)
        res = label(spec, config_of(variant), seed)
        out.update({q: float(res.get(q, np.nan)) for q in OUT_GREEKS})
        out.update({k: res.get(k, np.nan) for k in EXTRA})
        out["design"], out["rule"] = variant.split("-")[0], variant.split("-")[-1]
    out["seconds"] = time.perf_counter() - t
    if mem:
        out["peak_mb"] = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
    out["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # process peak so far
    return out


def run(part: str, task: int, ntasks: int, out: str, limit: int | None = None,
        chunk: int = 10, mem: bool = False, variants=None):
    jobs = jobs_of(part, limit, variants)[task::ntasks]
    task_dir = Path(out) / part / f"task_{task:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    done = _done(task_dir)
    todo = [j for j in jobs if j not in done]
    k = len(list(task_dir.glob("chunk_*.parquet")))
    print(f"{part} task {task}/{ntasks}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to do",
          flush=True)
    opts = options(part)
    rows = []
    for n, job in enumerate(todo, 1):
        rows.append(one(part, opts, job, mem))
        if len(rows) >= chunk or n == len(todo):
            pd.DataFrame(rows).to_parquet(task_dir / f"chunk_{k:05d}.parquet", index=False)
            k += 1
            rows = []
            print(f"  {n}/{len(todo)} written", flush=True)


def _local_task(args):
    run(*args)


def local(part: str, workers: int, out: str, limit=None, chunk=10, mem=False, variants=None):
    """All tasks of a part in one process pool (e.g. in tmux on an interactive node)."""
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_local_task, [(part, t, workers, out, limit, chunk, mem, variants)
                                  for t in range(workers)]))


def load(part: str, out: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(out) / part / "task_*" / "chunk_*.parquet")))
    if not files:
        raise SystemExit(f"no output for {part} in {out}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
QUANTS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi",
          "delta_r", "delta_d", "vera"]


def check_reference(part: str, out: str) -> pd.DataFrame:
    """Reference values for the european / american check options (cached)."""
    path = Path(out) / f"reference_{part}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for _, o in options(part).iterrows():
        spec = spec_of(o)
        if part == "european":
            f = bs_put(spec)
            res = {q: float(f[q]) for q in QUANTS}
            res.update({f"{q}_fwd": res[q] for q in FORWARD_GREEKS})
        else:
            res = label_reference(spec)
        rows.append(dict(id=int(o.id), **{q: res[q] for q in QUANTS},
                         **{f"{q}_fwd": res[f"{q}_fwd"] for q in FORWARD_GREEKS}))
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def report_check(part: str, out: str) -> str:
    """(a) flags, median / max abs(bias) %, median noise % of one label, per variant and Greek."""
    lab, ref = load(part, out), check_reference(part, out).set_index("id")
    variants = [v for v in (EU_VARIANTS if part == "european" else VARIANTS + EXTRA_VARIANTS)
                if v in set(lab.variant)]
    lines = [f"# {part} check: {lab.groupby('variant').size().to_dict()} labels\n",
             "Per variant and Greek: options flagged (|mean - ref| / (sd / sqrt(reps)) > 2.576) / "
             "options used, median and max abs(bias) %, median noise % of one label. Options "
             "with abs(ref) below 1 % of the largest abs(ref) for that Greek are left out of the "
             "% columns. Rho/Phi against the central reference.\n",
             "| Greek | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for q in QUANTS:
        cells = []
        for v in variants:
            g = lab[lab.variant == v].groupby("id")[q]
            m, sd, n = g.mean(), g.std(ddof=1), g.count()
            r = ref[q].reindex(m.index)
            if m.isna().all():
                cells.append("–")
                continue
            t = (m - r) / (sd / np.sqrt(n))
            flags = int((t.abs() > 2.576).sum())
            use = r.abs() > 0.01 * r.abs().max()
            rb = ((m - r).abs() / r.abs())[use]
            rn = (sd / r.abs())[use]
            cells.append(f"{flags}/{int(m.notna().sum())} · {100*rb.median():.1f} · "
                         f"{100*rb.max():.1f} · {100*rn.median():.1f}")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    lines.append("\nseconds per label: " + ", ".join(
        f"{v} {lab[lab.variant == v].seconds.mean():.1f}" for v in variants))
    return "\n".join(lines) + "\n"


REGIONS = {
    "all": lambda o: np.ones(len(o), bool),
    "r = 0": lambda o: o.r == 0,
    "small r (0 < r < 3 %)": lambda o: (o.r > 0) & (o.r < 0.03),
    "deep ITM (S0/K <= 0.90)": lambda o: o.S0 / o.K <= 0.90,
    "OTM (S0/K >= 1.10)": lambda o: o.S0 / o.K >= 1.10,
    "short T (<= 0.5)": lambda o: o["T"] <= 0.5,
    "long T (>= 1.5)": lambda o: o["T"] >= 1.5,
    "low sigma (<= 0.15)": lambda o: o.sigma <= 0.15,
    "high sigma (>= 0.35)": lambda o: o.sigma >= 0.35,
}


def report_tournament(out: str, base: str = "D2-oos") -> str:
    """(b) relative errors (label - ref) / abs(ref) per variant and Greek: mean signed
    bias +- se overall and by region, RMSE %, seconds per label; paired differences
    against ``base``."""
    lab = load("tournament", out)
    ref = load("reference", out).drop_duplicates("id").set_index("id")
    opts = inputs().set_index("id")
    variants = [v for v in VARIANTS + EXTRA_VARIANTS if v in set(lab.variant)]
    lines = [f"# Tournament: {lab.groupby('variant').size().to_dict()} labels, "
             f"{len(ref)} reference values\n",
             "e = (label - ref) / abs(ref), over the options with abs(ref) above 1 % of the "
             "largest abs(ref) for that Greek and outside the exercise region. Cells: mean e "
             "+- se (%), RMSE (%). Rho, Phi, delta_r, delta_d also against the forward "
             "reference (_fwd).\n"]
    errs = {}
    for v in variants:
        L = lab[lab.variant == v].set_index("id")
        for q in QUANTS + [f"{q}_fwd" for q in FORWARD_GREEKS]:
            qq = q.replace("_fwd", "")
            r = ref[q].reindex(L.index)
            use = (r.abs() > 0.01 * ref[q].abs().max()) & ~ref.ex_region.reindex(L.index).astype(bool)
            errs[(v, q)] = ((L[qq] - r) / r.abs())[use].dropna()
    for region, sel in REGIONS.items():
        ids = set(opts.index[sel(opts)])
        lines += [f"\n## {region} ({len(ids)} options)\n",
                  "| Greek | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
        for q in QUANTS + [f"{q}_fwd" for q in FORWARD_GREEKS]:
            cells = []
            for v in variants:
                e = errs[(v, q)]
                e = e[e.index.isin(ids)]
                if len(e) < 2:
                    cells.append("–")
                    continue
                cells.append(f"{100*e.mean():+.2f} ± {100*e.std(ddof=1)/np.sqrt(len(e)):.2f}, "
                             f"{100*np.sqrt((e**2).mean()):.1f}")
            lines.append(f"| {q} | " + " | ".join(cells) + " |")
    lines += [f"\n## Paired differences against {base} (all options)\n",
              "Cells: mean (e_v - e_base) +- se (%); mean (e_v^2 - e_base^2) +- se (in %^2; "
              "negative = more accurate than the base).\n",
              "| Greek | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for q in QUANTS:
        cells = []
        for v in variants:
            a, b = errs[(base, q)], errs[(v, q)]
            j = a.index.intersection(b.index)
            if len(j) < 2 or v == base:
                cells.append("–")
                continue
            d1, d2 = (b[j] - a[j]) * 100, (b[j] ** 2 - a[j] ** 2) * 1e4
            cells.append(f"{d1.mean():+.2f} ± {d1.std()/np.sqrt(len(j)):.2f}; "
                         f"{d2.mean():+.1f} ± {d2.std()/np.sqrt(len(j)):.1f}")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    lines.append("\n## Seconds per label\n")
    lines.append("| variant | mean | p95 |")
    lines.append("|---|---|---|")
    for v in variants:
        s = lab[lab.variant == v].seconds
        lines.append(f"| {v} | {s.mean():.1f} | {s.quantile(0.95):.1f} |")
    return "\n".join(lines) + "\n"


def report_timing(out: str, part: str = "tournament") -> str:
    lab = load(part, out)
    lines = [f"# Timing pilot ({part}): {len(lab)} labels\n",
             "| variant | labels | seconds mean | p95 | max | peak MB (tracemalloc) mean | max | "
             "process peak RSS MB |",
             "|---|---|---|---|---|---|---|---|"]
    for v, g in lab.groupby("variant"):
        pk = g["peak_mb"] if "peak_mb" in g else pd.Series([np.nan])
        lines.append(f"| {v} | {len(g)} | {g.seconds.mean():.1f} | {g.seconds.quantile(.95):.1f} | "
                     f"{g.seconds.max():.1f} | {pk.mean():.0f} | {pk.max():.0f} | {g.rss_mb.max():.0f} |")
    tot = lab.groupby("variant").seconds.mean()
    lines.append(f"\nCPU hours for {N_OPTIONS} options x all variants: "
                 f"{tot.sum() * N_OPTIONS / 3600:.0f}; for 33 x 50 american labels: "
                 f"{tot.sum() * 33 * 50 / 3600:.0f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "local", "report", "count"])
    ap.add_argument("--part", required=True,
                    choices=["european", "american", "tournament", "reference", "timing", "production_check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="first LIMIT options only (timing pilot)")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--mem", action="store_true", help="record peak memory per label (tracemalloc)")
    ap.add_argument("--variants", default=None, help="comma-separated subset of variants")
    a = ap.parse_args()
    variants = a.variants.split(",") if a.variants else None
    if a.cmd == "count":
        print(len(jobs_of(a.part, a.limit, variants)))
    elif a.cmd == "run":
        run(a.part, a.task, a.ntasks, a.out, a.limit, a.chunk, a.mem, variants)
    elif a.cmd == "local":
        local(a.part, a.workers, a.out, a.limit, a.chunk, a.mem, variants)
    else:
        text = (report_timing(a.out) if a.part == "timing" else
                report_tournament(a.out) if a.part == "tournament" else report_check(a.part, a.out))
        path = Path(a.out) / f"report_{a.part}.md"
        path.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
