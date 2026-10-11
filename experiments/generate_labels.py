"""Generate labels (price and Greeks) for a sample of Bermudan puts. Not part of
the paper.

Options are drawn on scrambled Sobol points (seeded) over strike, volatility,
rate, dividend yield and maturity at a fixed S0, then rounded to readable
values. Each label is one call of simgreeks.multi.label with the default
MultiConfig; labels scale exactly with (S0, K), so S0 only sets the units.

    python experiments/generate_labels.py [--n 100] [--S0 100] [--workers N]
                                          [--seed 1] [--exact] [--out PATH]

Writes one CSV row per option: the inputs, the label (price, delta, gamma,
theta, vega, volga, vanna, rho, phi, delta_r, delta_d, ex_region, alpha_S,
alpha_sigma) and, with --exact, the reference values as exact_<greek>
(reference.put_fd_greeks; where exercising at t_0 is optimal, the exercise
values K - S0, -1 and 0, as in the labels). The reference values are for
checking the labels (experiments/analyze_labels.py) and add about 25 % to the
time.

Small positive rates. Near r = 0 the early-exercise premium has a kink (it is 0
for r <= 0, d >= 0), and Rho changes fast just above it. Shrinking windows,
MultiConfig(alpha_r_shrink=True, alpha_d_shrink="r0"): alpha_r = alpha_d =
min(0.03, r0), remove most of the Rho and Phi bias there (200-replication check
at r0 = 0.5 % and 1 %, sigma = 20 %, T = 1), but the noise of one label grows
like 1 / r0: Rho 11-19 % at r0 = 1 % and 16-29 % at 0.5 %, Phi 13-23 % and
19-38 %, against 3-6 % with the 0.03 window. When building a training set,
oversample r0 in (0, 0.03) so that a network sees enough labels there to
average the extra noise out; r0 = 0 itself is exact with
MultiConfig(european_region=True). Open: at r0 = d0 = 1 % (K = 40, 44) Rho and
Phi stay 3-4 % off with any window.

Progress is printed at every 1 %. Each finished option is appended to
<out>.partial.csv, so an interrupted run resumes where it stopped when the same
command is run again; the partial files are removed once the CSV is written.
Each process uses one thread (OMP/MKL/OpenBLAS/Accelerate), so --workers should
be the number of physical cores.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")          # before numpy; inherited by the workers

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from simgreeks.core import PutSpec  # noqa: E402
from simgreeks.multi import MultiConfig, label  # noqa: E402
from simgreeks.reference import put_fd_greeks  # noqa: E402
from simgreeks.runner import job_seed  # noqa: E402

GREEKS = ["price", "delta", "gamma", "theta", "vega", "volga", "vanna", "rho", "phi",
          "delta_r", "delta_d"]
INPUTS = ["id", "S0", "K", "sigma", "r", "d", "T"]
EXTRA = ["ex_region", "alpha_S", "alpha_sigma"]
# (low, high, rounding step) of each input; K as a fraction of S0
RANGES = {"K": (0.85, 1.15, 0.01), "sigma": (0.10, 0.40, 0.01), "r": (0.0, 0.06, 0.0025),
          "d": (0.0, 0.04, 0.0025), "T": (0.25, 2.0, 0.05)}


def sample_options(n: int, S0: float, seed: int) -> pd.DataFrame:
    with warnings.catch_warnings():           # n need not be a power of 2 here
        warnings.simplefilter("ignore", UserWarning)
        U = qmc.Sobol(d=len(RANGES), scramble=True, seed=seed).random(n)
    cols = {}
    for j, (k, (lo, hi, step)) in enumerate(RANGES.items()):
        cols[k] = np.round((lo + (hi - lo) * U[:, j]) / step) * step
    df = pd.DataFrame(cols)
    df["K"] = np.round(df["K"] * S0, 2)
    df.insert(0, "S0", S0)
    df.insert(0, "id", np.arange(n))
    return df.round(6)


def spec_of(row) -> PutSpec:
    return PutSpec(S0=row["S0"], K=row["K"], sigma=row["sigma"], r=row["r"], d=row["d"],
                   T=row["T"])


def exact(spec: PutSpec) -> dict:
    f = put_fd_greeks(spec)
    out = {g: f[g] for g in GREEKS}
    if f["ex_region"]:                  # exercise at t_0, as the labels do
        out = {g: 0.0 for g in GREEKS}
        out["price"], out["delta"] = spec.K - spec.S0, -1.0
    return {f"exact_{g}": v for g, v in out.items()}


def one_option(job) -> dict:
    """Label (and reference values) of one option; runs in a worker."""
    row, base_seed, with_exact = job
    spec = spec_of(row)
    out = dict(row)
    lab = label(spec, MultiConfig(), job_seed(base_seed, {"id": int(row["id"])}, 0))
    out.update({k: lab[k] for k in GREEKS + EXTRA})
    if with_exact:
        out.update(exact(spec))
    return out


def _clock(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--exact", action="store_true",
                    help="add the reference PDE values (for checking the labels)")
    ap.add_argument("--no-exact", action="store_true", help=argparse.SUPPRESS)  # old default
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "labels_sample.csv")
    args = ap.parse_args()

    columns = INPUTS + GREEKS + EXTRA + ([f"exact_{g}" for g in GREEKS] if args.exact else [])
    settings = json.dumps(dict(n=args.n, S0=args.S0, seed=args.seed, exact=args.exact,
                               ranges=RANGES), sort_keys=True)
    partial = args.out.with_name(args.out.stem + ".partial.csv")
    meta = args.out.with_name(args.out.stem + ".partial.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    opts = sample_options(args.n, args.S0, args.seed)
    done = set()
    if partial.exists():
        if not meta.exists() or meta.read_text() != settings:
            sys.exit(f"{partial} was made with other settings; delete it (and {meta.name}) "
                     "or use another --out")
        done = set(pd.read_csv(partial, usecols=["id"])["id"])
        print(f"resuming: {len(done)} of {args.n} options already done", flush=True)
    else:
        meta.write_text(settings)
    todo = [dict(r) for r in opts.to_dict("records") if r["id"] not in done]

    workers = args.workers or os.cpu_count()
    print(f"{len(todo)} options to label on {workers} processes"
          f"{', with reference values' if args.exact else ''}", flush=True)
    t0, n_new, last_pct = time.time(), 0, int(100 * len(done) / args.n)
    with open(partial, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if fh.tell() == 0:
            writer.writeheader()
        ex = ProcessPoolExecutor(max_workers=workers)
        futures = [ex.submit(one_option, (r, args.seed, args.exact)) for r in todo]
        try:
            for fut in as_completed(futures):
                writer.writerow(fut.result())
                fh.flush()
                n_new += 1
                n_done = len(done) + n_new
                pct = int(100 * n_done / args.n)
                if pct > last_pct or n_done == args.n:
                    last_pct = pct
                    elapsed = time.time() - t0
                    left = elapsed / n_new * (args.n - n_done)
                    print(f"{pct:3d}% ({n_done}/{args.n}) · {_clock(elapsed)} elapsed · "
                          f"~{_clock(left)} left", flush=True)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            sys.exit(f"\nstopped after {len(done) + n_new} of {args.n} options; "
                     "run the same command again to resume")
        ex.shutdown()

    df = pd.read_csv(partial).sort_values("id").reset_index(drop=True)
    df.insert(df.columns.get_loc("K") + 1, "moneyness", (df.S0 / df.K).round(4))
    df.to_csv(args.out, index=False, float_format="%.6g")
    partial.unlink()
    meta.unlink()
    print(f"wrote {args.out} ({len(df)} options, {_clock(time.time() - t0)})")


if __name__ == "__main__":
    main()
