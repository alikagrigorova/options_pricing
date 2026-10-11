"""Width sweep for the fixed r and d dispersions (alpha_r = alpha_d), paired with
the baseline multi_check american labels (same seeds, (S, r) and (S, d) runs only)."""
import sys; sys.path.insert(0, "/home/user/options_pricing")
import simgreeks  # noqa: F401  (single-thread BLAS)
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, run_group
from simgreeks.runner import job_seed

KEYS = ["T", "sigma", "r", "d", "K"]
CELLS = [(1.0, 0.20, 0.0, 0.0, K) for K in (36, 40, 44)] + \
        [(1.0, 0.20, 0.06, 0.06, K) for K in (36, 40, 44)] + \
        [(1.0, 0.20, 0.06, 0.0, 40), (0.5, 0.10, 0.06, 0.0, 40), (2.0, 0.40, 0.06, 0.0, 40)]
WIDTHS = tuple(float(w) for w in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0.01, 0.02, 0.03, 0.04)
REPS = 25

def one(args):
    cell, w, rep = args
    T, s, r, d, K = cell
    spec = PutSpec(K=K, sigma=s, T=T, r=r, d=d)
    cfg = MultiConfig(alpha_r=w, alpha_d=w)
    kids = job_seed(7002, dict(zip(KEYS, cell)), rep).spawn(3)
    gr = run_group(spec, cfg, ("S", "r"), kids[1])
    gd = run_group(spec, cfg, ("S", "d"), kids[2])
    return dict(zip(KEYS, cell), width=w, rep=rep, rho=gr["rho"], delta_r=gr["delta_r"],
                phi=gd["phi"], delta_d=gd["delta_d"], price_r=gr["price"], price_d=gd["price"],
                gamma_r=gr["gamma"], gamma_d=gd["gamma"])

if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(c, w, rep) for c in CELLS for w in WIDTHS for rep in range(REPS)]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for n, row in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.append(row)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", flush=True)
