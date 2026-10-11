"""Symmetric vs one-sided (floored at 0) r, d spreads at alpha = 0.03, paired seeds
with multi_check american ((S, r) and (S, d) runs only)."""
import sys; sys.path.insert(0, "/home/user/options_pricing")
import simgreeks  # noqa: F401
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, run_group
from simgreeks.runner import job_seed

KEYS = ["T", "sigma", "r", "d", "K"]
CELLS = [(1.0, 0.20, 0.0, 0.0, K) for K in (36, 40, 44)] + [(1.0, 0.20, 0.06, 0.0, K) for K in (36, 40, 44)]
VARIANTS = {"sym": dict(alpha_r=0.03, alpha_d=0.03), "floor": dict(alpha_r=0.03, alpha_d=0.03, rd_floor=True)}
REPS = 50

def one(args):
    cell, v, rep = args
    T, s, r, d, K = cell
    spec = PutSpec(K=K, sigma=s, T=T, r=r, d=d)
    cfg = MultiConfig(**VARIANTS[v])
    kids = job_seed(7002, dict(zip(KEYS, cell)), rep).spawn(3)
    gr = run_group(spec, cfg, ("S", "r"), kids[1])
    gd = run_group(spec, cfg, ("S", "d"), kids[2])
    return dict(zip(KEYS, cell), variant=v, rep=rep, rho=gr["rho"], delta_r=gr["delta_r"],
                phi=gd["phi"], delta_d=gd["delta_d"], price_r=gr["price"], price_d=gd["price"])

if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(c, v, rep) for c in CELLS for v in VARIANTS for rep in range(REPS)]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for n, row in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.append(row)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", flush=True)
