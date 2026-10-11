"""Small positive r0: current (alpha 0.03 symmetric), kink-aware r basis, and
shrinking r window. (S, sigma) and (S, d) runs are shared; the (S, r) run is
repeated per variant on the same random numbers. Paired seeds."""
import sys; sys.path.insert(0, "/home/user/options_pricing")
import simgreeks  # noqa: F401
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, run_group
from simgreeks.runner import job_seed

KEYS = ["T", "sigma", "r", "d", "K"]
CELLS = [(1.0, 0.20, r, d, K) for r in (0.005, 0.01, 0.02, 0.03) for d in (0.0, 0.01) for K in (36, 40, 44)]
BASE = dict(alpha_r=0.03, alpha_d=0.03)
VARIANTS = {"current": MultiConfig(**BASE), "kink": MultiConfig(**BASE, kink_r=True),
            "shrink": MultiConfig(**BASE, alpha_r_shrink=True)}
REPS = 50

def one(args):
    cell, rep = args
    T, s, r, d, K = cell
    spec = PutSpec(K=K, sigma=s, T=T, r=r, d=d)
    kids = job_seed(9100, dict(zip(KEYS, cell)), rep).spawn(3)
    gs = run_group(spec, VARIANTS["current"], ("S", "sigma"), kids[0])
    gd = run_group(spec, VARIANTS["current"], ("S", "d"), kids[2])
    rows = []
    for v, cfg in VARIANTS.items():
        gr = run_group(spec, cfg, ("S", "r"), kids[1])
        rows.append(dict(zip(KEYS, cell), variant=v, rep=rep,
                         price=np.mean([gs["price"], gr["price"], gd["price"]]),
                         delta=np.mean([gs["delta"], gr["delta"], gd["delta"]]),
                         gamma=np.mean([gs["gamma"], gr["gamma"], gd["gamma"]]),
                         rho=gr["rho"], phi=gd["phi"], delta_r=gr["delta_r"],
                         price_r=gr["price"], delta_rrun=gr["delta"], gamma_rrun=gr["gamma"],
                         alpha_r=gr["alpha_r"]))
    return rows

if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(c, rep) for c in CELLS for rep in range(REPS)]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for n, rr in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.extend(rr)
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", flush=True)
