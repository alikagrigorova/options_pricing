"""200-rep confirmation at r0 in {0.5%, 1%}: shrinking r window (alpha_r = min(0.03, r0))
and shrinking d window (min(0.03, r0) or min(0.03, d0)). Every group run is stored,
so labels can be assembled for any combination. Paired within a replication."""
import sys; sys.path.insert(0, "/home/user/options_pricing")
import simgreeks  # noqa: F401
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, run_group
from simgreeks.runner import job_seed

KEYS = ["T", "sigma", "r", "d", "K"]
CELLS = [(1.0, 0.20, r, d, K) for r in (0.005, 0.01) for d in (0.0, 0.01) for K in (36, 40, 44)]
BASE = dict(alpha_r=0.03, alpha_d=0.03)
RUNS = [("sigma", "cur", ("S", "sigma"), MultiConfig(**BASE)),
        ("r", "cur", ("S", "r"), MultiConfig(**BASE)),
        ("r", "shrink", ("S", "r"), MultiConfig(**BASE, alpha_r_shrink=True)),
        ("d", "cur", ("S", "d"), MultiConfig(**BASE)),
        ("d", "shrink_r0", ("S", "d"), MultiConfig(**BASE, alpha_d_shrink="r0")),
        ("d", "shrink_d0", ("S", "d"), MultiConfig(**BASE, alpha_d_shrink="d0"))]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

def one(args):
    cell, rep = args
    T, s, r, d, K = cell
    spec = PutSpec(K=K, sigma=s, T=T, r=r, d=d)
    slot = {"sigma": 0, "r": 1, "d": 2}
    rows = []
    for grp, var, dims, cfg in RUNS:
        if var == "shrink_d0" and d <= 0:
            continue
        # a fresh SeedSequence per run: run_group spawns from the one it gets,
        # so reusing one object would break the pairing between variants
        seed = job_seed(9200, dict(zip(KEYS, cell)), rep).spawn(3)[slot[grp]]
        g = run_group(spec, cfg, dims, seed)
        rows.append(dict(zip(KEYS, cell), rep=rep, group=grp, variant=var,
                         **{k: g[k] for k in ("price", "delta", "gamma", "rho", "phi", "delta_r",
                                              "delta_d", "vega") if k in g},
                         alpha=g.get(f"alpha_{grp}")))
    return rows

if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(c, rep) for c in CELLS for rep in range(REPS)]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for n, rr in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.extend(rr)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", flush=True)
