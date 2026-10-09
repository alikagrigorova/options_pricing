"""Parallel Monte Carlo driver: R independent replications per configuration."""
from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .core import PutSpec
from .methods import AlgoConfig, run_fixed_alpha, run_once


def job_seed(base_seed: int, tags: dict, rep: int) -> np.random.SeedSequence:
    """Seed of one replication, from (base_seed, tags, rep) only."""
    key = zlib.crc32(repr(sorted(tags.items())).encode())
    return np.random.SeedSequence([base_seed, key, rep])


def _job(args):
    spec, cfg, seed, extra, tags = args
    if "fn" in extra:               # e.g. simgreeks.multi.label -> one dict row
        rows = [extra["fn"](spec, cfg, seed)]
    elif "alpha_stars" in extra:
        rows = run_fixed_alpha(spec, cfg, seed, extra["alpha_stars"], extra["modes"])
    else:
        rows = run_once(spec, cfg, seed, M0_list=extra.get("M0_list"),
                        nu_list=extra.get("nu_list"))
    for r in rows:
        r.update(tags)
    return rows


def run_configs(configs, reps: int, base_seed: int = 20191214,
                workers: int | None = None, progress: bool = True) -> pd.DataFrame:
    """Run ``reps`` replications of each configuration.

    ``configs`` is a list of dicts with keys ``spec`` (PutSpec), ``cfg``
    (AlgoConfig), optional ``M0_list``/``nu_list`` (or ``alpha_stars`` and
    ``modes`` for the fixed-alpha* diagnostic, or ``fn``, a picklable
    fn(spec, cfg, seed) -> dict used instead of run_once) and ``tags`` (dict of
    columns to attach to each output row, which also identifies the config).
    Seeds are derived from (base_seed, tags, rep) so results are reproducible
    and independent across configurations and replications.
    """
    jobs = []
    for c in configs:
        tags = c.get("tags", {})
        for rep in range(reps):
            seed = job_seed(base_seed, tags, rep)
            t = dict(tags, rep=rep)
            extra = {k: c[k] for k in ("M0_list", "nu_list", "alpha_stars", "modes", "fn")
                     if k in c}
            jobs.append((c["spec"], c["cfg"], seed, extra, t))
    workers = workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_job, jobs, chunksize=1)):
            rows.extend(out)
            if progress and (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} replications done", flush=True)
    return pd.DataFrame(rows)


__all__ = ["run_configs", "job_seed", "PutSpec", "AlgoConfig"]
