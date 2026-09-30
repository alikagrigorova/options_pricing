"""Replication of Letourneau & Stentoft (2019), "Simulated Greeks for American
Options": prices, Deltas and Gammas of American puts from LSM with initial
state dispersion and the proposed 2-step method."""
import os as _os

# One BLAS thread per process: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
