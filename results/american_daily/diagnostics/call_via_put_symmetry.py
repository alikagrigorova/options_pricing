"""Call labels from put labels by put-call symmetry C(S, K, r, d) = P(K, S, d, r)
(diagnostic). Put label at spot K, strike S0, rate d, yield r; Greeks mapped by
homogeneity: P_y = (P - x P_x) / y, P_yy = x^2 P_xx / y^2."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
sys.path.insert(0, "/home/user/options_pricing")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, label
from simgreeks.runner import job_seed

def mapped(p, S, K, sfx=""):
    g = lambda q: p[q + sfx]
    P, D, G = g("price"), g("delta"), g("gamma")
    return dict(price=P, delta=(P - K * D) / S, gamma=K * K * G / S ** 2, vega=g("vega"),
                rho=g("phi"), phi=g("rho"), volga=g("volga"), vanna=(g("vega") - K * g("vanna")) / S)

def one(job):
    i, rep, cell = job
    T, s, r, d, K = cell
    sp = PutSpec(S0=float(K), K=40.0, sigma=s, r=d, d=r, T=T, steps_per_year=252, kind="put")
    p = label(sp, MultiConfig(vf_time=0.02), job_seed(8010, {"id": i}, rep))
    out = dict(id=i, rep=rep)
    for sfx, name in (("", ""), ("_oos", "_oos"), ("_ins", "_ins")):
        out.update({k + name: v for k, v in mapped(p, 40.0, float(K), sfx).items()})
    return out

if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",")]; R = int(sys.argv[2])
    ref = pd.read_parquet("/home/user/options_pricing/results/american_daily/raw/check_reference.parquet").set_index("id")
    jobs = [(i, k, (ref.loc[i, "T"], ref.loc[i, "sigma"], ref.loc[i, "r"], ref.loc[i, "d"], ref.loc[i, "K"])) for i in ids for k in range(R)]
    with ProcessPoolExecutor(4) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs)))
    df.to_csv(sys.argv[3], index=False)
    lab = pd.read_parquet("/home/user/options_pricing/results/american_daily/raw/check_labels.parquet")
    lab = lab[lab.variant == "D2-daily-vf50"]
    for i, g in df.groupby("id"):
        o = ref.loc[i]; c = lab[lab.id == i]
        print(f"id {i} (T {o['T']:g}, s {o.sigma:g}, K {o.K:g}):")
        for q in ("price", "delta", "gamma", "vega", "rho", "phi"):
            b = g[q].mean() - o[q]; se = g[q].std() / np.sqrt(len(g))
            bc = c[q].mean() - o[q]; sec = c[q].std() / np.sqrt(len(c))
            print(f"   {q}: ref {o[q]:+.4f} | symmetric put label ({len(g)}): {b:+.4f} ± {se:.4f} ({100*b/abs(o[q]):+.2f}%), oos {100*(g[q+'_oos'].mean()-o[q])/abs(o[q]):+.2f}% ins {100*(g[q+'_ins'].mean()-o[q])/abs(o[q]):+.2f}% | direct call label (100): {bc:+.4f} ± {sec:.4f} ({100*bc/abs(o[q]):+.2f}%)")
