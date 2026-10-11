"""Signed bias +- se per option (200 reps), and paired differences, for the
r-window and d-window variants. Labels assembled from the stored group runs."""
import json
import numpy as np, pandas as pd
S = "/tmp/claude-0/-home-user-thesis-BU/a890be38-fc55-5a0f-bb6c-432be32154cf/scratchpad"
d = pd.read_csv(f"{S}/confirm_run.csv")
refs = json.load(open(f"{S}/kink_refs.json"))
K = ["r", "d", "K"]
def piv(grp, var):
    x = d[(d.group == grp) & (d.variant == var)].set_index(K + ["rep"])
    return x
sg, rc, rs, dc = piv("sigma", "cur"), piv("r", "cur"), piv("r", "shrink"), piv("d", "cur")
dr0, dd0 = piv("d", "shrink_r0"), piv("d", "shrink_d0")
def fmt(x, ref):
    n = x.notna().sum(); b = x.mean() - ref; se = x.std(ddof=1) / np.sqrt(n); sd = x.std(ddof=1)
    z = b / se
    return f"{100*b/abs(ref):+.2f} ± {100*se/abs(ref):.2f}{'†' if abs(z) > 2.576 else ''} ({100*sd/abs(ref):.1f})"
def pdiff(a, b, ref):
    x = (b - a).dropna(); m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return f"{100*m/abs(ref):+.2f} ± {100*se/abs(ref):.2f}{'†' if abs(m/se) > 2.576 else ''}"
print(f"Reps per option: {d[(d.group=='r')&(d.variant=='cur')].groupby(K).size().min()}. "
      "Cells: signed bias ± se of the mean, % of the reference († = |bias/se| > 2.576), "
      "(noise of one label, %). Paired diff: mean of (variant − current) over the same "
      "replications ± se. Reference: central bump (h ≤ 0.002 < r0).\n")
opts = sorted(set(map(tuple, d[K].drop_duplicates().values)))
# labels: price, delta, gamma averaged over the three runs
def lab(rrun, drun, q):
    return (sg[q] + rrun[q].reindex(sg.index) + drun[q].reindex(sg.index)) / 3
print("## Rho ((S, r) run)\n")
print("| r0 | d0 | K | ref | current α_r = 0.03 | shrink α_r = min(0.03, r0) | paired diff |")
print("|---|---|---|---|---|---|---|")
for o in opts:
    ref = refs[f"{o[0]}_{o[1]}_{int(o[2])}"]["rho"]["central"]
    a, b = rc["rho"].xs(o, level=K), rs["rho"].xs(o, level=K)
    print(f"| {o[0]:.1%} | {o[1]:.0%} | {int(o[2])} | {ref:+.3f} | {fmt(a, ref)} | {fmt(b, ref)} | {pdiff(a, b, ref)} |")
print("\n## Phi ((S, d) run)\n")
print("| r0 | d0 | K | ref | current α_d = 0.03 | α_d = min(0.03, r0) | α_d = min(0.03, d0) | paired diff (r0 window) |")
print("|---|---|---|---|---|---|---|---|")
for o in opts:
    ref = refs[f"{o[0]}_{o[1]}_{int(o[2])}"]["phi"]["central"]
    a, b = dc["phi"].xs(o, level=K), dr0["phi"].xs(o, level=K)
    c = dd0["phi"].xs(o, level=K) if o[1] > 0 else None
    print(f"| {o[0]:.1%} | {o[1]:.0%} | {int(o[2])} | {ref:+.3f} | {fmt(a, ref)} | {fmt(b, ref)} | "
          f"{fmt(c, ref) if c is not None else '– (d0 = 0)'} | {pdiff(a, b, ref)} |")
for q in ("price", "delta", "gamma"):
    print(f"\n## {q} (label = mean of the three runs)\n")
    print("| r0 | d0 | K | ref | current | shrink r only | shrink r and d (r0) | paired diff (r and d) |")
    print("|---|---|---|---|---|---|---|---|")
    cur, sr, srd = lab(rc, dc, q), lab(rs, dc, q), lab(rs, dr0, q)
    for o in opts:
        ref = refs[f"{o[0]}_{o[1]}_{int(o[2])}"][q]["central"]
        a, b, c = cur.xs(o, level=K), sr.xs(o, level=K), srd.xs(o, level=K)
        print(f"| {o[0]:.1%} | {o[1]:.0%} | {int(o[2])} | {ref:+.4f} | {fmt(a, ref)} | {fmt(b, ref)} | {fmt(c, ref)} | {pdiff(a, c, ref)} |")
