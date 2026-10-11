import json, sys
import numpy as np, pandas as pd
S = "/tmp/claude-0/-home-user-thesis-BU/a890be38-fc55-5a0f-bb6c-432be32154cf/scratchpad"
d = pd.read_csv(f"{S}/floor_run.csv")
refs = json.load(open(f"{S}/floor_refs.json"))
base = pd.read_csv("/home/user/options_pricing/results/raw/multi_check_american.csv")
base = base[(base["T"] == 1.0) & (base.sigma == 0.2) & (base.d == 0.0) & (base.rep < 50)]
def stat(x, ref):
    b, s = x.mean() - ref, x.std(ddof=1); t = b / (s / np.sqrt(len(x)))
    return f"{100*b/abs(ref):+.1f}{'†' if abs(t) > 2.576 else ''} / {100*s/abs(ref):.1f}"
for q in ("rho", "phi"):
    print(f"\n### {q}: bias % / noise % of one label vs each reference († = significant at 1 %), 50 paired reps\n")
    print("| option | reference | value | 0.02 sym (current) | 0.03 sym (before) | 0.03 one-sided (after) |")
    print("|---|---|---|---|---|---|")
    for r in (0.0, 0.06):
        for K in (36, 40, 44):
            R = refs[f"{r}_0.0_{K}"][q]
            g = d[(d.r == r) & (d.K == K)]
            b0 = base[(base.r == r) & (base.K == K)][q]
            for kind in (("central", "forward", "european") if r == 0 else ("central",)):
                print(f"| r={r:.0%} d=0% K={K} | {kind} | {R[kind]:+.3f} | {stat(b0, R[kind])} | "
                      f"{stat(g[g.variant == 'sym'][q], R[kind])} | {stat(g[g.variant == 'floor'][q], R[kind])} |")
    print("\nmean label value:")
    for r in (0.0, 0.06):
        for K in (36, 40, 44):
            g = d[(d.r == r) & (d.K == K)]
            print(f"  r={r:.0%} K={K}: sym {g[g.variant=='sym'][q].mean():+.3f}  floor {g[g.variant=='floor'][q].mean():+.3f}")
