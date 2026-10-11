import json
import numpy as np, pandas as pd
S = "/tmp/claude-0/-home-user-thesis-BU/a890be38-fc55-5a0f-bb6c-432be32154cf/scratchpad"
d = pd.read_csv(f"{S}/kink_run.csv")
refs = json.load(open(f"{S}/kink_refs.json"))
V = ["current", "kink", "shrink"]
def cell(x, ref):
    b, s = x.mean() - ref, x.std(ddof=1); t = b / (s / np.sqrt(len(x)))
    return f"{100*b/abs(ref):+.1f}{'†' if abs(t) > 2.576 else ''} / {100*s/abs(ref):.1f}", b / abs(ref), s / abs(ref)
print(f"{d.groupby('variant').rep.count().to_dict()} rows; 50 paired reps per option. Cells: bias % / noise % of one label "
      "(† = |t| > 2.576). Price, Delta, Gamma: average of the three runs (only the (S, r) run differs between variants). "
      "Phi comes from the (S, d) run, which is the same in all variants.\n")
summ = {}
for q, kinds in (("price", ("central",)), ("delta", ("central",)), ("gamma", ("central",)),
                 ("rho", ("central", "forward")), ("phi", ("central", "forward"))):
    print(f"\n### {q}\n")
    print("| r0 | d0 | K | reference | value | " + " | ".join(V) + " |")
    print("|---|---|---|---|---|" + "---|" * len(V))
    for (r, dd, K), g in d.groupby(["r", "d", "K"]):
        for kind in kinds:
            ref = refs[f"{r}_{dd}_{K}"][q][kind]
            cs = []
            for v in V:
                txt, b, s = cell(g[g.variant == v][q], ref)
                cs.append(txt)
                summ.setdefault((q, kind, r, v), []).append((abs(b), s))
            print(f"| {r:.1%} | {dd:.0%} | {K} | {kind} | {ref:+.4f} | " + " | ".join(cs) + " |")
print("\n## Summary: median abs(bias) % / median noise % over the 6 options (d0 x K) at each r0\n")
for q, kind in (("price", "central"), ("delta", "central"), ("gamma", "central"), ("rho", "central"),
                ("rho", "forward"), ("phi", "central"), ("phi", "forward")):
    print(f"\n**{q} vs {kind}**\n")
    print("| r0 | " + " | ".join(V) + " |"); print("|---|" + "---|" * len(V))
    for r in sorted(d.r.unique()):
        cs = []
        for v in V:
            a = np.array(summ[(q, kind, r, v)])
            cs.append(f"{100*np.median(a[:,0]):.2f} / {100*np.median(a[:,1]):.2f}")
        print(f"| {r:.1%} | " + " | ".join(cs) + " |")
print("\nmean alpha_r by variant and r0:", d.groupby(["variant", "r"]).alpha_r.mean().unstack().round(4).to_dict("index"))
