import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/options_pricing")
d = pd.read_csv(sys.argv[1])
ref = {}
for line in open("/home/user/options_pricing/results/multi_check_american.md"):
    p = [x.strip() for x in line.split("|")[1:-1]]
    if len(p) == 10 and p[0][:1].isdigit():
        key = (float(p[0]), float(p[1].rstrip("%")) / 100, float(p[2].rstrip("%")) / 100,
               float(p[3].rstrip("%")) / 100, int(p[4]))
        ref[key + (p[5],)] = float(p[6])
K = ["T", "sigma", "r", "d", "K"]
def name(c):
    T, s, r, dd, k = c
    return f"T={T:g} σ={s:.0%} r={r:.0%} d={dd:.0%} K={k}"
widths = sorted(d.width.unique())
for q in ("rho", "phi", "delta_r", "delta_d"):
    print(f"\n### {q}: bias % / noise % of one label (|t| > 2.58 marked †); last col: RMSE % by width")
    print("| option | ref | " + " | ".join(f"w={w:g}" for w in widths) + " |")
    print("|---|---|" + "---|" * len(widths))
    rm = {w: [] for w in widths}
    for c, g in d.groupby(K):
        r = ref[tuple(c) + (q,)]
        cells = []
        for w in widths:
            x = g[g.width == w][q].values
            b, s = x.mean() - r, x.std(ddof=1)
            t = b / (s / np.sqrt(len(x)))
            cells.append(f"{100*b/abs(r):+.1f} / {100*s/abs(r):.1f}{'†' if abs(t) > 2.576 else ''}")
            rm[w].append(np.sqrt(b**2 + s**2) / abs(r))
        print(f"| {name(c)} | {r:+.3f} | " + " | ".join(cells) + " |")
    print("| **median RMSE of one label** | | " + " | ".join(f"**{100*np.median(rm[w]):.1f}%**" for w in widths) + " |")
