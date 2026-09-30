"""Summaries in the format of the paper's tables: mean estimate, standard
deviation across replications, and a dagger when the mean differs from the
benchmark at the 1% level (|mean - BM| / (sd / sqrt(R)) > 2.576)."""
from __future__ import annotations

import numpy as np
import pandas as pd

Z99 = 2.576
QUANTS = ("price", "delta", "gamma")


def summary_table(df: pd.DataFrame, keys, paper: pd.DataFrame, paper_keys) -> pd.DataFrame:
    """Aggregate replications and merge with the paper's BM and estimates."""
    agg = df.groupby(keys)[list(QUANTS)].agg(["mean", "std", "count"])
    agg.columns = [f"{q}_{s}" for q, s in agg.columns]
    agg = agg.reset_index()
    p = paper.copy()
    for k in paper_keys:
        if k == "K":
            p[k] = p[k].astype(int)
    out = agg.merge(p, left_on=keys, right_on=paper_keys, how="left")
    for q in QUANTS:
        se = out[f"{q}_std"] / np.sqrt(out[f"{q}_count"])
        out[f"{q}_t"] = (out[f"{q}_mean"] - out[f"{q}_bm"].astype(float)) / se
        out[f"{q}_sig"] = out[f"{q}_t"].abs() > Z99
    out.attrs["keys"] = list(keys)
    return out


def _fmt(est, sd, sig):
    if pd.isna(est):
        return "–", ""
    return f"{est:.4f}", f"({sd:.4f}){'†' if sig else ''}"


def to_markdown(tab: pd.DataFrame, keys) -> str:
    head = keys + ["Q", "BM", "Estim", "StDev", "Paper Estim", "Paper StDev"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in tab.iterrows():
        for i, q in enumerate(QUANTS):
            kv = [f"{r[k]:g}" if isinstance(r[k], (int, float, np.number)) else str(r[k])
                  for k in keys] if i == 0 else [""] * len(keys)
            est, sd = _fmt(r[f"{q}_mean"], r[f"{q}_std"], r[f"{q}_sig"])
            bm = r.get(f"{q}_bm")
            pe = r.get(f"{q}_paper")
            psd = r.get(f"{q}_paper_sd")
            psig = r.get(f"{q}_paper_sig")
            p_sd = "" if pd.isna(psd) else f"({float(psd):.4f}){'†' if psig == 1 else ''}"
            lines.append("| " + " | ".join(kv + [
                q, "" if pd.isna(bm) else f"{float(bm):.4f}", est, sd,
                "" if pd.isna(pe) else f"{float(pe):.4f}", p_sd]) + " |")
    return "\n".join(lines)


def write_markdown(path, title, sections):
    """sections: list of (subtitle, summary_table_df, extra_text)."""
    parts = [f"# {title}\n",
             "Estim/StDev: this replication (mean and standard deviation across "
             "independent replications). BM, Paper Estim and Paper StDev are copied "
             "from the paper. † = differs from BM at the 1% level.\n"]
    for sub, tab, extra in sections:
        keys = tab.attrs["keys"]
        n_sig = int(sum(tab[f"{q}_sig"].sum() for q in QUANTS))
        n_tot = int(sum(tab[f"{q}_mean"].notna().sum() for q in QUANTS))
        parts.append(f"## {sub}\n")
        parts.append(to_markdown(tab, keys))
        parts.append(f"\n\nSignificant at 1%: {n_sig} of {n_tot} estimates.{extra}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
