# Simulated Greeks for American Options — replication

Python replication of the experiments that showcase the method proposed in
Letourneau & Stentoft (2019), *Simulated Greeks for American Options*
(SSRN 3503889): prices, Deltas and Gammas of American puts from Least Squares
Monte Carlo (LSM) with **initial state dispersion (ISD)**, a **value-function
step at t = 1**, and the proposed **2-step method** that rescales the ISD to
an estimated optimal size α\*.

Not replicated, on purpose: the binomial benchmark values (and Figure 1, which
is binomial-based) and the competing methods of Table 10 (PDM, LRM, MLSM).
Benchmark values are copied from the paper (`data/paper_tables.csv`) and
used only to flag significant differences. Table 10's 2-step row uses the
same settings as Table 5's σ = 20 %, T = 1 rows.

## Layout

| Path | Content |
|---|---|
| `simgreeks/core.py` | ISD (eq. 23, 25), GBM paths, LSM with naive and value-function payoffs, t = 0 regression (eq. 22) |
| `simgreeks/bandwidth.py` | Optimal α\* selector (Appendix A.2: ROT pilot + local plug-in of eq. A.1) |
| `simgreeks/methods.py` | One replication of NAIVE, NAIVE-VF, TRUNC-VF and 2STEP-VF |
| `simgreeks/runner.py`, `report.py` | Parallel replications; paper-style tables with † flags |
| `experiments/run.py` | All experiments |
| `data/paper_tables.csv` | Tables 1–9 of the paper (BM, estimates, std devs, † flags) |
| `results/*.md`, `results/figures/`, `results/raw/` | Output tables, figures, and per-replication estimates |

## Running

```bash
pip install -r requirements.txt
python experiments/run.py all --reps 100          # ≈ 75 min on 4 cores
python experiments/run.py section3 --reps 10      # quick look at Tables 1-4, Fig. 3
python -m pytest tests
```

All settings follow the paper: S0 = 40, K ∈ {36, 40, 44}, 50 exercise dates per
year, N = 100,000 paths, Mτ = M0 = 9, 100 independent replications. A † marks
|mean − BM| / (sd/√100) > 2.576.

## Results (100 replications)

| Experiment | Output | Significant at 1 % (ours / paper) | Verdict |
|---|---|---|---|
| Fig. 2: t = 0 regression data | `figures/figure2.png` | – | Reproduced |
| Table 1: naive | `section3_tables1-4.md` | 9 / 10 of 27 | Reproduced (α = 25 biases match to ~0.002) |
| Table 2: value function | same | 10 / 13 | Close at α = 5 and 25; noisier Greeks at α = 0.5 |
| Table 3: truncation | same | 10 / 3 | Bias at α = 25 only partly removed |
| Table 4: 2-step | same | 7 / 2 | Matches at α = 5; residual bias at α = 25; Greeks very noisy at α = 0.5 |
| Fig. 3: all methods across α | `figures/figure3.png` | – | Main message reproduced; ITM Gamma biased ≈ +0.006 for α ≥ 20 |
| Table 5: 27 options | `table5.md` | 15 / 1 of 81 | σ = 20 %: 0 flags, matches; σ = 40 %: 2 price flags; σ = 10 %: 13 flags |
| Table 6: r and d | `table6.md` | 6 / 0 of 27 | Small price biases (r = d = 0 is borderline in the paper too) |
| Table 7: N and M0 | `table7.md` | 6 / 15 of 81 | Same pattern: M0 = 5 and small N are worst |
| Table 8: Mτ | `table8.md` | 6 / 0 of 27 | Little effect of Mτ, as in the paper; some small biases |
| Table 9: ISD kernel, α\* target | `table9.md` | 0 / 0 of 45 | Estimates fine, but the paper's effects of kernel and target are absent |

## Main deviation: the optimal α\* selector

Everything except the α\* selector behaves as in the paper. The naive method
at α = 25 reproduces the paper's biases almost exactly, and the 2-step method
matches wherever the selected α\* is moderate (initial α = 5; σ = 20 % in
Table 5).

The selector is implemented as described in Appendix A.2. It needs the
(M0 + 1)-th derivative of the price function, i.e. the 10th for M0 = 9. With
100,000 paths this cannot be estimated from the data. The estimate is noise,
and noise makes α\* ≈ 0.5–0.6 × initial α:

| initial α | 0.5 | 5 | 10 | 25 |
|---|---|---|---|---|
| our mean α\* | 0.3 | 3.0 | 6.0 | 12.4 |

From the paper's standard deviations, its α\* seems to stay near 4–5 regardless
of the initial α. The consequences:
- α\* ≈ 12 crosses the early-exercise boundary: residual bias for ITM options,
  σ = 10 % options, and α = 25 in Tables 3–4.
- α\* ≈ 0.3 is too narrow: very noisy Greeks at α = 0.5.
- α\* barely depends on the targeted derivative or the ISD kernel, so Table 9's
  effects do not appear.

Variants tried, all with the same proportional behaviour: local pilot order
q vs q + 2, β from the global pilot, an uncapped ROT bandwidth, and pilot
orders 3–9.

### Choices where the paper is not explicit
- **Value function at t = 1:** paths exercised at t1 take Z(t1). The others
  take a continuation value fitted on *all* paths, so out-of-the-money paths
  also get one. The exercise decision itself uses the standard in-the-money
  regression. At the same α our value-function Gammas are ~35 % noisier than
  the paper's, so their implementation may differ here.
- **Bias constant in (A.1):** we use Fan & Gijbels' b = [S⁻¹c]_ν, not "the
  diagonal of Q" as written in the paper. Rule-of-thumb (A.2): standard Fan &
  Gijbels form with w0 = indicator of the ISD support. Local fit of order q
  (the lowest that identifies β_q). When M0 − ν is even, the next-order bias
  term is used.
- **2-step (default `second_step="refit_t1"`):** X' = S0 + (α\*/α0)(X − S0)
  and each path is scaled by X'/X (same shocks). The pilot's stored
  in-the-money exercise rules are re-applied at t_{J−1}..t_2 without
  re-estimation. The t_1 regression is refitted on all rescaled paths, and
  Y' = e^{−r·dt} max(Z, Ĉ_new). The Tables 4–9 results above were produced
  with the earlier `rerun` mode, which re-estimates the whole LSM on the
  rescaled paths; the diagnostic below shows the two give the same results.
- **Regression bases:** Chebyshev polynomials for the LSM regressions; scaled
  monomials centred at S0 for the t = 0 regression, so the coefficients are
  the price and its derivatives.

### Fixed-α\* diagnostic (`results/fixed_alpha.md`)
`python experiments/run.py fixed_alpha` bypasses the selector: α\* ∈ {4, 5, 6},
α0 ∈ {5, 10, 25}, K ∈ {36, 40, 44}, 100 runs, three second-step modes on the
same paths.
- `refit_t1` and `rerun`: no estimate significantly different from the
  benchmark (0 of 81 each), and the two agree to within ~0.002. With α\* = 6
  the standard deviations match the paper's Table 4 at α = 25 (e.g. ATM
  Gamma sd 0.0069 vs 0.0065).
- `reuse_t1` (pilot t_1 curve reused): clearly biased when α0 = 25, e.g. ATM
  Gamma +0.0076, ITM price +0.043.

So the second step is not the cause of the remaining differences: with a
sensible α\* the method reproduces the paper, and the gap comes from the α\*
selector.

### Possible next steps
1. Make the α\* selector land near 5–6 regardless of the initial α.
2. Try other readings of the value-function step.
3. Use a more robust curvature estimate for α\*, e.g. a low-order pilot or a
   large pilot simulation (our own extension, not the paper's).
4. Check the paper's supplementary document for implementation details.
