# Simulated Greeks for American Options — replication

Python replication of the experiments that showcase the method proposed in
Letourneau & Stentoft (2019), *Simulated Greeks for American Options*
(SSRN 3503889): prices, Deltas and Gammas of American puts from Least Squares
Monte Carlo (LSM) with **initial state dispersion (ISD)**, a **value-function
step at t = 1**, and the proposed **2-step method** that rescales the ISD to
an estimated optimal size α\*.

The code implements the paper's method **as printed**. Where the paper leaves
a detail open, the choice is listed under [Choices](#choices-the-paper-leaves-open).
No rule that is not in the paper is used to produce the main results.

Not replicated, on purpose: the binomial benchmark values (and Figure 1, which
is binomial-based) and the competing methods of Table 10 (PDM, LRM, MLSM).
Benchmark values are copied from the paper (`data/paper_tables.csv`) and used
only to flag significant differences. Table 10's 2-step row uses the same
settings as Table 5's σ = 20 %, T = 1 rows.

## Layout

| Path | Content |
|---|---|
| `simgreeks/core.py` | ISD (eq. 23, 25), GBM paths, LSM with naive and value-function payoffs, t = 0 regression (eq. 22) |
| `simgreeks/selector.py` | Optimal α\* selector, Appendix A.2: ROT pilot (A.2)–(A.3), local fit, plug-in (A.1) |
| `simgreeks/methods.py` | One replication of NAIVE, NAIVE-VF, TRUNC-VF and 2STEP-VF; the 2-step second step |
| `simgreeks/runner.py`, `report.py` | Parallel replications; paper-style tables with † flags |
| `experiments/run.py` | All experiments of Sections 3 and 4.1–4.3, plus the fixed-α\* diagnostic |
| `experiments/diagnostics/` | Diagnostics of the α\* selector (see below) |
| `data/paper_tables.csv` | Tables 1–9 of the paper (BM, estimates, std devs, † flags) |
| `results/` | Output tables (`*.md`), figures, per-replication estimates (`raw/`), diagnostics |

## Running

```bash
pip install -r requirements.txt
python experiments/run.py all --reps 100            # Figs 2-3, Tables 1-9 (≈ 80 min on 4 cores)
python experiments/run.py fixed_alpha --reps 100    # fixed-alpha* diagnostic (≈ 15 min)
python experiments/diagnostics/table9_selector_trace.py
python experiments/diagnostics/beta_scaling.py
python experiments/diagnostics/synthetic_selector_check.py
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
| Table 4: 2-step | same | 4 / 2 | Matches at α = 5; residual bias at α = 25; very noisy Greeks at α = 0.5 |
| Fig. 3: all methods across α | `figures/figure3.png` | – | Main message reproduced: only 2-step stays near the benchmark as α grows |
| Table 5: 27 options | `table5.md` | 15 / 1 of 81 | σ = 20 % matches; misses concentrated at σ = 10 % |
| Table 6: r and d | `table6.md` | 3 / 0 of 27 | Small price biases |
| Table 7: N and M0 | `table7.md` | 4 / 15 of 81 | Same pattern: M0 = 5 and small N are worst |
| Table 8: Mτ | `table8.md` | 5 / 0 of 27 | Little effect of Mτ, as in the paper |
| Table 9: ISD kernel, α\* target | `table9.md` | 0 / 0 of 45 | Estimates fine, but the paper's effects of kernel and target are much weaker |

In total 60 of 369 estimates are flagged (paper: 44). The NAIVE and NAIVE-VF
tables, which do not use α\*, match the paper. The differences in the other
tables come from the α\* selector.

## The α\* selector: why the results differ

The selector produces an α\* proportional to the initial ISD size α0, while the
paper's standard deviations imply an α\* near 4–6 for any α0 ≥ 5:

| initial α0 | 0.5 | 1 | 2.5 | 5 | 10 | 15 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|
| mean α\* (Section 3 runs) | 0.28 | 0.58 | 1.5 | 3.1 | 6.2 | 9.3 | 12.9 | 14.3 |

A too-wide α\* (≈ 13 at α0 = 25) crosses the early-exercise boundary and biases
the Greeks. A too-narrow one (≈ 0.3 at α0 = 0.5) makes them very noisy.

The diagnostics in `results/diagnostics/` and `results/fixed_alpha.md` show why:

1. **The rest of the method is right** (`fixed_alpha.md`). With α\* fixed at
   4–6, the 2-step method reproduces the paper's Table 4: no estimate is flagged,
   and the standard deviations are close to the paper's.
2. **No implementation bug** (`diagnostics/synthetic`). On a known curve (the
   Black-Scholes put) without noise, the selector recovers the true β₁₀ to within
   1–7 %, and with tiny noise α\* matches the oracle α\*.
3. **The curvature estimate is noise** (`diagnostics/beta_scaling`). On the real
   value-function data the estimated β₁₀ scales as α0^−9.9 (noise in a window
   ∝ α0 predicts −10), with a random sign up to α0 ≈ 10. Plugged into (A.1) this
   forces α\* ∝ α0. The measured slope is 0.997.
4. **The constants are not the cause** (`diagnostics/table9_trace`). The printed
   constants and Fan & Gijbels' constants give the same proportional α\*. The
   printed ones reproduce the *direction* of the paper's Table 9 (a smaller α\*
   when optimising for the price) but not its size.

In short, with N = 100,000 the (M0 + 1)-th derivative that (A.1) needs cannot be
estimated from the data, so the selector as described cannot produce the stable
α\* the paper's results imply. The paper does not report its α\* values, and the
implementation details that would explain the difference are not in the paper.

## Choices the paper leaves open

**Selector** (`simgreeks/selector.py`). Stated in the paper and implemented as
written: constants a, b as the (ν+1)-th diagonal elements of Q⁻¹Q\*Q⁻¹ and Q;
the global pilot of order M + 3 "using all data"; w0 "the indicator function";
(A.2)–(A.3); "use the ROT to locally fit a polynomial and estimate σ²(x0) and
β_{M+1}"; (A.1); one pass; OLS as a uniform kernel (eq. 21). Our choices:
- w0 is the indicator of |x − x0| ≤ 0.9 α0. The interval is not given, and
  ∫w0/f is infinite if w0 covers the whole Epanechnikov support.
- The local fit has order M + 1, the lowest order that identifies β_{M+1}. It
  is unweighted within |X − x0| ≤ h_ROT. The paper recommends "a weighted
  regression" but gives no weights.
- h_ROT is not clipped. For ν = 1 the printed (A.3) integral is zero, so h_ROT
  is infinite and the local fit uses all data.
- α\* is capped at 0.75 S0, only to keep rescaled prices positive. The cap
  never binds in the reported runs.
- `AlgoConfig(selector_reading="fg")` uses Fan & Gijbels' constants instead of
  the printed ones.

**Value function at t = 1.** Paths exercised at t1 take Z(t1); the others take
a continuation value fitted on all paths, so out-of-the-money paths also get
one. The exercise decision uses the standard in-the-money regression.

**2-step second step** (`second_step="refit_t1"`, Section 3.4, step 4).
X' = S0 + (α\*/α0)(X − S0), and each path is scaled by X'/X with the same
shocks. The pilot's stored exercise rules are re-applied at t_{J−1}, …, t_2
without re-estimation, the t1 regression is refitted on the rescaled paths, and
Y' = e^{−r·dt} max(Z, Ĉ). The fixed-α\* diagnostic also runs two alternatives
on the same paths. Re-running the whole LSM gives the same results. Reusing the
pilot's t1 curve is biased when α0 = 25.

**Regression bases.** Chebyshev polynomials for the LSM regressions; scaled
monomials centred at S0 for the t = 0 regression, so the coefficients are the
price and its derivatives.
