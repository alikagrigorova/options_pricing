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
| `simgreeks/methods.py` | One replication of NAIVE, NAIVE-VF, TRUNC-VF and 2STEP-VF; the 2-step second step; the α\* rule |
| `simgreeks/reference.py` | Reference PDE pricer (Crank–Nicolson) for validation: price, Δ, Γ, Θ (not part of the paper) |
| `simgreeks/runner.py`, `report.py` | Parallel replications; paper-style tables with † flags |
| `experiments/run.py` | All experiments of Sections 3 and 4.1–4.3, plus the fixed-α\* diagnostic |
| `experiments/diagnostics/` | Diagnostics of the α\* selector (see below) |
| `experiments/theta_check.py` | Simulated Θ vs the reference pricer's finite-difference Θ |
| `simgreeks/multi.py` | Vega, Rho, dividend Rho, Vanna by multivariate ISD (not part of the paper) |
| `experiments/multi_check.py` | Multivariate Greeks vs closed form (European) and the reference (American) |
| `data/paper_tables.csv` | Tables 1–9 of the paper (BM, estimates, std devs, † flags) |
| `results/` | Output tables (`*.md`), figures, per-replication estimates (`raw/`), diagnostics |

## Running

```bash
pip install -r requirements.txt
python experiments/run.py all --reps 100            # Figs 2-3, Tables 1-9 (≈ 80 min on 4 cores)
python experiments/run.py fixed_alpha --reps 100    # fixed-alpha* diagnostic (≈ 15 min)
python experiments/run.py table5_heuristic          # placeholder alpha* on Table 5 (≈ 4 min on 8 cores)
python experiments/theta_check.py                   # Theta vs reference pricer (≈ 5 min on 8 cores)
python experiments/multi_check.py european american timing   # multivariate Greeks (≈ 40 min, 8 GB RAM)
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

## Beyond the paper: α\* rule, Theta, reference pricer

Everything in this section is **not part of the paper**. It is used for the
label-generation work and does not change the replication results above.

### α\* rule (`AlgoConfig.alpha_star_rule`)

| Rule | α\* | Status |
|---|---|---|
| `"selector"` (**default**) | Appendix A.2 selector, `simgreeks/selector.py` | The paper's method as printed; produces α\* ∝ α0 (see above) |
| `"fixed"` | `alpha_star_fixed` | Diagnostics |
| `"heuristic"` | `alpha_star_c · S0 · σ · √T` (c = 0.6 by default), capped at 0.75 S0 | **TEMPORARY PLACEHOLDER, not from the paper.** Stands in for the selector until it is fixed |

The default stays `"selector"` so that `experiments/run.py` keeps reproducing
the paper as printed; the heuristic must be requested explicitly. With c = 0.6,
α\* = 4.8 for S0 = 40, σ = 20 %, T = 1, inside the 4–6 range where the fixed-α\*
diagnostic reproduces Table 4.

Check on the Table 5 grid (`results/table5_heuristic.md`, 100 replications,
α0 = 10): **4 of 81** estimates differ from the benchmark at 1 % (paper: 1;
selector as printed: 15):

| T | σ | K | α\* | Flagged |
|---|---|---|---|---|
| 1 | 20 % | 36 | 4.8 | Delta, +0.0013 (t = 2.6; borderline) |
| 2 | 40 % | 36 | 13.6 | Price +0.0084 (t = 3.5), Gamma −0.0009 (t = −2.8, −5 %) |
| 2 | 40 % | 44 | 13.6 | Price +0.0096 (t = 2.6) |

The σ = 10 % cells that the selector got wrong are now unbiased. But the
heuristic's scaling in σ√T is too steep compared with the paper's implied α\*:
- at σ√T = 0.07 (α\* = 1.7) the Gamma standard deviations are 2–3× the paper's;
- at σ√T = 0.57 (α\* = 13.6) the price is biased and the Gamma standard
  deviations are 2–3× smaller than the paper's.

### Theta (`core.theta_pde`)

Theta = ∂V/∂t (calendar time, per year) from the Black-Scholes PDE:
Θ = rV − (r − d) S Δ − ½ σ² S² Γ in the continuation region, Θ = 0 in the
exercise region. S0 is classified as in the exercise region when the estimated
price is at or below K − S0. Every result row now has `theta` and `ex_region`.

Check (`experiments/theta_check.py`, `results/theta_check.md`): the Table 5
grid plus the (r, d) = (0, 0) and (6 %, 6 %) options of Table 6, 100
replications, heuristic α\*. **1 of 33** cells differs from the reference at 1 %
(T = 2, σ = 10 %, K = 40, t = 2.9). The identity reproduces the reference's
finite-difference Θ to 3·10⁻⁵ when fed the reference's own V, Δ, Γ. The
simulated Θ is noisy because the Gamma noise is multiplied by ½σ²S²: per
replication its sd is 25–50 % of |Θ| at σ = 20 %. In the exercise region
(K = 44, σ = 10 %) Θ = 0, while the Bermudan continuation value has
Θ ≈ +2.5 per year.

### Reference pricer (`simgreeks/reference.py`)

Crank–Nicolson in ln S with Rannacher smoothing, on the same exercise grid as
the LSM (no exercise at t0), plus American and European variants. Θ is a
central difference in calendar time across t0. Accuracy:
- European: matches the closed form to 1e-5 in price, Δ, Γ and Θ;
- Bermudan: matches the paper's binomial benchmarks on all 36 Table 5–6 options
  to ≤ 1.3e-4, the level of the benchmarks' 4-decimal rounding;
- halving the grid steps changes every output by < 3e-4;
- about 0.1 s per option.

### Vega, Rho, dividend Rho, Vanna: multivariate ISD (`simgreeks/multi.py`)

Work in progress. Each label combines three independent runs that disperse
(S, σ), (S, r) and (S, d), with 100,000 paths each:
- every path keeps its own σ, r, d for its whole life,
  S(t) = S0 exp((r − d − σ²/2) t + σ W(t)), and is discounted with its own r;
- the LSM regressions use a basis in S and the dispersed parameter;
- the t = 0 regression is a multivariate Taylor polynomial in
  (S − S0, θ − θ0), and each Greek is a coefficient times a!/h^a;
- a European control variate is on by default: only the early-exercise
  premium is regressed, and the closed-form European Greeks are added back;
- price, Δ and Γ are averaged over the three runs, and Θ comes from the PDE
  identity.

Dispersion sizes are placeholders: α_S = 0.6 S0 σ √T, α_σ = 0.25 σ,
α_r = α_d = 0.02. A single joint 4-D run was tried first. It was noisier and
biased Vega for in-the-money options (−9 % at K = 44), so it was replaced by
the three 2-D runs.

Checks (`experiments/multi_check.py`, 100 replications per option):
- **European**, no early exercise, no control variate, 6 options: 0 of 66
  Greek estimates biased.
- **American**, 33 options (the Table 5 grid plus Table 6's (r, d)), against
  the reference with σ, r, d bumped:

| Greek | biased options | median bias | noise of one label |
|---|---|---|---|
| Vega | 0 / 33 | 0.3 % | 6 % |
| Rho, dividend Rho | 2–3 / 33 | 2 % | 18–22 % |
| Vanna | 2 / 33 | 6 % | 77 % |
| ∂Δ/∂r, ∂Δ/∂d | 1–2 / 33 | 10–17 % | ≈ 190 % |
| Θ | 1 / 33 | 1.7 % | 16 % |
| Γ | 4 / 33 | 0.8 % | 8 % |
| Δ | 7 / 33 | 0.1 % | 1 % |
| Price | 15 / 33 | 0.1 % | < 1 % |
| Volga | 21 / 33 | 51 % | – (not usable) |

Open issues:
- **Price biases** of +0.005 to +0.014 at r = d = 0. They appear in every run,
  and also in the 1-D method; the paper reports the same case in Table 6. A
  simpler exercise rule (`MultiConfig(weight=9)`) is not a fix
  (`results/diagnostics/exercise_basis/summary.md`). It removes the bias only
  in the (S, σ) run; the (S, r) and (S, d) runs stay biased, which suggests a
  kink at r = 0. It also biases Vega and the control options, so weight 3
  stays the default. Untested candidate: an exercise rule fitted on an
  independent set of paths, which removes in-sample bias.
- **Exercise region** (K = 44, σ = 10 %): the windows straddle the exercise
  boundary, which biases Γ and Vanna. `american_t0=True` reports American
  values there instead (still to decide).
- **Runtime:** about 0.6 / 1.1 / 2.0 s per label at T = 0.5 / 1 / 2 on one
  core, and about 300 MB per worker. With 8 workers on an 8 GB machine the
  American check swapped and took 40 min.

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
