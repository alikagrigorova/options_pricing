# Simulated Greeks for American Options — replication

Python replication of the experiments that showcase the method proposed in
Letourneau & Stentoft (2019), *Simulated Greeks for American Options*
(SSRN 3503889): prices, Deltas and Gammas of American puts from Least Squares
Monte Carlo (LSM) with **initial state dispersion (ISD)**, a **value-function
step at t = 1**, and the proposed **2-step method** that rescales the ISD to
an estimated optimal size α\*.

The code implements the paper's method as printed. Where the paper leaves a
detail open, the choice is listed under [Choices](#choices-the-paper-leaves-open);
the most important one is the order M in the α\* selector, see
[The selector order M](#the-selector-order-m). No rule that is not in the
paper is used to produce the replication results.

The [second part](#beyond-the-paper-labels-for-a-neural-network-surrogate) extends the method to
the Greeks with respect to σ, r and d, to generate training labels
(price, Δ, Γ, Θ, Vega, Volga, Vanna, Rho, Phi) for a neural-network
surrogate. It is not part of the paper.

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
| `simgreeks/reference.py` | Reference PDE pricer (Crank–Nicolson) for validation: price, Δ, Γ, Θ and bumped σ, r, d Greeks (not part of the paper) |
| `simgreeks/runner.py`, `report.py` | Parallel replications; paper-style tables with † flags |
| `experiments/run.py` | All experiments of Sections 3 and 4.1–4.3, plus the fixed-α\* diagnostic |
| `experiments/diagnostics/` | Diagnostics of the α\* selector's M = M0 reading (see below), of the multivariate design and of the label price's bias |
| `experiments/theta_check.py` | Simulated Θ vs the reference pricer's finite-difference Θ |
| `simgreeks/multi.py` | Label generator: price and Δ, Γ, Θ, Vega, Volga, Vanna, Rho, Phi (∂P/∂d, dividend Rho) by multivariate ISD (not part of the paper) |
| `experiments/multi_check.py` | Labels vs closed form (European) and the reference pricer (American); resumable |
| `experiments/generate_labels.py` | A labelled data set: options drawn over K, σ, r, d, T; progress every 1 %; resumable; `--exact` adds the reference values |
| `experiments/analyze_labels.py` | Sanity checks of a label CSV and, with reference values, the bias and noise of each Greek and where the bias is |
| `data/paper_tables.csv` | Tables 1–9 of the paper (BM, estimates, std devs, † flags) |
| `results/` | Output tables (`*.md`), figures, per-replication estimates (`raw/`), diagnostics |

## Running

```bash
pip install -r requirements.txt
python experiments/run.py all --reps 100            # Figs 2-3, Tables 1-9 (≈ 80 min on 4 cores)
python experiments/run.py fixed_alpha --reps 100    # fixed-alpha* diagnostic (≈ 15 min)
python experiments/run.py table5_heuristic          # placeholder alpha* on Table 5 (≈ 4 min on 8 cores)
python experiments/theta_check.py                   # Theta vs reference pricer (≈ 5 min on 8 cores)
python experiments/multi_check.py american european timing   # labels (≈ 2.5 h on 4 cores; resumes if interrupted)
python experiments/generate_labels.py --n 1000 --workers 8 --exact --out results/labels_1000.csv   # ≈ 35 min on an M3
python experiments/analyze_labels.py results/labels_1000.csv --plots results/labels_1000_analysis
python experiments/diagnostics/multi_design_check.py          # exercise rule and σ design of the labels
python experiments/diagnostics/price_bias_check.py            # why the label price is low (≈ 12 min on 4 cores)
python experiments/diagnostics/table9_selector_trace.py
python experiments/diagnostics/beta_scaling.py
python experiments/diagnostics/synthetic_selector_check.py
python -m pytest tests
```

All settings follow the paper: S0 = 40, K ∈ {36, 40, 44}, 50 exercise dates per
year, N = 100,000 paths, Mτ = M0 = 9, 100 independent replications. A † marks
|mean − BM| / (sd/√100) > 2.576.

## Results (100 replications)

Selector order M = ν + 1 (the default, see [below](#the-selector-order-m)).

| Experiment | Output | Significant at 1 % (ours / paper) | Verdict |
|---|---|---|---|
| Fig. 2: t = 0 regression data | `figures/figure2.png` | – | Reproduced |
| Table 1: naive | `section3_tables1-4.md` | 9 / 10 of 27 | Reproduced (α = 25 biases match to ~0.002) |
| Table 2: value function | same | 10 / 13 | Close at α = 5 and 25; noisier Greeks at α = 0.5 |
| Table 3: truncation | same | 10 / 3 | Bias at α = 25 not removed (see gaps) |
| Table 4: 2-step | same | 2 / 2 | Reproduced at α = 5 and 25; very noisy Greeks at α = 0.5 |
| Fig. 3: all methods across α | `figures/figure3.png` | – | Reproduced: only 2-step stays near the benchmark as α grows |
| Table 5: 27 options | `table5.md` | 1 / 1 of 81 | Reproduced |
| Table 6: r and d | `table6.md` | 4 / 0 of 27 | Small price biases |
| Table 7: N and M0 | `table7.md` | 2 / 15 of 81 | Reproduced; fewer flags than the paper |
| Table 8: Mτ | `table8.md` | 4 / 0 of 27 | Little effect of Mτ, as in the paper |
| Table 9: ISD kernel, α\* target | `table9.md` | 0 / 0 of 45 | Effect of the target ν reproduced in direction; kernel effect absent |

In total 42 of 369 estimates are flagged (paper: 44).

Mean α\* selected for the Gamma (Section 3 runs, K = 40):

| initial α0 | 0.5 | 1 | 2.5 | 5 | 10 | 15 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|
| mean α\* | 0.28 | 0.62 | 1.7 | 3.3 | 4.9 | 5.6 | 6.6 | 7.8 |

For α0 ≥ 5 the selector settles near 4–7, which is where the fixed-α\*
diagnostic (`fixed_alpha.md`) shows the 2-step method reproduces Table 4.

### Remaining gaps

- **α0 = 0.5.** The pilot ISD is too narrow to estimate curvature; α\* ≈ 0.3
  and the 2-step Greeks are very noisy.
- **Table 3 at α0 = 25.** TRUNC-VF stays biased (price ≈ 0.890 vs 0.917): the
  truncated paths keep the value function fitted on the wide pilot ISD.
- **Table 9, ISD kernel.** The paper's lower variance with the uniform ISD
  does not appear.
- **Selection bias of the 2-step Gamma.** α\* is chosen on the same paths the
  Greeks are estimated from, so it is smallest when the noise makes the data look
  curved, which is when the Gamma estimate has that noise. Pooled over Tables 5–9
  (500 replications per strike, same option), the Gamma is biased by +2.6 % at
  K = 44 (t = 3.1), −1.6 % at K = 40 and −1.1 % at K = 36; in the quarter of
  replications with the smallest α\* the K = 44 bias is +12 %. The paper's
  estimates show about +1 % at K = 44.
- **Table 9, ν = 0.** α\* ≈ 0.34 is much smaller than the paper's results
  imply; Greek standard deviations are about 10× the paper's.

## The selector order M

Appendix A.2 uses a polynomial order M in (A.1)–(A.3) but does not define it.
The paper does **not** say M = ν + 1; this is our interpretation.

- **M = M0 = 9** (`AlgoConfig(selector_order="M0")`). (A.1) then needs the
  10th derivative of the price. With N = 100,000 that estimate is simulation
  noise: β̂₁₀ scales as α0^−9.9, which forces α\* ∝ α0 (measured slope 0.997).
  The selector then returns α\* ≈ 13 at α0 = 25, and the results differ from
  the paper (60 flags, 15 in Table 5 alone).
- **M = ν + 1** (default). This is the standard local-polynomial order for the
  ν-th derivative in Fan & Gijbels (1996). (A.1) then needs the (ν + 2)-th
  derivative, which the data can identify. It is used only in the selector; the
  t = 0 regression keeps order M0.

The diagnostics in `experiments/diagnostics/` and `results/diagnostics/` use
the M0 reading on purpose; they document why it fails:

1. `fixed_alpha.md`: with α\* fixed at 4–6, the 2-step method reproduces
   Table 4, so the rest of the method is right.
2. `diagnostics/synthetic`: on a known noiseless curve the selector recovers
   β₁₀, so there is no implementation bug.
3. `diagnostics/beta_scaling`: on the real data β̂₁₀ is noise.
4. `diagnostics/table9_trace`: the printed constants and Fan & Gijbels'
   constants give the same proportional α\* under the M0 reading.

## Beyond the paper: labels for a neural-network surrogate

Everything in this section is **not part of the paper**. It is used for the
label-generation work and does not change the replication results above.

### α\* rule (`AlgoConfig.alpha_star_rule`)

| Rule | α\* | Status |
|---|---|---|
| `"selector"` (**default**) | Appendix A.2 selector, `simgreeks/selector.py`, order `selector_order` | The paper's method; reproduces Table 5 with M = ν + 1 (above) |
| `"fixed"` | `alpha_star_fixed` | Diagnostics |
| `"heuristic"` | `alpha_star_c · S0 · σ · √T` (c = 0.6 by default), capped at 0.75 S0 | Placeholder from before the selector was fixed; kept for the diagnostics below |

Check of the heuristic on the Table 5 grid (`results/table5_heuristic.md`, 100
replications, α0 = 10): **4 of 81** estimates differ from the benchmark at 1 %
(paper: 1; selector with M = M0: 15; selector with M = ν + 1: 1):

| T | σ | K | α\* | Flagged |
|---|---|---|---|---|
| 1 | 20 % | 36 | 4.8 | Delta, +0.0013 (t = 2.6; borderline) |
| 2 | 40 % | 36 | 13.6 | Price +0.0084 (t = 3.5), Gamma −0.0009 (t = −2.8, −5 %) |
| 2 | 40 % | 44 | 13.6 | Price +0.0096 (t = 2.6) |

The heuristic's scaling in σ√T is too steep compared with the selector's α\*:
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
central difference in calendar time across t0. `put_fd_greeks` adds Vega,
Volga, Rho, Phi, Vanna, ∂Δ/∂r, ∂Δ/∂d and Vera by bumping σ, r and d
on the same x-grid, with Richardson extrapolation. Accuracy:
- European: matches the closed form to 1e-5 in price, Δ, Γ and Θ;
- Bermudan: matches the paper's binomial benchmarks on all 36 Table 5–6 options
  to ≤ 1.3e-4, the level of the benchmarks' 4-decimal rounding;
- halving the grid steps changes every output by < 3e-4;
- bumped Greeks: match the European closed forms to 1e-3; the Bermudan Volga
  moves by up to 0.3 under grid refinement at the money (Volga ≈ 5.7 there);
- about 0.1 s per option, about 1.5 s with all bumped Greeks at T = 1.

### Multivariate Greeks: the label generator (`simgreeks/multi.py`)

One call gives one label:

```python
from simgreeks.core import PutSpec
from simgreeks.multi import MultiConfig, label
label(PutSpec(S0=40, K=40, sigma=0.2, r=0.06, d=0.0, T=1.0), MultiConfig(), seed=1)
# -> price, delta, gamma, theta, vega, volga, vanna, rho, phi (dP/dd, dividend Rho),
#    delta_r, delta_d, ex_region, alpha_S, alpha_sigma, alpha_r, alpha_d
```

Only the moneyness S0/K matters: a label at (λS0, λK) equals the label at
(S0, K) with price, Θ, Vega, Volga, Rho and Phi multiplied by λ, Γ
divided by λ and the other outputs unchanged, to about 1e-10 with the same seed.
The checks below use S0 = 40 and K = 36, 40, 44, i.e. S0/K from 0.91 to 1.11.

The paper's method is extended from S to σ, r and d: each path starts from its
own (S_n, σ_n, r_n, d_n) and keeps its parameters for its whole life,
S_n(t) = S_n exp((r_n − d_n − σ_n²/2) t + σ_n W_n(t)), discounted with its own
r_n. The parameters are part of the state, so the LSM regressions use a basis
in S and the dispersed parameters, and the t = 0 regression is a multivariate
Taylor polynomial in (S − S0, σ − σ0, r − r0, d − d0): the coefficient c of
x_S^i x_σ^a x_r^b x_d^e gives ∂^(i+a+b+e)P / ∂S^i ∂σ^a ∂r^b ∂d^e = i! a! b! e! c.

A label combines three independent runs, (S, σ), (S, r) and (S, d), each in
two phases on independent paths:

1. **Pilot** (100,000 paths). S and σ are dispersed widely (α0_S = 0.25 S0, the
   paper's α = 10 at S0 = 40; α0_σ = 0.6 σ), r or d by 1.3 × 0.02. The paper's
   selector (M = ν + 1) on the pilot's partial residuals chooses α_S\*
   (target: Gamma) and α_σ\* (target: Volga). The pilot's starting points are
   then rescaled to the chosen widths, as in the paper's 2-step method (exact,
   since paths are rebuilt from W), and the exercise rule is fitted on them.
2. **Main** (100,000 paths) at the chosen widths: LSM with the pilot's exercise
   rule at every date including t1, a European control variate (only the
   early-exercise premium is regressed; the closed-form European Greeks are
   added back), the value-function label, and the t = 0 Taylor regression.

Price, Δ and Γ are averaged over the three runs; Vega, Volga and Vanna come from
the (S, σ) run, Rho from (S, r), Phi from (S, d); Θ follows from the
PDE identity. When exercising at t0 is optimal (the estimated continuation
value is at or below K − S0), the label is exact: price K − S0, Δ = −1, all
other Greeks 0. Vera (∂²P/∂σ∂r) is available by adding the group (S, σ, r),
at about 40 % more time per label.

Each design choice fixes a bias found in the checks:

| Choice | Bias it removes | Evidence |
|---|---|---|
| Exercise rule fitted on independent (pilot) paths | An in-sample rule has foresight, largest where the regression's leverage is largest (the edges of the parameter box): Volga +50 % median, 21 of 33 options flagged | `results/multi_check_american_v1.md`; `experiments/diagnostics/multi_design_check.py` |
| Uniform dispersion of σ, r, d; pilot box 1.3× wider | The parameters do not diffuse, so the Epanechnikov density, which vanishes at the edges, leaves the rule poorly fitted there | same diagnostic |
| Widths chosen on the pilot paths | Choosing α\* on the paths that give the Greeks biases them (+2.6 % Gamma at K = 44 in the paper's method, see above) | pooled Tables 5–9 |
| Widths from the selector instead of fixed formulas | Fixed α_S = 0.6 S0 σ √T: Gamma −4.4 % and Theta +54 % at σ = 10 %, T = 2 | quick checks |
| Rule refitted on the rescaled pilot | A rule fitted on the wide pilot is poor where the main paths are: price −0.4 % at σ = 10 % | quick checks |
| Exercise at t1 by the rule, not max(Z, Ĉ) | The max turns the fit's estimation error into an upward bias, convex in σ: Volga +34 % at K = 44, σ = 20 %, T = 1 (−> +1 %) | quick checks |

Checks (`experiments/multi_check.py`, 100 replications per option; † counts
options where |mean − ref| / (sd/√100) > 2.576):

- **European** (no early exercise, no control variate), 6 options against the
  closed form: **0 of 66** estimates flagged (`results/multi_check_european.md`).
- **American**, 33 options (the Table 5 grid plus Table 6's (r, d)) against the
  reference pricer with σ, r, d bumped (`results/multi_check_american.md`; the
  first design, fixed widths and max(Z, Ĉ) at t1, in `_v1`):

| Greek | flagged (v1) | median bias | noise of one label | flags |
|---|---|---|---|---|
| Price | 24 (22) | 0.1 % | < 1 % | all low: an estimated exercise rule is suboptimal, so the price is a lower bound (see below) |
| Δ | 9 (7) | 0.2 % | 1 % | out of the money and r = d options, +0.2 to +1.5 %: the price's lower bound (see below) |
| Γ | 3 (5) | 1.3 % | 10 % | K = 36, σ = 20 % (−1.4 %, −2.4 %); σ = 10 %, K = 40, T = 1 (−2.5 %) |
| Θ | 3 (4) | 1.4 % | 18 % | the same options as Γ |
| Vega | 1 (0) | 0.4 % | 5 % | r = 0 (−0.7 %) |
| Volga | **0** (4) | 5.5 % | 161 % | – |
| Vanna | **0** (1) | 5.1 % | 72 % | – |
| Rho | 4 (2) | 1.2 % | 14 % | three at r = 0 (+2 to +3 %); K = 44, σ = 40 %, T = 1 (−5.8 %) |
| Phi | 3 (2) | 1.1 % | 18 % | all at r = 0 (−2 to −4 %) |

Seconds per label on one core (`results/multi_check_timing.md`, Apple M3, one
process): 1.7, 3.3 and 6.2 at T = 0.5, 1 and 2, about proportional to the 50 T
exercise dates. The first design took 0.7, 1.4 and 2.5, so the pilot, the rule
refit and the selection cost about 2.4×. Peak memory is 0.6–0.7 GB per process.
Labels are independent, but processes share the memory bandwidth: with 4 at
once a one-year label took 4.5 s each, with 8 at once 7.7 s each, i.e. 2.9× and
4.3× the throughput of one process (the M3 has 4 performance and 4 efficiency
cores).

Open issues:
- **r = 0.** A put is never exercised early when r ≤ 0, so the premium has a
  kink at r = 0 and the r and d windows straddle it: Rho and Phi are
  biased by 2–4 % there. The paper also finds r = d = 0 the hardest case.
- **Price** about 0.1 % low, a lower bound; accepted for now
  (`results/diagnostics/price_bias`). On six of the flagged options, each run's
  exercise rule, valued on paths that all start at z0, is below the reference:
  −0.12 % for a rule in S alone (the paper's kind of rule) and −0.16 % for the
  rules in (S, σ), (S, r), (S, d). So most of the shortfall is the usual
  Longstaff–Schwartz lower bound of a rule estimated from finite data, not the
  multivariate basis; the t = 0 regression adds nothing significant. The
  paper's estimator shows no price bias, presumably because two upward biases
  offset its rule's shortfall: the rule is applied to the shocks it was fitted
  on, and V(t1) = max(Z, Ĉ). The generator removes both (they bias Volga), and
  the offset is not reliable anyway: our replication flags the three r = d = 0
  prices of Table 6 high (+0.1 to +0.2 %), where no early exercise is optimal
  and only the upward biases remain. Unbiased alternatives: the PDE price (this
  model only) or a duality upper bound (costly).
  Two cheap fixes tried on the six options (paired, same seeds;
  `price_bias_check.py --compare`, `results/diagnostics/price_bias/comparison.md`):
  a European control variate in the exercise rule (regress Y_j − D_j, D_j the
  discounted European payoff) **doubles** the shortfall, −0.23 % vs −0.12 %:
  on the in-the-money paths the rule is fitted on, Y_j − D_j has 1.3–3× the
  residual variance of Y_j (most are exercised soon, so Y_j and D_j are weakly
  correlated); removed. N_pilot = 400,000 halves the rule's shortfall
  (−0.19 % → −0.095 %, paired +0.095 ± 0.007 %) and moves the label's price
  −0.12 % → −0.08 % (paired +0.04 ± 0.03 %, 25 reps), at 3.2× the CPU time. Not adopted
  (`results/diagnostics/price_bias/pilot_size.md`).
- **The t1 control variate** (default on) is worth keeping. Paired check
  without it (`multi_check.py american --set control_variate=False`, 25 reps,
  `results/multi_check_american_no_cv_vs_american.md`): one label's sd is
  1.5–1.8× larger for price, Δ, Γ, Θ, Rho and Phi, 1.3× for Vanna, about
  1.0–1.1× for Vega and Volga, i.e. 2.3–3.3× the labels for the same accuracy;
  no systematic change in bias (at most 2 of 30 paired differences
  flagged per Greek).
- **Δ** inherits the price's lower bound. Over the 30 options outside the
  exercise region the price is low in all 30 and |Δ| too small in 28, and the
  relative biases of price and Δ have correlation 0.91: the rule's shortfall
  grows as S falls, which flattens the fitted price curve. The bias reaches
  +1.5 % out of the money, where Δ is small. A better exercise rule would
  reduce both. **Γ, Θ** are 1–2.5 % off on three options.
- **Bermudan, not American.** Exercise is possible on J = round(50 T) dates.
  With the reference pricer, the American put is 0.2–0.3 % above the Bermudan
  one (S0 = 40, K = 36–44, σ = 20 %, T ≈ 0.5 and 1; the American variant is
  first order in time), and the Bermudan price jumps by 0.01–0.05 % where
  round(50 T) steps.
  Both matter if T is an input of a network.
- **Volga and Vanna** are unbiased but a single label is very noisy (median
  161 % and 72 % of the Greek): a network needs many labels to learn them.
- **Runtime.** About 3.3 s per one-year label on one core.

### Label generator defaults (October 2026) and the evidence behind them

`MultiConfig()` now encodes these choices; the older results files in `results/`
were produced with α_r = α_d = 0.02 and no shrinking or European region.

| Choice | Default | Evidence |
|---|---|---|
| α_S, α_σ | Paper's selector (Appendix A.2, M = ν + 1) on the pilot paths | Tables 4–5 reproduced with M = ν + 1 |
| α_r, α_d (r0 ≥ 3 %) | Fixed 0.03 | Width sweep 0.01–0.10 (9 options, paired): noise ∝ 1/α, bias grows above 0.05; 0.03–0.04 gives the lowest error once labels are averaged |
| α_r, α_d (0 < r0 < 3 %) | min(0.03, r0) (`alpha_r_shrink`, `alpha_d_shrink="r0"`) | 200-rep paired check at r0 = 0.5 %, 1 %: Rho unbiased on 9/12 options (current window: 0/12), Phi 9/12 (current 1/12); the d0 window is worse. Cost: Rho noise 11–29 % per label, Phi 13–38 % |
| r0 ≤ 0, d0 ≥ 0 | Exact European label (`european_region`) | No early exercise there: price, Δ, Γ, Θ, Vega match the Bermudan reference to 3 decimals; Rho, Phi are the no-exercise-side values |
| Selector for α_r, α_d | Off | Selected widths vary ±30 % per label; Rho, Phi noise +15–23 %, no bias gain |
| Kink basis in r (`kink_r`) | Off | No consistent bias gain, ≈2× noise, worse price and Γ |
| One-sided r, d spread (`rd_floor`) | Off | Noise ≈10× (derivative at the edge of the data) |
| Exercise-rule control variate (Y − D) | Removed | Price −0.23 % instead of −0.12 % (`results/diagnostics/price_bias/rule_cv`) |
| t1 control variate | On | Paired check without it: noise 1.5–1.8× (`results/multi_check_american_no_cv_vs_american.md`) |
| Exercise at t1 | By the rule, not max(Z, Ĉ) | max(Z, Ĉ) biased Volga +34 % |
| Exercise rule | Refitted on the rescaled pilot paths, applied out of sample | In-sample rule biased Volga +50 % |
| N_pilot = N | 100,000 each | 4× pilot paths did not remove the price bias (`results/diagnostics/price_bias/pilot_size.md`) |

Reference values near r = 0: `put_fd_greeks(..., one_sided=("r", "d"))` gives
forward bumps. Just above r = 0 the price bends sharply (the premium grows faster
than linearly in r), so at r0 = 0 the forward value depends on the bump; for
r0 ≥ 0.5 % central and forward bumps agree.

Open issues:
- Rho and Phi stay 3–4 % off at r0 = d0 = 1 % (K = 40, 44) with every window
  tried; the cause is unknown (the window does not cross r = 0 there).
- The price is about 0.1 % low: an estimated exercise rule is suboptimal, so the
  price is a lower bound; Δ inherits it. Shrinking the r and d windows adds
  0.04–0.10 percentage points at small r0.
- Labels are Bermudan (exercise on 50 dates per year), 0.2–0.3 % below American
  values.

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
- The order M in (A.1)–(A.3) is ν + 1 by default (see
  [The selector order M](#the-selector-order-m)).
- h_ROT is not clipped. For ν = 1 and M = M0 = 9 the printed (A.3) integral is
  zero, so h_ROT is infinite and the local fit uses all data.
- α\* is capped at 0.75 S0, only to keep rescaled prices positive. The cap
  never binds in the reported runs.
- For the stock, the selector runs in units where S0 = 40 (`x_unit = S0 / 40`).
  (A.3) as printed divides ∫w0/f, which has units of S², by a sum over paths,
  which has none, so h_ROT scales as S0^((2q+2)/(2q+1)) instead of S0, and α\*
  would depend on the currency unit: at S0 = 100 instead of 40, with the same
  S0/K and seed, the generator's α_S\* differed by 4–14 % and the Greeks by up
  to their noise. With fixed units, α\* and every output scale exactly with S0
  (`tests/test_core.py`). Results at S0 = 40, i.e. everything in this README,
  are unchanged.
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
