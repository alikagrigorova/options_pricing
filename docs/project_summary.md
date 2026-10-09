# Project summary: what it does, what the results say, what to do next

A plain-language overview. The README has the details and the evidence.

## 1. What the project is

Two things built on one method:

1. **A replication** of Letourneau & Stentoft (2019), *Simulated Greeks for
   American Options*: prices, Delta and Gamma of American (in practice
   Bermudan, 50 exercise dates per year) puts by Monte Carlo.
2. **An extension** that produces *training labels* for a neural network:
   for one option, the price and nine Greeks (Δ, Γ, Θ, Vega, Volga, Vanna,
   Rho, Phi, plus ∂Δ/∂r and ∂Δ/∂d).

## 2. How it works (no code)

**The problem.** Monte Carlo gives a price easily, but a Greek is a
derivative: how the price moves when an input moves. Re-running the
simulation with a bumped input is noisy and slow, and early exercise makes
it worse.

**The idea (initial state dispersion).** Instead of starting every simulated
path at today's stock price, start each path at a slightly different stock
price, scattered around today's. After pricing, you have many (start
value, discounted payoff) pairs. Fit a smooth curve through them: the
curve's height at today's price is the price, its slope is Delta, its
curvature is Gamma.

**Workflow for one label:**

1. **Pick the option**: moneyness S0/K, volatility σ, rate r, dividend d,
   maturity T.
2. **Pilot run (100,000 paths)**. Scatter the starting points widely: the
   stock price, and also one of σ, r or d. Simulate the paths.
3. **Choose how wide the scatter should be**. Too wide and the fitted curve
   is bent by points far from today (bias); too narrow and the curve is
   driven by noise. The paper's selector picks the width on the pilot data.
4. **Learn the exercise rule on the pilot**. Going backwards in time, at
   each exercise date, regress "value of waiting" on the state, and exercise
   when the payoff beats it (Longstaff–Schwartz). The rule is learned on the
   pilot only, so it has no foresight on the paths used for the answer.
5. **Main run (100,000 fresh paths)** at the chosen width. Apply the rule.
   Only the early-exercise premium is estimated by simulation; the European
   part is added back in closed form (control variate). Tested: without it
   one label is 1.5–1.8× noisier in price, Δ, Γ, Θ, Rho and Phi (2.3–3.3×
   the labels for the same accuracy), with no systematic change in bias.
6. **Fit the curve at t = 0**. A polynomial in (stock, σ or r or d) around
   today's values. Its coefficients are the price and the Greeks.
7. **Combine three such runs**: (S, σ) gives Vega, Volga, Vanna; (S, r)
   gives Rho; (S, d) gives Phi; Δ, Γ and the price are averaged over the
   three; Θ comes from the Black–Scholes equation.
8. **Shortcut**: if exercising today is optimal, the label is exact
   (price K − S0, Δ = −1, the rest 0).

About 3.3 s per one-year label on one core.

**Validation workflow.** Each label is compared with an independent,
accurate PDE (finite-difference) pricer:

- *33-option check*: 33 fixed options, 100 independent labels each. The
  average over 100 labels shows the bias precisely; the spread shows the
  noise of one label.
- *1,000-label check*: 1,000 random options, one label each, over
  S0/K 0.87–1.18, σ 10–40 %, T 0.25–2, r 0–6 %, d 0–4 %. Shows whether the
  behaviour holds across the whole input space.

## 3. The results in plain words

**Replication.** 42 of 369 estimates differ significantly from the paper's
benchmarks; the paper itself has 44. The method is reproduced. One
interpretation choice (the order M of the selector) is ours and documented.

**Labels, 1,000 random options** (errors as % of the true value; "typical
error" is the size of the noise in one label):

| Greek | typical error of one label | bias | significant? | bias / noise |
|---|---|---|---|---|
| Price | 0.14 % | −0.12 % | yes | 0.56 |
| Δ | 0.54 % | +0.22 % (|Δ| too small) | yes | 0.18 |
| Γ | 5.5 % | −1.3 % | yes | 0.07 |
| Θ | 6.4 % | +1.6 % | no | 0.11 |
| Vega | 2.2 % | −0.2 % | no | 0.03 |
| Rho | 5.3 % | −0.05 % | no | 0.03 |
| Phi | 6.4 % | −0.2 % | no | 0.01 |
| Volga | 40 % | – | no | 0.05 |
| Vanna | 26 % | – | no | 0.05 |

In money (S0 = 100, median put price ≈ 9.9): the price is on average
**1.1 cents too low**, with a typical label error of about 1.2 cents.
Δ is off by 0.0009 on average. All sanity checks pass (no negative Gamma or
Vega, no price below exercise value), and exercise-now agrees with the
reference on 999 of 1,000 options.

## 4. The biases: significant, important, acceptable, fixable?

**"Significant" is not "important".** With 1,000 labels the test can detect
a shift of a few hundredths of a percent. Significant means *real*, not
*large*.

### Non-significant: accept

Θ, Vega, Rho, Phi, Volga, Vanna, ∂Δ/∂r, ∂Δ/∂d: no bias detectable. Their
problem is **noise**, not bias (Volga 40 %, Vanna 26 % per label). A network
averages noise across many labels, so these are usable; they need more
labels or a lower loss weight.

### Significant: price, Δ, Γ — one common cause

All three come from the same place: the exercise rule is learned from finite
data, so it is slightly sub-optimal, so the price is slightly too low (a
lower bound). That shortfall grows as the stock falls, which flattens the
price curve: |Δ| a bit too small, Γ a bit too small. Correlation between
the price and Δ errors: 0.91.

**Can we just accept them?** For most uses, yes:

- 1.1 cents on a 10-dollar put is well below a typical bid-ask spread.
- The gap between Bermudan (what we simulate: 50 dates/year) and true
  American is 0.2–0.3 %, *larger* than the price bias. If the target is
  really American, that is the bigger error.
- Model error (constant volatility) dwarfs both.

**But note** why it matters for a network: the network averages away the
noise, but not the bias. Trained on these labels, its price will converge
to "true price − 0.12 %". Bias / noise for the price is 0.56: the bias is
half the size of the noise of a single label, so it is the most visible
one. For Γ the ratio is 0.07, negligible.

**How to fix (in increasing cost):**

1. **Better exercise rule.** Tested on the six options with the clearest
   price bias (paired, same random numbers):
   - *European control variate in the rule*: **worse**, price −0.23 %
     instead of −0.12 %. The rule is fitted on in-the-money paths, most of
     them exercised soon, whose cash flow has little to do with the
     terminal payoff, so subtracting it adds noise. Removed from the code.
   - *4× pilot paths (400,000)*: the rule's shortfall **halves**
     (−0.19 % → −0.095 %); the label's price bias −0.12 % → −0.08 %. Costs
     3.2× the CPU time. The shortfall is estimation noise that shrinks
     slowly with more paths.
2. **Duality upper bound** (Andersen–Broadie): gives an upper price; the
   true price lies between it and ours. Expensive (nested simulation).
3. **Statistical correction**: the bias is small, smooth and nearly uniform
   (−0.09 % to −0.17 % across all input bins). Learn it from the PDE on a
   subset and subtract. Works for Black–Scholes only.
4. **In Black–Scholes only**: the PDE pricer is exact *and* faster
   (0.1–1.5 s vs 3.3 s). See section 6.

**Other known weak points:** r = 0 (Rho, Phi biased 2–4 %; the early
exercise premium has a kink there), and heavy-tailed errors: 8–24 % of
labels lie beyond 3× the typical error (≈ 0.3 % if the noise were normal).
For a network the outliers probably hurt more than the bias.

## 5. Are 33 options too few (moneyness-wise)?

The 33-option check has **only three moneyness levels**: S0/K = 0.91, 1.00,
1.11 (K = 36, 40, 44 at S0 = 40). That is too few to say anything about the
moneyness *profile*. It was designed as a precise pointwise test (100
labels per option), not a coverage test.

The **1,000-label set fills the gap**: 31 strikes, S0/K 0.87–1.18, at least
50 options in every 0.05 bucket, and no bucket behaves differently (price
bias −0.09 % to −0.17 %; Δ bias somewhat larger out of the money, +0.34 %
for S0/K > 1.05, where Δ is small).

Still missing:

- **Wider moneyness**: deep in the money (S0/K < 0.85, where early
  exercise dominates) and far out of the money (S0/K > 1.2, where the price
  is tiny and relative errors blow up). A real surrogate typically needs
  0.7–1.4.
- **Short maturities** (T < 0.25) and higher σ.
- **Precise checks at the edges**: a few options with 100 replications each
  at S0/K ≈ 0.8 and 1.3, like the 33-option check but at the extremes.

## 6. Can it be used now?

**Yes, for training a research/prototype surrogate** inside the tested range
(S0/K 0.87–1.18, σ 10–40 %, T 0.25–2, r 0–6 %, d 0–4 %), with:

- price, Δ, Γ, Θ, Vega, Rho, Phi as main targets;
- Volga, Vanna, ∂Δ/∂r, ∂Δ/∂d with a low weight (noisy);
- a robust loss (e.g. Huber) because of the heavy tails;
- r = 0 avoided or treated carefully.

**Not yet** for production pricing or hedging, and not outside the range.

**The honest caveat.** In plain Black–Scholes the PDE pricer already gives
exact, faster answers, so the Monte Carlo labels add nothing there except as
a proof of concept. Their real value is for models where no PDE is
practical (stochastic volatility, several assets, path-dependent features).
Black–Scholes is the right place to validate the method because the true
answer is known.

## 7. Next steps (suggested order)

1. Price/Δ/Γ bias: the control-variate exercise rule failed; 4× pilot
   paths halves the rule's shortfall at 3.2× the cost. Decide whether that
   is worth it, or try a cheaper way to a better rule (e.g. more paths only
   for the rule's final fit, not the width selection).
2. Decide Bermudan vs American as the target; if American, add exercise
   dates or extrapolate (0.2–0.3 % gap).
3. Extend the input range (moneyness 0.7–1.4, shorter T) and add
   100-replication checks at the edges.
4. Handle the heavy tails: understand the outliers (often near-zero
   Greeks), or use a robust loss.
5. Fix or exclude r = 0.
6. Generate a large training set (tens of thousands of labels; at 3.3 s per
   label, 50,000 labels take about 13 hours with 8 processes on an M3), train the network,
   and compare it with the PDE on a held-out set.
7. Move to a model without a PDE (e.g. Heston), where the method earns its
   cost.
