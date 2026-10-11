# Stage 1: production check, and D2 vs D3 (order 2 and 3)

Not part of the paper. Run on the BU SCC (`experiments/tournament.py`, parts
`production_check`, `american`, `tournament`; seeds fixed per option and
replication; a local rerun of 510 labels matched the SCC labels to 1e-8).

| File | What |
|---|---|
| report.md | Production check, **D2-prod** (`MultiConfig()`): 33 American check options + r0 = d0 = 1 % (K = 40, 44), 100 reps; per option and Greek bias ± se, t, noise; pass criteria; side by side with the README run and the tournament's D2-oos |
| report_D3.md | The same for **D3o3-prod** (D3, order 3 in σ, r, d at t = 0, 100k + 100k paths per run) |
| d3_order3_comparison.md | D2-oos vs D3-oos (order 2) vs D3-oos (order 3) on the tournament's 33-option check (50 reps) and 2,500 options, paired, by region |
| raw/, raw_D3/ | Per-replication labels (with seed entropy) and reference values |

## Pass criteria

| Criterion | D2-prod | D3o3-prod |
|---|---|---|
| No consistent-sign bias with abs(t) > 3 on more than ~2 options | **fails for price** (20 options, all low) **and Δ** (8, all high: abs(Δ) too small); passes for Γ (2), Θ (1), Vega, Volga, Vanna, Rho, Phi (0) | **fails for price** (9, all low); passes for Δ (2), Volga (2, both negative), all others (0) |
| Price bias within ~0.2 % | median 0.11 % passes; largest 0.48 % fails (K = 36, T = 2, σ = 10 %, price 0.17) | median 0.07 % passes; largest 0.22 % borderline |
| Δ bias within ~0.3 % | median 0.19 % passes; largest 1.47 % fails (K = 36, σ = 10 %, abs(Δ) = 0.03) | median 0.13 % passes; largest 1.07 % fails (same option) |

All price and Δ failures are the known out-of-sample lower bound (every price
bias negative, every Δ bias positive); they are largest in relative terms for
out-of-the-money options with small prices (K = 36, low σ: absolute bias
0.0003-0.0008). The r0 = d0 = 1 % supplement still shows Rho -3 to -4 % and Phi
+3 to +4 % (t about 2-3.4) in both designs.

## D2 or D3

Order 3 removes D3's negative Rho bias but overcorrects (+0.67 ± 0.22 %, +1.6 %
at low σ) and raises its Rho noise above D2's; D3's Phi becomes D2's (with order
3 its (S, d) run is identical to D2's); its Vega bias (-0.55 ± 0.14 %, -2.4 %
deep in the money) is unchanged. D3 keeps the smaller price/Δ/Γ lower bound of
its run in S alone (price -0.08 % vs -0.12 %) and adds Vera, at 30 % more time.

**Recommendation: D2-oos** (bias first: no significant bias in any parameter
Greek in any region; its extra price/Δ bias over D3 is 0.04-0.07 percentage
points, the same lower-bound mechanism). The price/Δ criteria fail for both
designs and need a decision (see the options in the summary to the user).
