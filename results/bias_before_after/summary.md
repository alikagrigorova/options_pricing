# Label bias before and after averaging (Stage 2, 10,000 options)

Mean relative error of one label (%) ± se, t in brackets; 313 options with exact labels left out, and points with |PDE| < 1 % of the Greek's largest. Before: out-of-sample label; now: the default (averaged) label; same paths. † |t| > 2.576.

| Greek | n | before | now |
|---|---|---|---|
| Price | 9493 | -0.152 ± 0.003 (-59.2)† | -0.003 ± 0.002 (-1.1) |
| Δ | 9668 | +0.275 ± 0.011 (+24.2)† | -0.021 ± 0.011 (-1.9) |
| Γ | 9681 | -1.029 ± 0.125 (-8.2)† | +0.064 ± 0.121 (+0.5) |
| Θ | 9663 | +1.151 ± 0.289 (+4.0)† | -0.114 ± 0.282 (-0.4) |
| Vega | 9678 | -0.301 ± 0.086 (-3.5)† | -0.114 ± 0.080 (-1.4) |
| Rho | 9587 | +0.329 ± 0.117 (+2.8)† | -0.102 ± 0.108 (-0.9) |
| Phi | 9603 | +1.250 ± 1.892 (+0.7) | +0.806 ± 1.869 (+0.4) |

Volga, Vanna, ∂Δ/∂r and ∂Δ/∂d are the out-of-sample label in both, so they do not change. Figures: bias_overall.png, bias_by_input.png, errors_vs_inputs_before.png, errors_vs_inputs_now.png.
