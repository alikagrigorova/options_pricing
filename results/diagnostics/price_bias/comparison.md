# Price bias by variant

Mean over the six options of (value − reference) / reference, ± standard error; rule: value of the exercise rule at z0; label: the run's price. S: the run with S alone dispersed; multivariate: mean of the (S, σ), (S, r), (S, d) runs; label default: the default label (their mean). Δ vs default: paired difference (same seeds). CPU: mean seconds per multivariate run (one process). Replications: those every variant has.

| variant | settings | rule, S | rule, multivariate | label default | Δ rule multivariate vs default | Δ label vs default | CPU s |
|---|---|---|---|---|---|---|---|
| default | default | -0.146% ± 0.024% | -0.190% ± 0.025% | -0.116% ± 0.026% | +0.000% ± 0.000% | +0.000% ± 0.000% | nan |
| rule_cv | rule_control_variate=True | -0.285% ± 0.030% | -0.320% ± 0.025% | -0.244% ± 0.027% | -0.129% ± 0.012% | -0.128% ± 0.015% | 3.08 |
| pilot400k | N_pilot=400_000 | -0.074% ± 0.026% | -0.095% ± 0.025% | -0.079% ± 0.023% | +0.095% ± 0.007% | +0.037% ± 0.034% | 9.81 |
