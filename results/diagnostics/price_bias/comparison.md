# Price bias by variant

Mean over the six options of (value − reference) / reference, ± standard error; rule: value of the exercise rule at z0; label: the run's price. S: the run with S alone dispersed; multivariate: mean of the (S, σ), (S, r), (S, d) runs; label default: the default label (their mean). Δ vs default: paired difference (same seeds). CPU: mean seconds per multivariate run (one process).

| variant | settings | rule, S | rule, multivariate | label default | Δ rule multivariate vs default | Δ label vs default | CPU s |
|---|---|---|---|---|---|---|---|
| default | default | -0.117% ± 0.017% | -0.163% ± 0.016% | -0.115% ± 0.017% | +0.000% ± 0.000% | +0.000% ± 0.000% | nan |
| rule_cv | rule_control_variate=True | -0.243% ± 0.019% | -0.294% ± 0.017% | -0.234% ± 0.018% | -0.131% ± 0.009% | -0.119% ± 0.012% | 3.06 |
