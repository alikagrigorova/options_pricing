# multi_check american_no_cv vs american, paired

30 options outside the exercise region, 25-25 replications each (those both runs have; same seeds). Medians over options with |ref| above 1 % of the largest |ref| for that Greek. Flagged: |t| > 2.576 (bias against the reference, or paired difference B − A).

| Greek | noise ratio american_no_cv/american | bias american | flagged american | bias american_no_cv | flagged american_no_cv | median paired diff (american_no_cv − american) / ref | paired diffs flagged |
|---|---|---|---|---|---|---|---|
| price | 1.53 | 0.09% | 9 / 30 | 0.08% | 3 / 30 | +0.02% | 2 / 30 |
| delta | 1.58 | 0.31% | 1 / 30 | 0.28% | 0 / 30 | +0.01% | 0 / 30 |
| gamma | 1.83 | 1.28% | 1 / 30 | 2.27% | 2 / 30 | -0.85% | 2 / 30 |
| theta | 1.81 | 2.54% | 1 / 30 | 3.88% | 2 / 30 | +1.45% | 2 / 30 |
| vega | 1.09 | 0.73% | 0 / 30 | 0.83% | 0 / 30 | +0.40% | 1 / 30 |
| volga | 1.00 | 19.09% | 0 / 30 | 26.02% | 1 / 30 | -0.65% | 0 / 30 |
| rho | 1.53 | 2.29% | 2 / 30 | 2.82% | 0 / 30 | -0.94% | 0 / 30 |
| phi | 1.55 | 2.64% | 1 / 30 | 3.68% | 0 / 30 | +1.25% | 1 / 30 |
| vanna | 1.28 | 8.09% | 1 / 30 | 10.46% | 0 / 30 | -4.89% | 0 / 30 |
| delta_r | 1.41 | 13.86% | 0 / 30 | 40.66% | 2 / 30 | -1.63% | 1 / 30 |
| delta_d | 1.46 | 38.20% | 0 / 30 | 47.50% | 1 / 30 | +12.53% | 0 / 30 |
