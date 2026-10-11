# multi_check american_rd_selector vs american, paired

30 options outside the exercise region, 25-25 replications each (those both runs have; same seeds). Medians over options with |ref| above 1 % of the largest |ref| for that Greek. Flagged: |t| > 2.576 (bias against the reference, or paired difference B − A).

| Greek | noise ratio american_rd_selector/american | bias american | flagged american | bias american_rd_selector | flagged american_rd_selector | median paired diff (american_rd_selector − american) / ref | paired diffs flagged |
|---|---|---|---|---|---|---|---|
| price | 1.01 | 0.09% | 9 / 30 | 0.10% | 13 / 30 | -0.00% | 1 / 30 |
| delta | 0.99 | 0.31% | 1 / 30 | 0.31% | 2 / 30 | -0.00% | 3 / 30 |
| gamma | 1.02 | 1.28% | 1 / 30 | 1.41% | 1 / 30 | +0.18% | 0 / 30 |
| theta | 1.02 | 2.54% | 1 / 30 | 2.99% | 1 / 30 | -0.39% | 0 / 30 |
| vega | 1.00 | 0.73% | 0 / 30 | 0.73% | 0 / 30 | +0.00% | 0 / 30 |
| volga | 1.00 | 19.09% | 0 / 30 | 19.09% | 0 / 30 | +0.00% | 0 / 30 |
| rho | 1.15 | 2.29% | 2 / 30 | 2.65% | 0 / 30 | +0.20% | 0 / 30 |
| phi | 1.23 | 2.64% | 1 / 30 | 2.84% | 0 / 30 | +0.26% | 1 / 30 |
| vanna | 1.00 | 8.09% | 1 / 30 | 8.09% | 1 / 30 | +0.00% | 0 / 30 |
| delta_r | 1.16 | 13.86% | 0 / 30 | 20.31% | 1 / 30 | +0.23% | 0 / 30 |
| delta_d | 1.18 | 38.20% | 0 / 30 | 51.83% | 0 / 30 | -9.22% | 0 / 30 |
