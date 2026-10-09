# Label analysis

results/labels_1000.csv

## Data

- 1000 options from 1 file(s); 7 where exercising now is optimal (label exact)
- inputs: moneyness 0.8696-1.1765, sigma 0.1-0.4, T 0.25-2, r 0-0.06, d 0-0.04
- inside the validated range: 608 of 1000
- chosen spreads: alpha_S 3.2-15.5 % of S0, alpha_sigma 0.10-0.46 x sigma

## Sanity checks (labels)

| check                                 |   labels | ids   |
|:--------------------------------------|---------:|:------|
| missing or infinite value             |        0 |       |
| Delta outside [-1, 0]                 |        0 |       |
| price below the exercise value K - S0 |        0 |       |
| price above K                         |        0 |       |
| negative Gamma                        |        0 |       |
| negative Vega                         |        0 |       |

## Exercise now?

- label and reference agree on 999 of 1000 options
- label exercises, reference does not: 0
- reference exercises, label does not: 1

## Accuracy of one label (errors in %)

| greek   |   n |   typical_error |   bias |   bias_se | significant   |   median_error |   heavy_tail |    worst |   worst_id |
|:--------|----:|----------------:|-------:|----------:|:--------------|---------------:|-------------:|---------:|-----------:|
| price   | 942 |            0.14 |  -0.12 |      0.01 | yes           |          -0.11 |         7.64 |    -0.89 |        495 |
| delta   | 985 |            0.54 |   0.22 |      0.03 | yes           |           0.23 |         9.54 |     6.83 |        813 |
| gamma   | 992 |            5.51 |  -1.27 |      0.36 | yes           |          -1.19 |        13.41 |   -49.97 |        977 |
| theta   | 969 |            6.42 |   1.56 |      0.61 |               |           1.26 |        17.44 |   185.26 |        941 |
| vega    | 991 |            2.18 |  -0.2  |      0.21 |               |          -0.24 |        15.24 |    55.89 |        829 |
| volga   | 317 |           40.06 |  -9.84 |     11.65 |               |           1.15 |        24.29 | -1542.47 |        357 |
| vanna   | 688 |           26.29 |   0.04 |      3.15 |               |           0.32 |        18.9  |   767.31 |        249 |
| rho     | 935 |            5.29 |  -0.05 |      0.37 |               |          -0.37 |        11.98 |    77.83 |        797 |
| phi     | 948 |            6.43 |  -0.24 |      0.45 |               |           0.05 |        12.13 |   -71.66 |        606 |
| delta_r | 888 |           47.94 |  10.27 |      4.8  |               |           4.5  |        17.91 |  1235.6  |        254 |
| delta_d | 818 |           54.38 |   2.04 |      6.13 |               |           0.56 |        19.07 | -1551.14 |         26 |

## Where the bias is significant (mean error in %, 99 % level)

| greek   | input           | region           |   n |   bias |   bias_se |
|:--------|:----------------|:-----------------|----:|-------:|----------:|
| price   | validated range | inside           | 600 |  -0.11 |      0.01 |
| price   | validated range | outside          | 342 |  -0.13 |      0.01 |
| price   | moneyness       | (0.869, 0.952]   | 343 |  -0.09 |      0.01 |
| price   | moneyness       | (0.952, 1.053]   | 331 |  -0.12 |      0.01 |
| price   | moneyness       | (1.053, 1.176]   | 268 |  -0.17 |      0.01 |
| price   | sigma           | (0.099, 0.2]     | 298 |  -0.12 |      0.01 |
| price   | sigma           | (0.2, 0.3]       | 328 |  -0.12 |      0.01 |
| price   | sigma           | (0.3, 0.4]       | 316 |  -0.12 |      0.01 |
| price   | T               | (0.249, 0.85]    | 317 |  -0.11 |      0.01 |
| price   | T               | (0.85, 1.4]      | 302 |  -0.12 |      0.01 |
| price   | T               | (1.4, 2.0]       | 323 |  -0.13 |      0.01 |
| price   | r               | (-0.001, 0.02]   | 341 |  -0.11 |      0.01 |
| price   | r               | (0.02, 0.04]     | 313 |  -0.12 |      0.01 |
| price   | r               | (0.04, 0.06]     | 288 |  -0.13 |      0.01 |
| price   | d               | (-0.001, 0.0125] | 320 |  -0.14 |      0.01 |
| price   | d               | (0.0125, 0.0275] | 358 |  -0.12 |      0.01 |
| price   | d               | (0.0275, 0.04]   | 264 |  -0.1  |      0.01 |
| delta   | validated range | inside           | 607 |   0.21 |      0.04 |
| delta   | validated range | outside          | 378 |   0.24 |      0.05 |
| delta   | moneyness       | (0.952, 1.053]   | 333 |   0.21 |      0.05 |
| delta   | moneyness       | (1.053, 1.176]   | 309 |   0.34 |      0.05 |
| delta   | sigma           | (0.099, 0.2]     | 335 |   0.25 |      0.06 |
| delta   | sigma           | (0.2, 0.3]       | 334 |   0.24 |      0.05 |
| delta   | sigma           | (0.3, 0.4]       | 316 |   0.18 |      0.05 |
| delta   | T               | (0.249, 0.85]    | 345 |   0.21 |      0.04 |
| delta   | T               | (0.85, 1.4]      | 312 |   0.23 |      0.06 |
| delta   | T               | (1.4, 2.0]       | 328 |   0.24 |      0.07 |
| delta   | r               | (-0.001, 0.02]   | 352 |   0.21 |      0.04 |
| delta   | r               | (0.02, 0.04]     | 329 |   0.29 |      0.06 |
| delta   | r               | (0.04, 0.06]     | 304 |   0.17 |      0.06 |
| delta   | d               | (-0.001, 0.0125] | 336 |   0.2  |      0.06 |
| delta   | d               | (0.0125, 0.0275] | 369 |   0.25 |      0.05 |
| delta   | d               | (0.0275, 0.04]   | 280 |   0.21 |      0.05 |
| gamma   | validated range | inside           | 607 |  -1.51 |      0.47 |
| gamma   | moneyness       | (1.053, 1.176]   | 317 |  -1.65 |      0.49 |
| gamma   | T               | (0.85, 1.4]      | 312 |  -1.74 |      0.63 |
| gamma   | r               | (-0.001, 0.02]   | 355 |  -2.02 |      0.45 |
| theta   | moneyness       | (1.053, 1.176]   | 307 |   2.05 |      0.63 |
| theta   | r               | (-0.001, 0.02]   | 353 |   2.01 |      0.48 |

## Largest errors (in units of the typical error)

| file            |   id |   moneyness |   sigma |    T |    r |    d | greek   |   error_% |   x_typical |
|:----------------|-----:|------------:|--------:|-----:|-----:|-----:|:--------|----------:|------------:|
| labels_1000.csv |  357 |        0.91 |    0.21 | 0.6  | 0.02 | 0.01 | volga   |  -1542.47 |       38.51 |
| labels_1000.csv |  249 |        0.99 |    0.11 | 2    | 0.04 | 0    | vanna   |    767.31 |       29.18 |
| labels_1000.csv |  941 |        0.89 |    0.22 | 1.75 | 0.06 | 0    | theta   |    185.26 |       28.86 |
| labels_1000.csv |   26 |        0.91 |    0.34 | 1.6  | 0.02 | 0.04 | delta_d |  -1551.14 |       28.52 |
| labels_1000.csv |  254 |        0.95 |    0.35 | 1.65 | 0.05 | 0.01 | delta_r |   1235.6  |       25.78 |
| labels_1000.csv |  829 |        0.88 |    0.19 | 0.65 | 0.05 | 0.04 | vega    |     55.89 |       25.66 |
| labels_1000.csv |  637 |        0.88 |    0.18 | 0.75 | 0.04 | 0    | vega    |     54.1  |       24.84 |
| labels_1000.csv |  861 |        0.88 |    0.23 | 1.85 | 0.04 | 0.02 | delta_r |   1181.05 |       24.64 |
| labels_1000.csv |  450 |        0.88 |    0.29 | 1.3  | 0.06 | 0.02 | volga   |   -870.51 |       21.73 |
| labels_1000.csv |  977 |        0.93 |    0.17 | 1.95 | 0.04 | 0.02 | volga   |    834.16 |       20.82 |

