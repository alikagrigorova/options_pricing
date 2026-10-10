# Is N_pilot = 400,000 worth it?

Decision: **keep N_pilot = 100,000 (default)** for now. Revisit only if a
network trained on the labels is accurate to well below 0.1 % in price.

Data: `pilot400k/` vs the default (six price-bias options, 25 paired
replications, same seeds), `comparison.md`.

## Cost

| | N_pilot = 100,000 | N_pilot = 400,000 |
|---|---|---|
| CPU per run (one process) | 3.1 s | 9.8 s (3.2×) |
| one-year label (3 runs) | ≈ 3.3 s | ≈ 10.5 s |
| 50,000 labels, 8 processes on an M3 | ≈ 13 h | ≈ 40 h |
| peak memory per process | 0.6–0.7 GB (measured) | ≈ 2 GB or more (estimate, not measured: pilot arrays 4× larger) |

## Benefit

| | default | 400k | paired difference |
|---|---|---|---|
| exercise rule's value at z0 − reference | −0.19 % | −0.095 % | +0.095 % ± 0.007 % |
| label price − reference | −0.12 % | −0.08 % | +0.04 % ± 0.03 % (not significant) |

The gain in the label is about 0.04 % of the price: 0.4 cents on a 10-dollar
put.

## Why not

- **Diminishing returns.** The rule's shortfall shrinks roughly as 1/√N:
  4× the paths halve it, so halving it again needs about 1.6 million pilot
  paths, and removing it is out of reach.
- **The same compute is worth more as labels.** 3.2× more labels cut the
  noise a network sees by √3.2 ≈ 1.8×. The network's own approximation error
  will very likely exceed 0.04 %, so it cannot resolve the gain.
- **Not the largest error.** Bermudan (50 dates a year) vs American is
  0.2–0.3 %, five times the gain.

## Middle ground (not tested)

- N_pilot = 200,000: about 1.7–2× the cost for perhaps a 30 % smaller
  shortfall (from the 1/√N scaling).
- More paths only for the exercise rule's final refit, with the width
  selection on 100,000 paths: less than 3.2× the cost; how much less is not
  measured.

Either can be screened with
`python experiments/diagnostics/price_bias_check.py --reps 25 --set N_pilot=200_000 --name pilot200k`
(about 15–20 min on 4 cores) and compared with `--compare default pilot200k`.
