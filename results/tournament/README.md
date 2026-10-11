# Label-design tournament (SCC, 10 Oct 2026; not part of the paper)

`experiments/tournament.py`, run as SGE job arrays on the BU SCC (`scc/README.md`).
Eight variants: designs D1-D4 (`simgreeks.multi.DESIGNS`) x exercise rule out of
sample on the pilot paths (oos) or in sample (ins); 400,000 main + 400,000 pilot
paths per label; paired seeds. Raw Parquet output is not in the repository.

| File | What |
|---|---|
| report_european.md | D3, D4 vs the closed form, 6 options x 100 reps: 1/72 and 0/72 flagged |
| report_timing.md | pilot: seconds and memory per label |
| report_american.md | 33 options x 50 reps x 8 variants: flags, bias, noise |
| report_tournament.md | 2,500 options x 8 variants vs label_reference: mean bias +- se by region, RMSE, paired differences, seconds |

## Findings

1. **In-sample rules are out.** They bias Volga (+35 to +72 % on the 2,500
   options; 7-14 of 33 options flagged against 0-1 out of sample), flip the price
   bias to +0.10-0.13 % (foresight) and add Rho/Phi bias in the joint designs.
2. **D2-oos** (current default): the only design whose Rho, Phi and Vega are
   unbiased overall (+0.28 +- 0.19 %, +0.13 +- 0.24 %, -0.25 +- 0.16 %) and in
   every region; lowest noise for price, Delta, Gamma, Theta. Price -0.12 %,
   Delta +0.19 %, Gamma -0.8 % (the known lower bound of an estimated rule).
3. **D3-oos**: smallest price/Delta/Gamma/Theta bias, and 2-4x cheaper than D2 per
   unit of accuracy for Rho, Phi, Volga, dDelta/dr, dDelta/dd, plus Vera; but Rho
   -1.1 % (long T -1.4, low sigma -2.0, deep ITM -1.8), Phi +0.6 %, Vega -0.6 %
   (deep ITM -2.3), all significant. Gamma/Theta 1.6-1.7x less efficient.
4. **D4-oos**: lowest noise for every parameter Greek, but 4.5x the time (72 s per
   label), so 2-11x the cost per unit of accuracy except Volga; same Rho/Vega bias
   pattern as D3, dDelta/dr +7 %.
5. **D1-oos**: no advantage (12-75 % more expensive than D2 for every Greek).
6. r0 = 0 (50 options): exact in every design (European region).

The joint designs' Rho bias grows where the price is most curved in r (long T,
deep ITM, low sigma); their t = 0 polynomial is capped at order 2 in r and d
(D2 uses order 3, which the width sweep needed), a likely cause, untested.

## Recommendation

**D2-oos** (the current `MultiConfig()` default): no bias in any Greek beyond the
known price/Delta lower bound and a -1 % Gamma, in any region; cheapest after D1.
Its cost is noise in the parameter Greeks. D3-oos with order 3 in r and d at t = 0
is the candidate to beat it (half the cost for Rho/Phi/cross terms, plus Vera) if
a check shows the Rho/Phi/Vega bias gone.
