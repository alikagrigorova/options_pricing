# Widths of the r and d dispersions (label generator; not part of the paper)

Diagnostics behind the defaults alpha_r = alpha_d = 0.03, min(0.03, r0) for
0 < r0 < 0.03, and the exact European label at r0 <= 0, d0 >= 0 (main README,
"Label generator defaults"). All runs reuse the seeds of multi_check american
(base 7002) or their own base seed, paired across variants.

| File | What |
|---|---|
| width_sweep.md | alpha_r = alpha_d in 0.01-0.10, 9 options x 25 reps: bias and noise per option |
| r0_zero_one_sided.md | r = d = 0: central vs forward vs European reference; symmetric vs one-sided spread, 50 reps |
| small_r_kink_shrink_50reps.md | r0 in 0.5-3 %: current, kink basis, shrinking window, 50 reps (kink and shrink not paired with current, see below) |
| small_r_shrink_200reps.md | r0 in 0.5 %, 1 %: shrinking r and d windows, 200 paired reps, signed bias +- se |
| multi_check_american_rd_selector*.md (in results/) | selector for alpha_r, alpha_d: no gain |

The scripts in scripts/ were run from a scratch directory; they write to the path
given on the command line and read the reference JSON files in raw/. In
small_r_kink_shrink the three (S, r) variants of a replication reused one
SeedSequence, which run_group spawns from, so they are independent rather than
paired samples; confirm_run.py builds a fresh SeedSequence per run.
