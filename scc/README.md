# Running the label-design tournament on the BU SCC

Everything runs through `experiments/tournament.py` as SGE job arrays. Each task
runs every n-th label of a part, writes Parquet chunks every 10 labels and skips
finished labels when restarted, so a killed or failed task can be resubmitted
with the same command.

## 0. Get the code onto the SCC

Either clone the branch on the SCC (needs GitHub access there):

    git clone -b claude/focused-babbage-atq90l https://github.com/alikagrigorova/options_pricing.git

or copy the tarball from your laptop and unpack it:

    scp options_pricing.tar.gz <user>@scc1.bu.edu:/projectnb/<project>/<user>/
    ssh <user>@scc1.bu.edu
    cd /projectnb/<project>/<user> && mkdir -p options_pricing && tar xzf options_pricing.tar.gz -C options_pricing

## 1. One-time setup (login node)

    cd /projectnb/<project>/<user>/options_pricing
    bash scc/setup_env.sh                 # venv in ~/venvs/simgreeks, installs, runs the tests

If `module load python3/3.10.12` fails, pick a version from `module avail python3`
and edit the module line in `scc/setup_env.sh` and `scc/array.qsub`.

## 2. Start a tmux session and submit

    tmux new -s labels                    # later: tmux attach -t labels ; detach: Ctrl-b d
    cd /projectnb/<project>/<user>/options_pricing
    export SCC_PROJECT=<project>
    export OUT=/projectnb/<project>/<user>/tournament_out

Submit the parts in this order (estimates from a 4-core test machine; the timing
pilot gives the SCC numbers):

| Step | Command | Labels | CPU hours (est.) |
|---|---|---|---|
| a. timing pilot | `bash scc/submit.sh pilot 40 02:00:00` | 50 options x 8 variants | ~4 |
| b. European validation (D3, D4) | `bash scc/submit.sh european 60 03:00:00` | 6 x 100 x 2 | ~10 |
| c. reference values | `bash scc/submit.sh reference 25 02:00:00` | 2,500 | ~4 |
| d. American check | `bash scc/submit.sh american 200 06:00:00` | 33 x 50 x 8 | ~120 |
| e. tournament | `bash scc/submit.sh tournament 400 06:00:00` | 2,500 x 8 | ~180 |

a-c can run at the same time. After the pilot, adjust the number of tasks of d
and e so that one task takes about 1-2 hours (tasks = CPU hours / 1.5).

Each task uses one core and 4 GB (`mem_per_core=4G`); D4 at T = 2 peaks at about
2-2.5 GB.

## 3. Monitor

    qstat -u $USER | head                 # r = running, qw = waiting, Eqw = error
    watch -n 60 'qstat -u $USER | grep -c " r "'
    tail -f logs/tour_american.o*.1       # one task's log
    python experiments/tournament.py count --part american    # total labels of a part
    find $OUT/american -name '*.parquet' | wc -l               # chunks written (10 labels each)

Resubmit a part with the same command if tasks died: finished labels are skipped.

## 4. Reports (login node or an interactive session)

    source ~/venvs/simgreeks/bin/activate
    python experiments/tournament.py report --part timing --out $OUT/pilot
    python experiments/tournament.py report --part european --out $OUT
    python experiments/tournament.py report --part american --out $OUT
    python experiments/tournament.py report --part tournament --out $OUT

Reports are written to `$OUT/report_<part>.md`. Bring everything back with

    scp -r <user>@scc1.bu.edu:/projectnb/<project>/<user>/tournament_out .

(the Parquet files are small: ~20,000 rows for the tournament).

## Alternative: run inside tmux on an interactive node

    qrsh -P $SCC_PROJECT -pe omp 16 -l h_rt=12:00:00 -l mem_per_core=4G
    tmux new -s labels
    module load python3/3.10.12 && source ~/venvs/simgreeks/bin/activate
    cd /projectnb/<project>/<user>/options_pricing
    python experiments/tournament.py local --part european --workers 16 --out $OUT

(tmux on a compute node ends with the qrsh session; the job array above is the
robust option.)
