#!/bin/bash
# Submit one part of the tournament as an SGE job array.
#   SCC_PROJECT=<project> OUT=<dir> bash scc/submit.sh <part> <ntasks> [h_rt]
# parts: pilot (timing pilot: 50 options x 8 variants, with memory), european,
#        american, tournament, reference
# Optional environment: VENV (default ~/venvs/simgreeks), VARIANTS (colon-separated,
# e.g. D3-oos:D4-oos).
set -euo pipefail
PART=$1; NTASKS=$2; HRT=${3:-06:00:00}
: "${SCC_PROJECT:?set SCC_PROJECT to your SCC project name}"
: "${OUT:?set OUT to the output directory, e.g. /projectnb/<project>/$USER/tournament_out}"
mkdir -p logs "$OUT"
LIMIT=""; MEM=0; RUNPART=$PART; DEST=$OUT
if [ "$PART" = "pilot" ]; then RUNPART=tournament; LIMIT=50; MEM=1; DEST=$OUT/pilot; fi
qsub -P "$SCC_PROJECT" -N "tour_$PART" -t 1-"$NTASKS" -l h_rt="$HRT" \
     -v PART="$RUNPART",NTASKS="$NTASKS",OUT="$DEST",LIMIT="$LIMIT",MEM="$MEM",VARIANTS="${VARIANTS:-}",VENV="${VENV:-$HOME/venvs/simgreeks}" \
     scc/array.qsub
