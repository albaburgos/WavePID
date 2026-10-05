#!/bin/bash
# Slurm array for WavePID production on FASRC. One task = one chunk of events of
# one primary. Submit per primary, e.g. 50 chunks x 200 events = 10 000 events:
#   sbatch --export=ALL,PRIMARY=e-,ENERGY=50,IMPACT=20 --array=0-49 wavepid_array.sbatch
#   sbatch --export=ALL,PRIMARY=pi-,ENERGY=50,IMPACT=20,EXTRA=--hadronic --array=0-49 wavepid_array.sbatch
#   sbatch --export=ALL,PRIMARY=mu-,ENERGY=50,IMPACT=20 --array=0-49 wavepid_array.sbatch
# Then:  python ~/IceCube/wavepid/cluster/merge_parts.py ~/IceCube/wavepid/out/cluster/em_E50_d20
#SBATCH --job-name=wavepid
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
PRIMARY=${PRIMARY:-e-}; ENERGY=${ENERGY:-50}; IMPACT=${IMPACT:-20}
NEVENTS=${NEVENTS:-200}; EXTRA=${EXTRA:-}
# world must contain the start point at 1.7*d; photon kill beyond the far side of the track
WORLD=${WORLD:-$(python3 -c "print(max(30.0, 1.9*$IMPACT + 5))")}
KILL=${KILL:-$(python3 -c "print(1.4*$IMPACT + 5)")}
tag=${PRIMARY//-/m}
OUT=$HOME/IceCube/wavepid/out/cluster/${tag}_E${ENERGY}_d${IMPACT}; mkdir -p "$OUT"
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
cd $OMSIM_BUILD   # data are found via ../common/data
SEED=$((1000000 + SLURM_ARRAY_JOB_ID % 100000 * 100 + SLURM_ARRAY_TASK_ID))
echo "primary=$PRIMARY E=$ENERGY GeV d=$IMPACT m world=$WORLD m kill=$KILL m N=$NEVENTS seed=$SEED extra=$EXTRA"
$WAVEPID_BIN -n "$NEVENTS" --detector_type 3 --environment 2 -d "$IMPACT" -e "$ENERGY" -p "$PRIMARY" $EXTRA \
    --world_radius "$WORLD" --photon_kill_distance "$KILL" --seed "$SEED" \
    --multithreading --threads "$SLURM_CPUS_PER_TASK" \
    -o "$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)"
