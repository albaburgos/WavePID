#!/bin/bash
# Slurm array: OMSim WavePID on real GENIE nu_e final states (one chunk file per task).
# Prepare chunks with run_extract.sh ... --chunk-size 200, then per class:
#   N=$(ls ~/IceCube/wavepid/out/genie_nue/nue_cc_chunk*.txt | wc -l)
#   sbatch --export=ALL,CLASS=nue_cc --array=0-$((N-1)) wavepid_genie_array.sbatch
#   sbatch --export=ALL,CLASS=nue_nc --array=0-$((N-1)) wavepid_genie_array.sbatch
# Afterwards: python merge_parts.py ~/IceCube/wavepid/out/cluster/nue_cc_E2-50_d5   (and nue_nc_...)
#SBATCH --job-name=wavepid_genie
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
CLASS=${CLASS:-nue_cc}; IMPACT=${IMPACT:-5}; EVDIR=${EVDIR:-$HOME/IceCube/wavepid/out/genie_nue}
TAG=${TAG:-E2-50}
chunk=$(printf "%s/%s_chunk%03d.txt" "$EVDIR" "$CLASS" "$SLURM_ARRAY_TASK_ID")
[ -f "$chunk" ] || { echo "no chunk file $chunk"; exit 1; }
NEVENTS=$(awk '!/^#/{print $1}' "$chunk" | sort -un | wc -l)
WORLD=${WORLD:-$(python3 -c "print(max(30.0, 1.9*$IMPACT + 5))")}
KILL=${KILL:-$(python3 -c "print(1.4*$IMPACT + 5)")}
OUT=$HOME/IceCube/wavepid/out/cluster/${CLASS}_${TAG}_d${IMPACT}; mkdir -p "$OUT"
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
cd $OMSIM_BUILD
SEED=$((2000000 + SLURM_ARRAY_JOB_ID % 100000 * 100 + SLURM_ARRAY_TASK_ID))
echo "class=$CLASS chunk=$chunk events=$NEVENTS d=$IMPACT world=$WORLD kill=$KILL seed=$SEED"
$WAVEPID_BIN -n "$NEVENTS" --detector_type 3 --environment 2 -d "$IMPACT" --hadronic \
    --event_file "$chunk" --world_radius "$WORLD" --photon_kill_distance "$KILL" --seed "$SEED" \
    --multithreading --threads "$SLURM_CPUS_PER_TASK" \
    -o "$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)"
