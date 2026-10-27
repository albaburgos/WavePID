#!/bin/bash
# Slurm array: OMSim WavePID on GENIE nu_e final states with a LATTICE of modules, a random vertex and a
# random rotation per event, recording particle tracks (training data for pulse_regressor.py).
# Prepare chunks with run_extract.sh ... --chunk-size 200, then per class:
#   N=$(ls ~/IceCube/wavepid/out/genie_nue/nue_cc_chunk*.txt | wc -l)
#   sbatch --export=ALL,CLASS=nue_cc --array=0-$((N-1)) wavepid_multidom_array.sbatch
# Afterwards, per part: python gamma_functions.py part_NNNN_hits.root --all-events   (-> _profile_params.csv)
# then on the Mac: python pulse_regressor.py --build --real-dir out/cluster/nue_cc_multidom --train
#SBATCH --job-name=wavepid_multidom
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
CLASS=${CLASS:-nue_cc}; EVDIR=${EVDIR:-$HOME/IceCube/wavepid/out/genie_nue}
LAYOUT=${LAYOUT:-fill:10:25}       # every 10 m lattice point within 25 m of the origin (81 DOMs): all modules near any vertex
VERTEX=${VERTEX:-8}                # vertex uniform in [-8, 8]^3 m
WORLD=${WORLD:-40}                 # must contain the modules (+-20 m) and the vertex cube
TAG=${TAG:-multidom}
EXTRA=${EXTRA:-}                   # extra OMSim flags, e.g. "--efficiency_cut --charged_tracks 0.01" (1 TeV production)
GAMMA=$HOME/IceCube/wavepid/gamma_functions.py; [ -f "$GAMMA" ] || GAMMA=$HOME/IceCube/wavepid/Xgamma_functions.py
chunk=$(printf "%s/%s_chunk%03d.txt" "$EVDIR" "$CLASS" "$SLURM_ARRAY_TASK_ID")
[ -f "$chunk" ] || { echo "no chunk file $chunk"; exit 1; }
NEVENTS=$(awk '!/^#/{if($1>m)m=$1}END{print m+1}' "$chunk")
OUTROOT=${OUTROOT:-$HOME/IceCube/wavepid/out}   # run_1tev.sh points this at lab storage; the home quota is 100 GB
OUT=$OUTROOT/cluster/${CLASS}_${TAG}; mkdir -p "$OUT"
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
cd $OMSIM_BUILD
SEED=$((4000000 + SLURM_ARRAY_JOB_ID % 100000 * 100 + SLURM_ARRAY_TASK_ID))
echo "class=$CLASS chunk=$chunk events=$NEVENTS layout=$LAYOUT vertex=$VERTEX world=$WORLD seed=$SEED extra='$EXTRA'"
$WAVEPID_BIN -n "$NEVENTS" --detector_type 3 --environment 2 --hadronic --event_file "$chunk" \
    --dom_layout "$LAYOUT" --random_vertex "$VERTEX" --random_rotation --photon_tracks 1e-9 \
    --world_radius "$WORLD" --seed "$SEED" --multithreading --threads "$SLURM_CPUS_PER_TASK" $EXTRA \
    -o "$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)"
# shower axis + Gamma profile per event -> part_NNNN_profile_params.csv. Needs numpy, pandas, uproot, scipy in the
# omsim env (micromamba install -n omsim scipy matplotlib). Non-fatal: the hits file is complete either way and the
# fit can be redone later with:  python gamma_functions.py part_NNNN_hits.root --all-events
python "$GAMMA" "$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)_hits.root" --all-events \
    || echo "WARNING: profile fit failed (missing scipy?); rerun gamma_functions.py --all-events on this part"
