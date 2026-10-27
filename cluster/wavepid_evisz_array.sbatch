#!/bin/bash
# Slurm array: OMSim WavePID, DeepCore NC/CC, flat in deposited energy (2026-10-21). Events from
# out/pythia_evis50_100_z (gen_pythia_events.py --evis-min 50 --evis-max 100 --axis z, merged by merge_evis_jobs.py):
# E_vis = sum of visible final-state kinetic energies flat in [50, 100) GeV for both classes (NC E_nu up to 5 TeV),
# neutrino along +z. Fixed vertex (0, 0, 0), no random vertex or rotation. LAYOUT: 224-DOM window, 8 strings x
# (14 below + 14 above the vertex) at 7 m: cluster/deepcore224_layout.csv (vertex on the array axis, 36 m from the
# nearest strings) or cluster/deepcore224_str85_1m_layout.csv (same DOMs shifted, vertex 1 m from string 85).
# Other options as wavepid_dc224_array.sbatch. One task = one chunk (200 events).
# DET: OMSim --detector_type (3 = DOM, default; 2 = mDOM, run with TAG=gev_evisz_s85_mdom, 2026-10-22).
#   cd ~/IceCube/wavepid/logs && sbatch --export=ALL,CLASS=nue_cc,LAYOUT=file:..._layout.csv --array=0-29 ../cluster/wavepid_evisz_array.sbatch
# Output: ~/IceCube/wavepid/out/cluster/<CLASS>_<TAG>/part_NNNN_{hits.root,doms.json,args.json,profile_params.csv}
#SBATCH --job-name=wavepid_evisz
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
W=$HOME/IceCube/wavepid
CLASS=${CLASS:-nue_cc}; EVDIR=${EVDIR:-$W/out/pythia_evis50_100_z}
LAYOUT=${LAYOUT:-file:$W/cluster/deepcore224_str85_1m_layout.csv}   # chosen 2026-10-21: vertex 1 m from string 85
WORLD=${WORLD:-160}                # farthest DOM 118.8 m (centre) / 140.4 m (str85_1m layout)
TAG=${TAG:-gev_evisz_s85}
DET=${DET:-3}                      # --detector_type: 3 = DOM, 2 = mDOM
PHOTON_TRACKS=${PHOTON_TRACKS:-1e-9}; CHARGED_TRACKS=${CHARGED_TRACKS:-0.1}
chunk=$(printf "%s/%s_chunk%03d.txt" "$EVDIR" "$CLASS" "$SLURM_ARRAY_TASK_ID")
[ -f "$chunk" ] || { echo "no chunk file $chunk"; exit 1; }
[ -f "${LAYOUT#file:}" ] || { echo "layout file ${LAYOUT#file:} missing"; exit 1; }
NEVENTS=${NEV:-$(awk '!/^#/{if($1>m)m=$1}END{print m+1}' "$chunk")}
OUT=$W/out/cluster/${CLASS}_${TAG}; mkdir -p "$OUT"
PART="$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)"
if [ -f "${PART}_hits.root" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "${PART}_hits.root exists, skipping (FORCE=1 to redo)"; exit 0
fi
source $W/cluster/env_fasrc.sh
cd $OMSIM_BUILD
SEED=$((5000000 + SLURM_ARRAY_JOB_ID % 100000 * 100 + SLURM_ARRAY_TASK_ID))
echo "class=$CLASS chunk=$chunk events=$NEVENTS layout=$LAYOUT vertex=(0,0,0) dir=+z (from the event file) world=$WORLD det=$DET seed=$SEED"
$WAVEPID_BIN -n "$NEVENTS" --detector_type "$DET" --environment 2 --hadronic --event_file "$chunk" \
    --dom_layout "$LAYOUT" -d 0 \
    --photon_tracks "$PHOTON_TRACKS" --charged_tracks "$CHARGED_TRACKS" --efficiency_cut \
    --world_radius "$WORLD" --seed "$SEED" --multithreading --threads "$SLURM_CPUS_PER_TASK" \
    -o "$PART"
python $W/gamma_functions.py "${PART}_hits.root" --all-events \
    || echo "WARNING: profile fit failed"
ls -la "${PART}"_*
