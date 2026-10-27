#!/bin/bash
# Slurm array: OMSim WavePID on nu_e final states (GENIE or Pythia chunk files). Geometry from LAYOUT:
#   file:<path>  DeepCore layout written by make_deepcore_layout.py (8 strings, 50 DOMs at 7 m, HQE) -> WORLD=210 VERTEX=36
#   fill:10:35   the 1 TeV lattice (world 50 m, vertex [-8,8]^3) -> mixable with the 1 TeV sample
# Other options as the 1 TeV production (random rotation of the event, --efficiency_cut, --charged_tracks).
# Submitted by run_gev.sh (which sets LAYOUT/WORLD/VERTEX/TAG from GEOM); by hand:
#   N=$(ls ~/IceCube/wavepid/out/pythia_gev/nue_cc_chunk*.txt | wc -l)
#   sbatch --export=ALL,CLASS=nue_cc,EVDIR=$HOME/IceCube/wavepid/out/pythia_gev,TAG=gev_deepcore,\
#          LAYOUT=file:$HOME/IceCube/wavepid/cluster/deepcore_layout.csv,WORLD=210,VERTEX=36 --array=0-$((N-1)) wavepid_gev_array.sbatch
# Output: ~/IceCube/wavepid/out/cluster/<CLASS>_<TAG>/part_NNNN_{hits.root,doms.json,args.json,profile_params.csv}
#SBATCH --job-name=wavepid_gev
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
CLASS=${CLASS:-nue_cc}; EVDIR=${EVDIR:-$HOME/IceCube/wavepid/out/pythia_gev}
LAYOUT=${LAYOUT:-file:$HOME/IceCube/wavepid/cluster/deepcore_layout.csv}   # or fill:10:35 (1 TeV lattice)
VERTEX=${VERTEX:-36}               # vertex uniform in [-36, 36]^3 m: one string-pattern unit cell x 10 DOM spacings
WORLD=${WORLD:-210}                # encloses the farthest DOM (186 m) of the DeepCore layout; 50 for fill:10:35
KILL=${KILL:-}                     # --photon_kill_distance [m]; unset = not passed (photons tracked until absorbed or world edge)
TAG=${TAG:-gev_deepcore}
PHOTON_TRACKS=${PHOTON_TRACKS:-1e-9}
CHARGED_TRACKS=${CHARGED_TRACKS:-0.1}   # fraction of non-photon tracks kept for the profile labels (1 TeV: 0.01;
                                        # ~30x fewer tracks per event here -> 0.1 keeps the label precision)
EFFCUT=${EFFCUT:-1}                # 1 = --efficiency_cut (quantum efficiency applied, as 1 TeV); 0 = all photocathode photons
chunk=$(printf "%s/%s_chunk%03d.txt" "$EVDIR" "$CLASS" "$SLURM_ARRAY_TASK_ID")
[ -f "$chunk" ] || { echo "no chunk file $chunk"; exit 1; }
NEVENTS=$(awk '!/^#/{if($1>m)m=$1}END{print m+1}' "$chunk")
OUT=$HOME/IceCube/wavepid/out/cluster/${CLASS}_${TAG}; mkdir -p "$OUT"
PART="$OUT/part_$(printf %04d $SLURM_ARRAY_TASK_ID)"
if [ -f "${PART}_hits.root" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "${PART}_hits.root exists, skipping (FORCE=1 to redo)"; exit 0
fi
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
cd $OMSIM_BUILD
SEED=$((5000000 + SLURM_ARRAY_JOB_ID % 100000 * 100 + SLURM_ARRAY_TASK_ID))
case "$LAYOUT" in file:*) [ -f "${LAYOUT#file:}" ] || { echo "layout file ${LAYOUT#file:} missing"; exit 1; } ;; esac
EXTRA=""
[ "$EFFCUT" = 1 ] && EXTRA="$EXTRA --efficiency_cut"
[ -n "$KILL" ] && EXTRA="$EXTRA --photon_kill_distance $KILL"
echo "class=$CLASS chunk=$chunk events=$NEVENTS layout=$LAYOUT vertex=$VERTEX world=$WORLD kill=${KILL:-none} seed=$SEED" \
     "photon_tracks=$PHOTON_TRACKS charged_tracks=$CHARGED_TRACKS effcut=$EFFCUT"
$WAVEPID_BIN -n "$NEVENTS" --detector_type 3 --environment 2 --hadronic --event_file "$chunk" \
    --dom_layout "$LAYOUT" --random_vertex "$VERTEX" --random_rotation \
    --photon_tracks "$PHOTON_TRACKS" --charged_tracks "$CHARGED_TRACKS" $EXTRA \
    --world_radius "$WORLD" --seed "$SEED" --multithreading --threads "$SLURM_CPUS_PER_TASK" \
    -o "$PART"
# Gamma-profile labels (needs the PhotonTracks tree; redone by the slim array if this fails)
python $HOME/IceCube/wavepid/gamma_functions.py "${PART}_hits.root" --all-events \
    || echo "WARNING: profile fit failed; the slim array will retry gamma_functions.py --all-events"
ls -la "${PART}"_*
