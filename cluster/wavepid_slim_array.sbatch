#!/bin/bash
# Slurm array, one task per part of a wavepid multi-module production: fit the shower profile if the labels are
# still missing (gamma_functions.py --all-events; needs the PhotonTracks tree, so it runs before the slimming) and
# write a slim copy of the hits file with only the PhotonHits and Events trees (what pulse_regressor.py reads;
# ~70x smaller), with the _doms.json and _profile_params.csv next to it, into <CLASS>_<TAG>_slim/.
#   sbatch --export=ALL,CLASS=nue_cc --array=0-29 wavepid_slim_array.sbatch
#   sbatch --export=ALL,CLASS=nue_nc --array=0-29 --dependency=afterany:<production job id> wavepid_slim_array.sbatch
# then on the Mac:
#   rsync -avP fasrc:IceCube/wavepid/out/cluster/nue_cc_multidom_slim/ ~/IceCube/WavePID/out/cluster/nue_cc_multidom/
#SBATCH --job-name=wavepid_slim
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
set -eo pipefail
CLASS=${CLASS:-nue_cc}; TAG=${TAG:-multidom}
OUTROOT=${OUTROOT:-$HOME/IceCube/wavepid/out}
D=$OUTROOT/cluster/${CLASS}_${TAG}
part=$(printf "part_%04d" "$SLURM_ARRAY_TASK_ID")
[ -f "$D/${part}_hits.root" ] || { echo "no $D/${part}_hits.root"; exit 1; }
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
mkdir -p "${D}_slim"
if [ ! -f "$D/${part}_profile_params.csv" ]; then
    echo "fitting profiles of $part"
    GAMMA=$HOME/IceCube/wavepid/gamma_functions.py; [ -f "$GAMMA" ] || GAMMA=$HOME/IceCube/wavepid/Xgamma_functions.py
    python "$GAMMA" "$D/${part}_hits.root" --all-events
fi
rootcp --recreate "$D/${part}_hits.root:PhotonHits" "$D/${part}_hits.root:Events" "${D}_slim/${part}_hits.root"
cp "$D/${part}_doms.json" "$D/${part}_profile_params.csv" "${D}_slim/"
ls -la "${D}_slim/${part}"_*
