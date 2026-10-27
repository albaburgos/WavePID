#!/bin/bash
# Incremental rebuild of OMSim_WavePID_study after a sync_to_fasrc.sh (a few source files, ~1 min).
# For a fresh install use setup_job.sbatch instead.
#SBATCH --job-name=omsim_rebuild
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -eo pipefail
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
echo "host $(hostname), env $CONDA_PREFIX, build dir $OMSIM_BUILD"
grep -c charged_tracks $HOME/IceCube/OMSim/simulations/wavepid/OMSim_WavePID_study.cc $HOME/IceCube/OMSim/simulations/wavepid/src/OMSimTrackingAction.cc
echo "== space in \$HOME (home quota is 100 GB):"; df -h $HOME | tail -1; du -sh $HOME/IceCube/wavepid/out $HOME/mamba $OMSIM_BUILD 2>/dev/null
mkdir -p "$OMSIM_BUILD" && cd "$OMSIM_BUILD"
if [ ! -f build.ninja ]; then        # fresh build directory (e.g. OMSIM_BUILD on lab storage): configure like setup_fasrc.sh
    echo "== configuring a fresh build in $OMSIM_BUILD"
    # OMSim reads its data files as ../common/data/... relative to the directory it runs in, so a build directory
    # outside the source tree needs a `common` link next to it
    [ -e "$(dirname "$OMSIM_BUILD")/common" ] || ln -s $HOME/IceCube/OMSim/common "$(dirname "$OMSIM_BUILD")/common"
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DHDF5_ROOT=$CONDA_PREFIX \
          -DGeant4_DIR=$CONDA_PREFIX/lib/cmake/Geant4 -DCMAKE_CXX_FLAGS="-DFMT_CONSTEVAL=" $HOME/IceCube/OMSim 2>&1 | tail -n 15
    NJ=${SLURM_CPUS_PER_TASK:-8}
else
    NJ=2
fi
# On the NFS home directory a parallel build has been seen to "lose" freshly written objects at link time
# (ld: cannot find ....o). A second ninja pass rebuilds whatever is missing; -j2 keeps it gentle.
# flock: never let two builds run in this directory at the same time (that, not NFS, was the 2026-10-05 failure)
exec 8>"$OMSIM_BUILD/.build.lock"; flock 8
ninja -j$NJ 2>&1 | tail -n 40 || { echo "first ninja pass failed, retrying"; sync; sleep 5; ninja -j1 2>&1 | tail -n 40 || { echo "ninja FAILED twice (see above)"; exit 1; }; }
ls -la OMSim_WavePID_study
./OMSim_WavePID_study --help 2>&1 | grep -A1 'charged_tracks\|photon_fraction' || { echo "binary lacks --charged_tracks/--photon_fraction after the build"; exit 1; }
echo "timing test: 2 events, 300 GeV e-, production layout and options, $SLURM_CPUS_PER_TASK threads"
echo "  (multiply the wall time per event by ~3.3 x 16/$SLURM_CPUS_PER_TASK for a 1 TeV event on 16 cores)"
time ./OMSim_WavePID_study -n 2 --detector_type 3 --environment 2 -e 300 -p e- --dom_layout fill:10:35 --world_radius 50 \
    --photon_tracks 1e-9 --charged_tracks 0.01 --efficiency_cut --photon_fraction 0.25 --seed 7 \
    --multithreading --threads $SLURM_CPUS_PER_TASK -o /tmp/omsim_1tev_smoke_$USER | tail -3
ls -la /tmp/omsim_1tev_smoke_${USER}_*
echo "rebuild done"
