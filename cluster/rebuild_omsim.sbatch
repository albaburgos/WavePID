#!/bin/bash
# Incremental rebuild of the patched OMSim on FASRC (the env already exists; ninja recompiles only what changed,
# a few minutes) followed by a 3-event smoke test on the DeepCore layout with --qe_scale and --photon_fraction,
# so a parser or option problem shows up here and not in the 60 production tasks.
#   REBUILD=$(sbatch --parsable ~/IceCube/wavepid/cluster/rebuild_omsim.sbatch); echo $REBUILD
#   DEP=<gen job>:$REBUILD bash ~/IceCube/wavepid/cluster/run_gev.sh omsim
# Log: rebuild_omsim_<jobid>.out in the directory sbatch was run from. Exit status != 0 -> the omsim arrays
# (afterok) never start, which is what we want.
#SBATCH --job-name=rebuild_omsim
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
SRC=$HOME/IceCube/OMSim/simulations/wavepid
echo "patch check:"
grep -c 'f\[0\] == "file"' $SRC/src/OMSimWavePIDDetector.cc | sed 's/^/  file: branch occurrences: /'
grep -c '"qe_scale"' $SRC/OMSim_WavePID_study.cc | sed 's/^/  qe_scale option occurrences: /'
grep -q 'f\[0\] == "file"' $SRC/src/OMSimWavePIDDetector.cc || { echo "source on the cluster is NOT patched: run cluster/patch_omsim_deepcore.py on the Mac and sync"; exit 1; }
cd $OMSIM_BUILD
if [ ! -f build.ninja ]; then
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DHDF5_ROOT=$CONDA_PREFIX \
          -DGeant4_DIR=$CONDA_PREFIX/lib/cmake/Geant4 -DCMAKE_CXX_FLAGS="-DFMT_CONSTEVAL=" ..
fi
ninja -j${SLURM_CPUS_PER_TASK:-8}
ls -la OMSim_WavePID_study
CHUNK=$HOME/IceCube/wavepid/out/pythia_gev/nue_cc_chunk000.txt          # first production chunk, the real input path
if [ -f "$CHUNK" ]; then SRC_OPT="--event_file $CHUNK --hadronic"; else SRC_OPT="-e 10 -p e- -d 5"; fi
echo "smoke test: 3 events ($SRC_OPT) on the DeepCore layout, vertex box 36 m, qe_scale 1.35, photon_fraction 0.35"
SMOKE=/tmp/omsim_dc_smoke_$USER
./OMSim_WavePID_study -n 3 --detector_type 3 --environment 2 $SRC_OPT \
    --dom_layout file:$HOME/IceCube/wavepid/cluster/deepcore_layout.csv --random_vertex 36 --random_rotation \
    --efficiency_cut --qe_scale 1.35 --photon_fraction 0.35 --world_radius 210 --seed 7 \
    --multithreading --threads ${SLURM_CPUS_PER_TASK:-8} -o $SMOKE 2>&1 | grep -i 'dom_layout\|qe_scale\|warning\|error\|modules\|hits' | head -20
python - <<EOF
import json; d = json.load(open("$SMOKE" + "_doms.json")); print(f"  _doms.json: {len(d)} modules (expect 400)")
import uproot; f = uproot.open("$SMOKE" + "_hits.root"); n = f["PhotonHits"].num_entries; print(f"  PhotonHits: {n} hits in 3 events")
EOF
echo "rebuild done"
