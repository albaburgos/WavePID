#!/bin/bash
# One-time setup of OMSim/WavePID on FASRC. Run it as a job, not on a login node
# (login nodes kill the multi-GB conda install):
#   sbatch ~/IceCube/wavepid/cluster/setup_job.sbatch
# Installs micromamba + a conda-forge env with Geant4 11.3.2 and ROOT (same as the
# Mac setup), then builds OMSim with the same patches/flags. ~40 min, mostly download.
set -eo pipefail
export MAMBA_ROOT_PREFIX=$HOME/mamba
if ! command -v micromamba >/dev/null; then
  mkdir -p $HOME/bin && cd $HOME/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1
  export PATH=$HOME/bin:$PATH
fi
eval "$(micromamba shell hook -s bash)"
SPECS="geant4=11.3.2 root boost spdlog fmt cmake ninja make pkg-config openssl expat xerces-c compilers python=3.11 uproot awkward numpy pandas pyarrow scikit-learn"
if [ ! -d $MAMBA_ROOT_PREFIX/envs/omsim ]; then
  micromamba create -y -n omsim -c conda-forge $SPECS
else
  # idempotent: completes a partial env (a login-node create was killed half-way)
  micromamba install -y -n omsim -c conda-forge $SPECS
fi
micromamba activate omsim
cd $HOME/IceCube/OMSim && mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DHDF5_ROOT=$CONDA_PREFIX \
      -DGeant4_DIR=$CONDA_PREFIX/lib/cmake/Geant4 -DCMAKE_CXX_FLAGS="-DFMT_CONSTEVAL=" ..
ninja -j${SLURM_CPUS_PER_TASK:-8}
ls -la OMSim_WavePID_study
echo "smoke test (5 events):"
./OMSim_WavePID_study -n 5 --detector_type 3 --environment 2 -d 5 -e 10 -p e- -o /tmp/omsim_smoke_$USER | tail -2
echo "setup done"
