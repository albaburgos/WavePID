# source on the cluster before running OMSim
export MAMBA_ROOT_PREFIX=$HOME/mamba
export PATH=$HOME/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate omsim
export OMSIM_BUILD=${OMSIM_BUILD:-$HOME/IceCube/OMSim/build}   # run_1tev.sh may point this at a build on lab storage
export WAVEPID_BIN=$OMSIM_BUILD/OMSim_WavePID_study
