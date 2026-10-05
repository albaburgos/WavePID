# Source this before building or running OMSim:  source ~/IceCube/wavepid/env.sh
export MAMBA_ROOT_PREFIX=$HOME/mamba
eval "$(micromamba shell hook -s zsh)" 2>/dev/null
micromamba activate omsim
export OMSIM_BUILD=$HOME/IceCube/OMSim/build
export WAVEPID_BIN=$OMSIM_BUILD/OMSim_WavePID_study
