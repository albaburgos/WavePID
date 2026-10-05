#!/bin/bash
# usage: run_extract.sh <out-dir> <max-files> [extra args for extract_genie_events.py ...]
# The glob goes through GENIE_FILES because icetray-env re-expands quoted arguments.
OUT=$1; MAXF=${2:-0}; shift 2
export GENIE_FILES=${GENIE_FILES:-'/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/23257/*/*.i3.zst'}
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.3/setup.sh) 2>/dev/null
/cvmfs/icecube.opensciencegrid.org/py3-v4.4.3/icetray-env icetray/v1.19.1 \
  python $HOME/IceCube/wavepid/cluster/extract_genie_events.py --out-dir "$OUT" --max-files "$MAXF" "$@"
