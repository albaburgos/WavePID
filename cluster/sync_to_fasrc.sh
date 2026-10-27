#!/bin/zsh
# Ship the patched OMSim source (no build/) and the wavepid scripts to FASRC.
# Needs an open ControlMaster socket: run `ssh fasrc` once in your terminal first.
set -e
ssh -O check fasrc 2>/dev/null || { echo "open the connection first: ssh fasrc"; exit 1; }
rsync -av --delete --exclude build/ --exclude '*.orig' --exclude '*.orig2' --exclude '.git/' \
      ~/IceCube/OMSim/ fasrc:IceCube/OMSim/
rsync -av --exclude out/ --exclude __pycache__/ --exclude '.git/' --exclude PulseShapeInference/ \
      --exclude 'Claude outputs/' --exclude classifier/ --exclude '*.root' --exclude '*.npz' --exclude '*.parquet' \
      ~/IceCube/wavepid/ fasrc:IceCube/wavepid/
echo "synced. On the cluster:  bash ~/IceCube/wavepid/cluster/setup_fasrc.sh"
