#!/bin/zsh
# Run WavePID for a set of primaries so photon arrival-time distributions can be
# compared between CC-like (e-, EM cascade) and NC-like (pi-, hadronic cascade)
# events. mu- is kept as the track reference.
#
# usage: ./run_wavepid.sh [NEVENTS] [ENERGY_GeV] [IMPACT_m] [THREADS]
# OMSim looks for ../common/data relative to its cwd, so it must run from build/.
set -e
source "$(dirname "$0")/env.sh"
N=${1:-200}; E=${2:-10}; D=${3:-5}; T=${4:-8}
OUT=$HOME/IceCube/wavepid/out
cd "$OMSIM_BUILD"
for P in e- pi- mu-; do
  tag=${P//-/m}   # e- -> em, pi- -> pim, mu- -> mum
  extra=""
  [[ $P == pi* || $P == proton || $P == kaon* ]] && extra="--hadronic"
  echo ">>> $P  E=${E} GeV  d=${D} m  N=$N"
  "$WAVEPID_BIN" -n "$N" --detector_type 3 --environment 2 -d "$D" -e "$E" -p "$P" $extra \
      --multithreading --threads "$T" -o "$OUT/${tag}_E${E}_d${D}" 2>&1 | tail -3
done
ls -la "$OUT"/*_hits.root
