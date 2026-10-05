#!/bin/zsh
# Simulate the event pools for the NC/CC benchmark (see ~/IceCube/classifier.py).
# Each pool is one primary type with a log-uniform energy spectrum, so that events
# can later be superposed: CC nu_e = e- (EM, (1-y)E) + pi- (hadronic, yE),
# NC nu_e = pi- (hadronic, yE) only. Same vertex and direction for all events.
#
# usage: ./run_benchmark_pools.sh [N_EM] [N_HAD] [EMIN_GeV] [EMAX_GeV] [IMPACT_m] [THREADS]
set -e
source "$(dirname "$0")/env.sh"
N_EM=${1:-1200}; N_HAD=${2:-2400}; EMIN=${3:-0.5}; EMAX=${4:-30}; D=${5:-5}; T=${6:-12}
OUT=$HOME/IceCube/wavepid/out/benchmark; mkdir -p "$OUT"
X0=$(python3 -c "print(-1.7*$D)")
cd "$OMSIM_BUILD"
for spec in "e-:$N_EM:" "pi-:$N_HAD:--hadronic"; do
  P=${spec%%:*}; rest=${spec#*:}; N=${rest%%:*}; extra=${rest#*:}
  tag=${P//-/m}
  mac=$OUT/${tag}_pool.mac
  cat > "$mac" <<MAC
/gps/particle $P
/gps/ene/type Pow
/gps/ene/alpha -1
/gps/ene/min $EMIN GeV
/gps/ene/max $EMAX GeV
/gps/position $X0 0 $D m
/gps/direction 1 0 0
MAC
  echo ">>> $P pool: N=$N, E log-uniform in [$EMIN, $EMAX] GeV, d=$D m  ($(date +%H:%M:%S))"
  "$WAVEPID_BIN" -n "$N" --detector_type 3 --environment 2 --macro "$mac" $extra \
      --multithreading --threads "$T" -o "$OUT/${tag}_pool" 2>&1 | grep -E "Computation time|error|critical" || true
done
ls -la "$OUT"/*_hits.root
