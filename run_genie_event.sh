#!/bin/zsh
# Shoot ONE event of a GENIE/Pythia event file (cluster/extract_genie_events.py or
# cluster/gen_pythia_events.py format) through OMSim with the full DOM, recording the
# Geant4 tracks of the cascade and of the Cherenkov photons, then draw the light
# propagation with plot_light_propagation.py.
#
# usage: ./run_genie_event.sh [EVENT_FILE] [EVENT] [IMPACT_m] [PHOTON_FRACTION] [extra OMSim args...]
#   EVENT_FILE       default out/genie_nue/nue_nc_chunk000.txt
#   EVENT            index of the event inside the file (default 0)
#   IMPACT_m         impact parameter d; vertex at (-1.7 d, 0, d) (default 5)
#   PHOTON_FRACTION  fraction of the undetected photons whose paths are kept, or "auto"
#                    (default): min(0.02, 15000 / (6e5 * E_vis[GeV])), i.e. ~15k drawn
#                    photons whatever the event energy (a 50 GeV CC event has ~3e7 photons);
#                    all charged tracks and all detected photons are always kept
# Output: out/vis/<file>_ev<EVENT>_{hits.root,args.json,paths.png,snapshots.png}
#   GIF=1 ./run_genie_event.sh ...   also writes <...>_propagation.gif
#
# Runs single-threaded and without --photon_kill_distance so every photon path is
# complete. A few-GeV event takes seconds, a 30 GeV one a few minutes on one core.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
FILE=${1:-$HERE/out/genie_nue/nue_nc_chunk000.txt}; EV=${2:-0}; D=${3:-5}; FRAC=${4:-auto}
shift $(( $# < 4 ? $# : 4 ))
[[ -f $FILE ]] || { echo "no event file $FILE (download it: rsync fasrc:IceCube/wavepid/out/genie_nue/... out/genie_nue/)"; exit 1; }

OUT=$HERE/out/vis; mkdir -p "$OUT"
base=$(basename "$FILE" .txt)
one="$OUT/${base}_ev${EV}.txt"
# keep the header, take the lines of event EV, renumber it to 0
awk -v k="$EV" '/^#/ {print; next} $1==k {$1=0; print}' "$FILE" > "$one"
n=$(grep -cv '^#' "$one" || true)
[[ $n -gt 0 ]] || { echo "event $EV not found in $FILE"; exit 1; }
evis=$(grep -v '^#' "$one" | awk '{s+=$3} END {printf "%.4f", s}')
echo ">>> event $EV of $FILE: $n particles, E_vis = $evis GeV"
grep -v '^#' "$one" | awk '{printf "    %-10s %9.4f GeV  dir (%6.3f %6.3f %6.3f)\n", $2, $3, $4, $5, $6}'
if [[ $FRAC == auto ]]; then
  FRAC=$(python3 -c "print(f'{min(0.02, 15000 / (6e5 * max($evis, 1e-3))):.3g}')")
  echo "    photon fraction auto -> $FRAC (~15k drawn photons)"
fi

prefix="$OUT/${base}_ev${EV}"
cd "$OMSIM_BUILD"            # OMSim looks for ../common/data relative to its cwd
"$WAVEPID_BIN" -n 1 --detector_type 3 --environment 2 -d "$D" --hadronic \
    --event_file "$one" --photon_tracks "$FRAC" --seed 1 -o "$prefix" "$@" 2>&1 | tail -2

# truth csv of the same class, if the file came from extract_genie_events.py
truth=""; cls=${base%%_chunk*}; chunk=${base##*_chunk}
[[ -f $(dirname "$FILE")/${cls}_truth.csv && $chunk != $base ]] && truth="--truth $(dirname "$FILE")/${cls}_truth.csv --chunk $((10#$chunk))"
python "$HERE/plot_light_propagation.py" "$prefix" --event-file "$one" --event "$EV" $truth ${GIF:+--gif}
