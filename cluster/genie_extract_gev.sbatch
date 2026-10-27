#!/bin/bash
# Extract the GeV nu_e sample from GENIE 23257 with per-class E_nu windows, two passes of extract_genie_events.py
# (its window / --max-per-class apply to both classes, so pass 1 keeps nue_cc_*, pass 2 keeps nue_nc_*).
# Submitted by run_gev.sh gen (GEN=genie). Output: $EVDIR/nue_{cc,nc}_{events.txt,truth.csv,chunkNNN.txt}
#   + $EVDIR/passes.json with the windows and the number of GENIE files scanned per pass (needed for N_gen in the
#   physical weights: w = OneWeight x Phi / (NEvents per file x files scanned in that class's pass)).
#SBATCH --job-name=genie_extract_gev
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
EVDIR=${EVDIR:-$HOME/IceCube/wavepid/out/genie_gev}
N_PER_CLASS=${N_PER_CLASS:-6000}; CHUNK=${CHUNK:-200}; MAXFILES=${MAXFILES:-0}
EMIN_CC=${EMIN_CC:-4.5}; EMAX_CC=${EMAX_CC:-110}; EMIN_NC=${EMIN_NC:-8}; EMAX_NC=${EMAX_NC:-300}
export GENIE_FILES=${GENIE_FILES:-'/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/23257/*/*.i3.zst'}
RUNX=$HOME/IceCube/wavepid/cluster/run_extract.sh
mkdir -p "$EVDIR"
declare -A LO=([cc]=$EMIN_CC [nc]=$EMIN_NC) HI=([cc]=$EMAX_CC [nc]=$EMAX_NC)
CLASSES=${CLASSES:-"cc nc"}; entries=()
for cls in $CLASSES; do
    P="$EVDIR/pass_$cls"; rm -rf "$P"; mkdir -p "$P"
    echo "== pass $cls: E_nu in [${LO[$cls]}, ${HI[$cls]}] GeV, $N_PER_CLASS events, chunks of $CHUNK"
    bash "$RUNX" "$P" "$MAXFILES" --enu-min "${LO[$cls]}" --enu-max "${HI[$cls]}" \
         --max-per-class "$N_PER_CLASS" --chunk-size "$CHUNK" 2>&1 | tee "$P/extract.log"
    # files scanned for this class: the extractor prints "reached N events per class after K files" when it stops
    # early, otherwise every file was scanned
    K=$(grep -oE "after [0-9]+ files" "$P/extract.log" | grep -oE "[0-9]+" | tail -1)
    [ -n "$K" ] || K=$(ls $GENIE_FILES 2>/dev/null | wc -l)
    N=$(awk -F, 'NR>1{n++}END{print n+0}' "$P/nue_${cls}_truth.csv")
    rm -f "$EVDIR"/nue_${cls}_*
    cp "$P"/nue_${cls}_events.txt "$P"/nue_${cls}_truth.csv "$P"/nue_${cls}_chunk*.txt "$EVDIR/"
    entries+=("  \"$cls\": {\"emin\": ${LO[$cls]}, \"emax\": ${HI[$cls]}, \"n_extracted\": $N, \"files_scanned\": $K, \"chunk\": $CHUNK}")
done
{ echo "{"; (IFS=$'\n'; printf "%s,\n" "${entries[@]}" | sed '$ s/,$//'); echo "}"; } > "$EVDIR/passes.json"
cat "$EVDIR/passes.json"
for cls in $CLASSES; do echo "nue_$cls: $(ls "$EVDIR"/nue_${cls}_chunk*.txt | wc -l) chunk files"; done
echo "extraction done -> $EVDIR"
