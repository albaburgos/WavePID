#!/bin/bash
# GeV nu_e NC/CC production (Pythia 8 DIS, E_vis 9-100 GeV) on FASRC in the DeepCore geometry (GEOM=deepcore, default: 8 strings,
# 50 DOMs at 7 m from cluster/deepcore_layout.csv, HQE PMTs) or on the 1 TeV lattice (GEOM=lattice: fill:10:35,
# world 50 m, vertex [-8,8]^3 -> mixable with the 1 TeV sample). Generator-level events are geometry independent and
# shared by both geometries (same EVDIR); only the OMSim stage differs.
#
#   bash ~/IceCube/wavepid/cluster/run_gev.sh all        # gen -> OMSim arrays -> slim arrays, chained with dependencies
#   bash ~/IceCube/wavepid/cluster/run_gev.sh gen|omsim|slim   # one stage (omsim/slim need the previous stage's output)
#   bash ~/IceCube/wavepid/cluster/run_gev.sh status     # chunk / part / slim counts and failed-task hints
#   bash ~/IceCube/wavepid/cluster/run_gev.sh download   # print the rsync commands for the Mac
#   DRYRUN=1 bash run_gev.sh all                         # print the sbatch commands only
#   Before the first run: `python cluster/pythia_lowE_check.py` (login node, omsim env) -> sigma/E and p.next() failure
#   rate at 5-300 GeV with these settings; if 9 GeV misbehaves use EMIN_CC=10 (the 9-12 GeV E_vis bin is then 65 % complete
#   for CC) or start the analysis window at 12 GeV.
#   GENIE alternative (QE+RES+DIS, nuclear effects, for an E_vis window down to 5 GeV): GEN=genie EMIN_CC=4.5 EMAX_CC=110
#   EMIN_NC=8 EMAX_NC=300 (genie_extract_gev.sbatch, GENIE 23257).
#
# Knobs (env):  GEN=pythia|genie (default pythia: DIS only, no QE/RES, no nuclear effects -> E_nu >= ~9-10 GeV, chosen
#               because the events of interest are above ~10 GeV; GENIE for the QE/RES region),
#               N_PER_CLASS=6000 CHUNK=200 EMIN_CC=9 EMAX_CC=110 EMIN_NC=10 EMAX_NC=300 SPECTRAL_INDEX=1 (log-uniform)
#               GEOM=deepcore|lattice  (deepcore: LAYOUT=file:$C/deepcore_layout.csv WORLD=210 VERTEX=36 TAG=gev_deepcore
#                                        KILL=  ; lattice: LAYOUT=fill:10:35 WORLD=50 VERTEX=8 TAG=gev_multidom)
#               TAG (-> out/cluster/nue_{cc,nc}_$TAG) CLASSES="cc nc" CPUS=8 MEM=16G TIME=03:00:00
#               CHARGED_TRACKS=0.1 EFFCUT=1 SEED=2001 KILL=<m> (--photon_kill_distance, unset = not passed)
#   DeepCore layout: `python cluster/make_deepcore_layout.py [--gcd-csv gcd_doms.csv]` writes cluster/deepcore_layout.csv
#   (synced by sync_to_fasrc.sh); needs the `file:` branch of the --dom_layout parser in OMSim_WavePID_study.cc.
#   Smoke test first:  N_PER_CLASS=200 CHUNK=100 TAG=gev_dc_smoke bash run_gev.sh all   (times a 100-event task)
# Energy windows (PulseShapeInference/energy_window_design.md, Pythia section): analysis window E_vis 9-100 GeV = the
# 8 oscNext bins above 9.1 GeV. CC E_nu [9,110] log-uniform: 96 % of the events in the window, ~700 per bin, every bin
# complete at the low edge. NC [10,300]: 60 % in-window, 350-510 per bin, 96 % complete in the 9-12 GeV bin and >= 96 %
# at the top under the atmospheric flux; tiles with the 1 TeV NC sample (300 GeV - 3 TeV). EMAX_NC=1000 if the GeV
# sample must stand alone under an E^-2 flux.
set -eo pipefail
STAGE=${1:-status}
W=$HOME/IceCube/wavepid; C=$W/cluster; OUTC=$W/out/cluster
GEN=${GEN:-pythia}; CLASSES=${CLASSES:-"cc nc"}
GEOM=${GEOM:-deepcore}
if [ "$GEOM" = deepcore ]; then
    LAYOUT=${LAYOUT:-file:$C/deepcore_layout.csv}; WORLD=${WORLD:-210}; VERTEX=${VERTEX:-36}; TAG=${TAG:-gev_deepcore}
elif [ "$GEOM" = lattice ]; then
    LAYOUT=${LAYOUT:-fill:10:35}; WORLD=${WORLD:-50}; VERTEX=${VERTEX:-8}; TAG=${TAG:-gev_multidom}
else
    echo "GEOM must be deepcore or lattice (got $GEOM)"; exit 1
fi
N_PER_CLASS=${N_PER_CLASS:-6000}; CHUNK=${CHUNK:-200}
EMIN_CC=${EMIN_CC:-9}; EMAX_CC=${EMAX_CC:-110}; EMIN_NC=${EMIN_NC:-10}; EMAX_NC=${EMAX_NC:-300}
SPECTRAL_INDEX=${SPECTRAL_INDEX:-1}; SEED=${SEED:-2001}
CPUS=${CPUS:-8}; MEM=${MEM:-16G}; TIME=${TIME:-03:00:00}
if [ -z "${EVDIR:-}" ]; then                      # event dir follows the tag so a top-up never clears the main sample
    case "$TAG" in gev_multidom|gev_deepcore) EVDIR=$W/out/${GEN}_gev ;; *) EVDIR=$W/out/${GEN}_${TAG} ;; esac
fi
NCHUNK=$(( (N_PER_CLASS + CHUNK - 1) / CHUNK ))
export EVDIR TAG N_PER_CLASS CHUNK EMIN_CC EMAX_CC EMIN_NC EMAX_NC CHARGED_TRACKS EFFCUT LAYOUT WORLD VERTEX GEOM KILL

run() { if [ "${DRYRUN:-0}" = 1 ]; then echo "+ $*" >&2; echo "DRY$RANDOM"; else "$@"; fi; }
sb()  { run sbatch --parsable "$@" | tail -1; }       # job id
count() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n+1)); done; echo "$n"; }   # files matching a glob (set -e safe)
nrows() { if [ -f "$1" ]; then awk -F, 'NR>1{n++}END{print n+0}' "$1"; else echo 0; fi; }   # data rows of a csv

stage_gen() {
    mkdir -p "$EVDIR"
    if [ "$GEN" = genie ]; then
        GEN_JOB=$(sb --export=ALL,EVDIR="$EVDIR",CLASSES="$CLASSES",N_PER_CLASS="$N_PER_CLASS",CHUNK="$CHUNK",EMIN_CC="$EMIN_CC",EMAX_CC="$EMAX_CC",EMIN_NC="$EMIN_NC",EMAX_NC="$EMAX_NC" \
                    "$C/genie_extract_gev.sbatch")
        echo "GENIE extraction job $GEN_JOB -> $EVDIR (per-class windows CC [$EMIN_CC,$EMAX_CC] NC [$EMIN_NC,$EMAX_NC] GeV)"
    else
        if [ "${DRYRUN:-0}" != 1 ]; then       # pythia8 must be importable in the omsim env (install needs the network: login node)
            source "$C/env_fasrc.sh"
            python -c "import pythia8" 2>/dev/null || micromamba install -y -n omsim -c conda-forge pythia8
        fi
        CMD="source $C/env_fasrc.sh"
        [[ " $CLASSES " == *" cc "* ]] && CMD="$CMD && python $C/gen_pythia_events.py --out-dir $EVDIR --classes cc --n-per-class $N_PER_CLASS --enu-min $EMIN_CC --enu-max $EMAX_CC --spectral-index $SPECTRAL_INDEX --chunk-size $CHUNK --seed $SEED"
        [[ " $CLASSES " == *" nc "* ]] && CMD="$CMD && python $C/gen_pythia_events.py --out-dir $EVDIR --classes nc --n-per-class $N_PER_CLASS --enu-min $EMIN_NC --enu-max $EMAX_NC --spectral-index $SPECTRAL_INDEX --chunk-size $CHUNK --seed $((SEED + 1))"
        GEN_JOB=$(sb --job-name=pythia_gen_gev --account=arguelles_delgado_lab --partition=shared --cpus-per-task=1 --mem=4G --time=02:00:00 \
                    --output=pythia_gen_gev_%j.out --wrap="$CMD")
        echo "Pythia generation job $GEN_JOB -> $EVDIR (E^-$SPECTRAL_INDEX, CC [$EMIN_CC,$EMAX_CC] NC [$EMIN_NC,$EMAX_NC] GeV)"
    fi
    echo "$GEN_JOB" > "$EVDIR/.gen_job"
}

stage_omsim() {           # $1 = optional dependency job id
    case "$LAYOUT" in file:*) [ -f "${LAYOUT#file:}" ] || { echo "layout file ${LAYOUT#file:} missing: run cluster/make_deepcore_layout.py and sync"; exit 1; } ;; esac
    DEP=${1:-}; [ -n "$DEP" ] && DEP="--dependency=afterok:$DEP"
    for cls in $CLASSES; do
        n=$(count "$EVDIR"/nue_${cls}_chunk*.txt); [ "$n" -gt 0 ] || n=$NCHUNK   # not generated yet: planned count
        JOB=$(sb $DEP --export=ALL,CLASS=nue_$cls,EVDIR="$EVDIR",TAG="$TAG",CHARGED_TRACKS="${CHARGED_TRACKS:-0.1}",EFFCUT="${EFFCUT:-1}",LAYOUT="$LAYOUT",WORLD="$WORLD",VERTEX="$VERTEX",KILL="${KILL:-}" \
                   --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --array=0-$((n-1)) "$C/wavepid_gev_array.sbatch")
        echo "OMSim array nue_$cls ($GEOM: $LAYOUT, world $WORLD m, vertex [-$VERTEX,$VERTEX]^3): job $JOB, $n tasks x $CHUNK events -> $OUTC/nue_${cls}_$TAG"
        echo "$JOB" > "$EVDIR/.omsim_job_$cls"
    done
}

stage_slim() {
    for cls in $CLASSES; do
        DEP=""; [ -f "$EVDIR/.omsim_job_$cls" ] && DEP="--dependency=afterany:$(cat "$EVDIR/.omsim_job_$cls")"
        n=$(count "$EVDIR"/nue_${cls}_chunk*.txt); [ "$n" -gt 0 ] || n=$NCHUNK
        JOB=$(sb $DEP --export=ALL,CLASS=nue_$cls,TAG="$TAG" --array=0-$((n-1)) "$C/wavepid_slim_array.sbatch")
        echo "slim array nue_$cls: job $JOB -> $OUTC/nue_${cls}_${TAG}_slim"
    done
}

stage_status() {
    echo "GEN=$GEN GEOM=$GEOM EVDIR=$EVDIR TAG=$TAG LAYOUT=$LAYOUT"
    [ -f "$EVDIR/passes.json" ] && cat "$EVDIR/passes.json"
    for cls in $CLASSES; do
        nchunk=$(count "$EVDIR"/nue_${cls}_chunk*.txt)
        nev=$(nrows "$EVDIR/nue_${cls}_truth.csv")
        nhits=$(count "$OUTC/nue_${cls}_$TAG"/part_*_hits.root)
        nprof=$(count "$OUTC/nue_${cls}_$TAG"/part_*_profile_params.csv)
        nslim=$(count "$OUTC/nue_${cls}_${TAG}_slim"/part_*_hits.root)
        printf "nue_%s: %s events in %s chunks | OMSim parts %s (profiles %s) | slim parts %s\n" "$cls" "$nev" "$nchunk" "$nhits" "$nprof" "$nslim"
        if [ "$nchunk" -gt 0 ] && [ "$nhits" -lt "$nchunk" ] && [ "$nhits" -gt 0 ]; then
            missing=$(for i in $(seq 0 $((nchunk-1))); do [ -f "$OUTC/nue_${cls}_$TAG/part_$(printf %04d $i)_hits.root" ] || printf "%s," $i; done)
            echo "   missing parts: ${missing%,}  (resubmit: sbatch --export=ALL,CLASS=nue_$cls,EVDIR=$EVDIR,TAG=$TAG --array=${missing%,} $C/wavepid_gev_array.sbatch)"
        fi
    done
    du -sh "$OUTC"/nue_*_${TAG}* 2>/dev/null || true
    squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.6D %R" 2>/dev/null | grep -E "gev|JOBID" || true
}

stage_download() {
    echo "# on the Mac (open 'ssh fasrc' first):"
    for cls in $CLASSES; do
        echo "rsync -avP fasrc:IceCube/wavepid/out/cluster/nue_${cls}_${TAG}_slim/ ~/IceCube/WavePID/PulseShapeInference/nue_${cls}_${TAG}/"
    done
    echo "rsync -avP fasrc:IceCube/wavepid/out/$(basename "$EVDIR")/ ~/IceCube/WavePID/PulseShapeInference/$(basename "$EVDIR")/ --include='*.csv' --include='passes.json' --exclude='*'"
    echo "# then: cd ~/IceCube/WavePID/PulseShapeInference && python run_gev_regressor.py --parts-tag $TAG"
}

case "$STAGE" in
    gen)      stage_gen ;;
    omsim)    stage_omsim ;;
    slim)     stage_slim ;;
    all)      stage_gen; stage_omsim "$GEN_JOB"; stage_slim ;;
    status)   stage_status ;;
    download) stage_download ;;
    *) echo "usage: run_gev.sh gen|omsim|slim|all|status|download"; exit 1 ;;
esac
