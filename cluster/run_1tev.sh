#!/bin/bash
# ~1 TeV nu_e DIS production, end to end on FASRC, as one chain of Slurm jobs (run on a login node):
#   bash ~/IceCube/wavepid/cluster/run_1tev.sh            submit everything
#   bash ~/IceCube/wavepid/cluster/run_1tev.sh status     progress: jobs, finished parts, slim files, failed jobs
#   bash ~/IceCube/wavepid/cluster/run_1tev.sh logs [N]   last N lines of the newest log of each job type
#   bash ~/IceCube/wavepid/cluster/run_1tev.sh progress   events completed per part (AutoSaved ROOT files + OMSim progress lines)
#   MAIL_USER=you@fas.harvard.edu bash run_1tev.sh         same, plus an e-mail when each job or array ends/fails
# Re-running the submit command is safe: finished stages (events, rebuilt binary, slim parts) are skipped.
# Chain:  rebuild OMSim (patched --charged_tracks)  ─┐
#         Pythia 8 events, 6000 CC + 6000 NC, 0.3-3 TeV ─┴─> OMSim arrays (nue_cc, nue_nc) ─> slim arrays (profile fit + PhotonHits/Events only)
# Names follow the low-energy samples: events in out/pythia_nue_1tev/ (nue_cc_chunkNNN.txt, nue_cc_truth.csv),
# hits in out/cluster/nue_cc_1tev_multidom/ and nue_cc_1tev_multidom_slim/ (same for nue_nc).
# Afterwards on the Mac (see README):
#   rsync -avP fasrc:IceCube/wavepid/out/cluster/nue_cc_1tev_multidom_slim/ ~/IceCube/WavePID/PulseShapeInference/nue_cc_1tev_multidom/
set -eo pipefail
W=$HOME/IceCube/wavepid; C=$W/cluster
TAG=1tev_multidom
# Output root for the hits files. The home directory has a 100 GB quota and the full 1 TeV production is ~100 GB,
# so use lab storage when it exists (first match wins); override with OUTROOT=/path bash run_1tev.sh
LAB=arguelles_delgado_lab
for cand in "${OUTROOT:-}" /n/holylfs05/LABS/$LAB/Users/$USER/wavepid/out /n/netscratch/$LAB/Users/$USER/wavepid/out \
            /n/holylfs05/LABS/$LAB/Lab/$USER/wavepid/out /n/netscratch/$LAB/Lab/$USER/wavepid/out; do
    [ -n "$cand" ] || continue
    if [ -d "$cand" ] || mkdir -p "$cand" 2>/dev/null; then OUTROOT=$cand; break; fi
done
OUTROOT=${OUTROOT:-$W/out}
export OUTROOT
# Build directory. Default: the existing build in $HOME. If the NFS home keeps losing object files at link time,
# build on lab storage instead:  OMSIM_BUILD=$OUTROOT/../OMSim_build bash run_1tev.sh   (fresh configure, ~20 min)
export OMSIM_BUILD=${OMSIM_BUILD:-$HOME/IceCube/OMSim/build}
EVDIR=$W/out/pythia_nue_1tev
N_PER_CLASS=${N_PER_CLASS:-6000}; CHUNK=${CHUNK:-100}
NCHUNK=$(( (N_PER_CLASS + CHUNK - 1) / CHUNK ))
# OMSim settings for TeV showers: 10 m lattice out to 35 m (~180 DOMs) in a 50 m world, vertex in [-8,8]^3 m,
# photons that fail the PMT QE are dropped at the photocathode (--efficiency_cut, ~4x fewer hits), only 1 % of the
# charged tracks go to the PhotonTracks tree (--charged_tracks 0.01; unbiased profile, affordable file size), and only
# 25 % of the Cherenkov photons are tracked at all (--photon_fraction 0.25: exact for the DOM, whose QE x CE <= 0.245,
# because the detection probability is divided by 0.25; 4x faster). 2026-10-05/06: without the last option 100 events
# did not finish in 12 h on 16 cores (> 2 core-hours per event), every task timed out and lost its output.
LAYOUT=fill:10:35; WORLD=50; VERTEX=8
EXTRA="--efficiency_cut --charged_tracks 0.01 --photon_fraction 0.25"
CPUS=${CPUS:-16}; MEM=${MEM:-48G}; TIME=${TIME:-24:00:00}
# PILOT=N bash run_1tev.sh  -> submit only the first N missing parts per class (measure the per-event cost first)
PILOT=${PILOT:-0}
# MAIL_USER=you@fas.harvard.edu bash run_1tev.sh  -> Slurm e-mails when each job/array ends or fails (one mail per array)
MAILOPT=""; [ -n "${MAIL_USER:-}" ] && MAILOPT="--mail-type=END,FAIL --mail-user=$MAIL_USER"

if [ "${1:-}" = "status" ]; then
    echo "== output root: $OUTROOT"; df -h "$OUTROOT" | tail -1
    echo "== jobs"; squeue -u $USER -o "%.10i %.16j %.8T %.10M %.6D %R" | grep -E "JOBID|1tev|omsim_rebuild|pythia|wavepid" || true
    echo "== events"; for c in nue_cc nue_nc; do printf "  %s: %s chunks\n" $c "$(ls $EVDIR/${c}_chunk*.txt 2>/dev/null | wc -l)"; done
    echo "== parts (hits / profile fits / slim)"
    for c in nue_cc nue_nc; do
        D=$OUTROOT/cluster/${c}_$TAG
        printf "  %-7s %3s / %3s / %3s   of %d\n" $c "$(ls $D/part_*_hits.root 2>/dev/null | wc -l)" \
            "$(ls $D/part_*_profile_params.csv 2>/dev/null | wc -l)" "$(ls ${D}_slim/part_*_hits.root 2>/dev/null | wc -l)" $NCHUNK
        du -sh $D ${D}_slim 2>/dev/null | sed 's/^/    /' || true
    done
    echo "== download (on the Mac):"
    echo "  for c in nue_cc nue_nc; do rsync -avP fasrc:$OUTROOT/cluster/\${c}_${TAG}_slim/ ~/IceCube/WavePID/PulseShapeInference/\${c}_${TAG}/; done"
    echo "== jobs of the last day that did not complete (sacct)"
    sacct -u $USER -S now-1day -X --name=wavepid_multidom_nue_cc,wavepid_multidom_nue_nc,wavepid_slim_nue_cc,wavepid_slim_nue_nc,pythia_1tev,omsim_rebuild \
        -o JobID%18,JobName%24,State%22,ExitCode,Elapsed,MaxRSS,Reason%30 | grep -vE "COMPLETED|RUNNING|PENDING" | head -40 || true
    exit 0
fi
if [ "${1:-}" = "progress" ]; then    # events completed so far in every (running or finished) part, from the AutoSaved ROOT files
    source $C/env_fasrc.sh
    for c in nue_cc nue_nc; do
        D=$OUTROOT/cluster/${c}_$TAG; ls $D/part_*_hits.root >/dev/null 2>&1 || { echo "$c: no parts yet"; continue; }
        echo "== $c  (slim done: $(ls ${D}_slim/part_*_hits.root 2>/dev/null | wc -l))"
        python - $D/part_*_hits.root <<'PY'
import sys, os, time
import uproot
for f in sys.argv[1:]:
    age = (time.time() - os.path.getmtime(f)) / 60
    try:
        n = uproot.open(f)["Events"].num_entries
    except Exception:
        n = "<10 (no header yet)"
    print(f"  {os.path.basename(f)}: {n} events, {os.path.getsize(f)/1e6:.0f} MB, last written {age:.0f} min ago")
PY
    done
    echo "== last progress line of the newest logs (OMSim prints one every 10 events)"
    cd $W/logs 2>/dev/null && for f in $(ls -t wavepid_multidom_nue_*.out 2>/dev/null | head -8); do
        p=$(grep -h 'progress:' $f | tail -1); echo "  $f: ${p:-no progress line yet}"; done
    exit 0
fi
if [ "${1:-}" = "logs" ]; then        # last lines of the most recent log of every job type
    cd $W/logs 2>/dev/null || { echo "no logs yet"; exit 0; }
    for pat in omsim_rebuild_ pythia_1tev_ wavepid_multidom_nue_cc_ wavepid_multidom_nue_nc_ wavepid_slim_nue_cc_ wavepid_slim_nue_nc_; do
        f=$(ls -t ${pat}*.out 2>/dev/null | head -1); [ -n "$f" ] || continue
        echo "==================== $f"; tail -n ${2:-25} "$f"
    done
    exit 0
fi

[ -x $HOME/IceCube/OMSim/build/OMSim_WavePID_study ] || [ "$OMSIM_BUILD" != "$HOME/IceCube/OMSim/build" ] || { echo "OMSim not built on the cluster: sbatch $C/setup_job.sbatch first"; exit 1; }
grep -q charged_tracks $HOME/IceCube/OMSim/simulations/wavepid/OMSim_WavePID_study.cc || { echo "patched OMSim source missing: run ./cluster/sync_to_fasrc.sh on the Mac first"; exit 1; }
mkdir -p "$OUTROOT/cluster" $W/logs && cd $W/logs
# Guard against submitting the chain twice (2026-10-05: two simultaneous submissions ran two ninja builds in the same
# build directory and both failed). Refuse while any job of the chain is queued or running, unless RESUBMIT=1.
exec 9>$W/logs/.submit.lock
flock -n 9 || { echo "another run_1tev.sh is submitting right now; wait for it"; exit 1; }
active=$(squeue -u $USER -h -o '%j' | grep -E '^(omsim_rebuild|pythia_1tev|wavepid_multidom_nue_(cc|nc)|wavepid_slim_nue_(cc|nc))$' | sort | uniq -c || true)
if [ -n "$active" ] && [ "${RESUBMIT:-0}" != 1 ]; then
    echo "jobs of this chain are already queued or running:"; echo "$active"
    echo "use   bash run_1tev.sh status   to follow them, or RESUBMIT=1 bash run_1tev.sh to add missing parts anyway"; exit 1
fi
echo "hits files go to $OUTROOT/cluster  ($(df -h "$OUTROOT" | awk 'NR==2{print $4}') free)"
case "$OUTROOT" in $HOME/*) echo "WARNING: output root is in \$HOME (100 GB quota); set OUTROOT=/n/netscratch/$LAB/Users/$USER/wavepid/out or similar";; esac
echo "home directory:"; df -h $HOME | tail -1
source $C/env_fasrc.sh
python -c "import pythia8" 2>/dev/null || { echo "installing pythia8 into the omsim env"; micromamba install -y -n omsim -c conda-forge pythia8; }
python -c "import scipy, uproot, pandas" 2>/dev/null || micromamba install -y -n omsim -c conda-forge scipy uproot pandas

# Re-runnable: stages whose output exists are skipped (FORCE_REBUILD=1 / FORCE_GEN=1 override).
deps=""
echo "build directory: $OMSIM_BUILD"
if [ "${FORCE_REBUILD:-0}" = 1 ] || ! "$OMSIM_BUILD/OMSim_WavePID_study" --help 2>/dev/null | grep -q photon_fraction; then
    rebuild=$(sbatch --parsable $MAILOPT --export=ALL,OMSIM_BUILD=$OMSIM_BUILD $C/rebuild_fasrc.sbatch); rebuild=${rebuild%%;*}
    echo "rebuild job $rebuild"; deps="$deps:$rebuild"
else
    echo "OMSim binary already has --charged_tracks: rebuild skipped"
fi
have_cc=$(ls $EVDIR/nue_cc_chunk*.txt 2>/dev/null | wc -l); have_nc=$(ls $EVDIR/nue_nc_chunk*.txt 2>/dev/null | wc -l)
if [ "${FORCE_GEN:-0}" = 1 ] || [ "$have_cc" -lt "$NCHUNK" ] || [ "$have_nc" -lt "$NCHUNK" ]; then
    gen=$(sbatch --parsable $MAILOPT --export=ALL,EVDIR=$EVDIR,N_PER_CLASS=$N_PER_CLASS,CHUNK=$CHUNK,EMIN=300,EMAX=3000,SEED=1001 $C/gen_pythia_1tev.sbatch); gen=${gen%%;*}
    echo "event generation job $gen"; deps="$deps:$gen"
else
    echo "events already generated ($have_cc + $have_nc chunk files in $EVDIR): generation skipped"
fi
[ -n "$deps" ] && depopt="--dependency=afterok${deps}" || depopt=""
declare -A sim
for c in nue_cc nue_nc; do
    # tasks whose slim output already exists, or that are queued/running right now (e.g. a PILOT batch), are left out
    queued=" $(squeue -u $USER -h -r -o '%j %K' | awk -v n="wavepid_multidom_$c" '$1==n{print $2}' | paste -sd' ' -) "
    todo=""; for i in $(seq 0 $((NCHUNK-1))); do
        [ -f "$OUTROOT/cluster/${c}_${TAG}_slim/part_$(printf %04d $i)_hits.root" ] && continue
        case "$queued" in *" $i "*) continue;; esac
        todo="$todo,$i"; done
    todo=${todo#,}
    [ -n "$todo" ] || { echo "$c: all $NCHUNK slim parts exist, nothing to do"; continue; }
    [ "$PILOT" -gt 0 ] && todo=$(echo "$todo" | tr ',' '\n' | head -n "$PILOT" | paste -sd, -)
    sim[$c]=$(sbatch --parsable $MAILOPT $depopt --array=$todo --cpus-per-task=$CPUS --mem=$MEM --time=$TIME --job-name=wavepid_multidom_$c \
        --export=ALL,CLASS=$c,EVDIR=$EVDIR,TAG=$TAG,OUTROOT=$OUTROOT,OMSIM_BUILD=$OMSIM_BUILD,LAYOUT=$LAYOUT,WORLD=$WORLD,VERTEX=$VERTEX,EXTRA="$EXTRA" \
        $C/wavepid_multidom_array.sbatch); sim[$c]=${sim[$c]%%;*}
    # aftercorr: slim task i starts when production task i has COMPLETED (not when the whole array ends, and never for a failed task)
    slim=$(sbatch --parsable $MAILOPT --dependency=aftercorr:${sim[$c]} --array=$todo --job-name=wavepid_slim_$c \
        --export=ALL,CLASS=$c,TAG=$TAG,OUTROOT=$OUTROOT $C/wavepid_slim_array.sbatch); slim=${slim%%;*}
    n=$(echo "$todo" | tr ',' '\n' | wc -l)
    echo "$c: OMSim array ${sim[$c]} ($n tasks x $CHUNK events, $CPUS cores, $TIME), slim array $slim"
done
cat <<MSG

submitted. Logs in $W/logs. Follow with:   bash $C/run_1tev.sh status
When the slim arrays are done, on the Mac:
  for c in nue_cc nue_nc; do rsync -avP fasrc:$OUTROOT/cluster/\${c}_${TAG}_slim/ ~/IceCube/WavePID/PulseShapeInference/\${c}_${TAG}/; done
  rsync -avP fasrc:IceCube/wavepid/out/pythia_nue_1tev/{nue_cc_truth.csv,nue_nc_truth.csv} ~/IceCube/WavePID/PulseShapeInference/pythia_nue_1tev/
MSG
