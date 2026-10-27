#!/bin/bash
# Generate the ~1 TeV nu_e DIS sample with Pythia 8 (gen_pythia_events.py): N_PER_CLASS CC + N_PER_CLASS NC events,
# E_nu log-uniform in [EMIN, EMAX] GeV, CHUNK events per chunk file, into EVDIR with the same file names as the
# low-energy GENIE extraction (nue_cc_chunkNNN.txt, nue_cc_truth.csv, ...). Submitted by run_1tev.sh.
#SBATCH --job-name=pythia_1tev
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
set -eo pipefail
EVDIR=${EVDIR:-$HOME/IceCube/wavepid/out/pythia_nue_1tev}
N_PER_CLASS=${N_PER_CLASS:-6000}; CHUNK=${CHUNK:-100}
EMIN=${EMIN:-300}; EMAX=${EMAX:-3000}; SEED=${SEED:-1001}
source $HOME/IceCube/wavepid/cluster/env_fasrc.sh
python -c "import pythia8" 2>/dev/null || micromamba install -y -n omsim -c conda-forge pythia8
mkdir -p "$EVDIR"
python $HOME/IceCube/wavepid/cluster/gen_pythia_events.py --out-dir "$EVDIR" --n-per-class "$N_PER_CLASS" \
    --enu-min "$EMIN" --enu-max "$EMAX" --spectral-index 1 --flavour 12 --classes cc,nc \
    --chunk-size "$CHUNK" --seed "$SEED"
for c in nue_cc nue_nc; do
    n=$(ls "$EVDIR"/${c}_chunk*.txt | wc -l)
    echo "$c: $n chunk files, $(awk '!/^#/{print $1}' "$EVDIR"/${c}_events.txt | sort -un | wc -l) events"
done
