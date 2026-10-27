# WavePID: NC/CC discrimination of ν_e cascades from DOM photon arrival times

Neutral-current and charged-current ν_e cascades look alike in a neutrino telescope. This repository
tests whether the *longitudinal profile* of the cascade, recovered from the photon arrival times at the
optical modules, separates them:

1. simulate ν_e CC and NC final states (Pythia 8 DIS or GENIE) through Geant4 light propagation with a
   patched [OMSim](https://github.com/icecube/OMSim) WavePID study, on a lattice or the DeepCore layout;
2. label every event with the Gamma-function fit of its longitudinal shower profile, (ln α, ln β);
3. infer (ln α, ln β) from the DOM pulses alone with a [Neptune](https://github.com/felixyu7/neptune)
   set transformer (bivariate Gaussian posterior trained by conditional NLL);
4. classify NC vs CC from the two profile parameters and tabulate the classifier response for an
   oscillation analysis.

## Layout

| step | files |
|---|---|
| OMSim patch | `omsim/omsim_wavepid.patch`: all local changes to icecube/OMSim at `dc9c247c` (hadronic physics, photon/charged-track recording, multi-module layouts incl. `file:` layouts, random vertex and rotation, photon kill distance, photon-fraction speed-up, QE cut at the photocathode, periodic AutoSave). Apply with `git apply` in an OMSim checkout of that commit. `cluster/patch_omsim_deepcore.py` re-applies the DeepCore part alone (idempotent; already contained in the patch). |
| FASRC setup | `cluster/setup_fasrc.sh` (micromamba env + OMSim build; submit through `cluster/setup_job.sbatch`), `cluster/env_fasrc.sh`, `cluster/rebuild_omsim.sbatch` / `cluster/rebuild_fasrc.sbatch` (incremental rebuilds), `cluster/sync_to_fasrc.sh` (ships OMSim + scripts from a Mac; needs `ssh fasrc`) |
| events | `cluster/gen_pythia_events.py` (Pythia 8 ν DIS CC/NC on p and n, + `cluster/pythia_lowE_check.py`), `cluster/extract_genie_events.py` + `cluster/run_extract.sh` + `cluster/genie_extract_gev.sbatch` (GENIE dataset 23257, needs icetray) |
| geometry | `cluster/make_deepcore_layout.py` → `cluster/deepcore_layout.csv` (for `--dom_layout file:<path>`) |
| production | drivers `cluster/run_gev.sh` (E_vis 9–100 GeV, DeepCore geometry by default) and `cluster/run_1tev.sh` (~1 TeV, lattice; uses `cluster/gen_pythia_1tev.sbatch`); Slurm arrays `cluster/wavepid_gev_array.sbatch`, `cluster/wavepid_evisz_array.sbatch`, `cluster/wavepid_multidom_array.sbatch`, `cluster/wavepid_slim_array.sbatch` (fits the labels, then keeps only the `PhotonHits` and `Events` trees, ~70× smaller) |
| labels | `gamma_functions.py part_NNNN_hits.root --all-events` → `part_NNNN_profile_params.csv` (shower axis and Gamma fit per event; needs the `PhotonTracks` tree) |
| inference | `PulseShapeInference/pulse_regressor.py` (dataset build + Neptune posterior over (ln α, ln β)), `run_gev_regressor.py` (build → train → classify for a GeV production), `weights.py` (visible-energy matching, flux weights, class-prior shift) |
| classification | `PulseShapeInference/classify_gamma.py` (NC/CC from (ln α, ln β), true or inferred, with QDA reference), `vis.py` (Bokeh view of P(NC)), `classifier_response.py` (response table R(E) per class for the oscillation fit) |

Each script documents its options and outputs in its docstring or header comments.

## Requirements

* Simulation: Geant4 11.3, ROOT, Boost and spdlog (the environment `cluster/setup_fasrc.sh` builds), OMSim with the patch above.
* Events: the `pythia8` Python bindings; icetray for the GENIE extraction.
* Analysis: Python with numpy, scipy, pandas, uproot, torch, lightning, scikit-learn and bokeh, plus
  [Neptune](https://github.com/felixyu7/neptune) (found as a `neptune/` directory in a parent folder) and the `iwpc` package.

## Not included

* Simulation outputs, datasets and checkpoints (ROOT, npz, parquet; see `.gitignore`).
* The newest flat-E_vis production (`cluster/wavepid_evisz_array.sbatch`) also uses `merge_evis_jobs.py`, the
  `deepcore224*_layout.csv` layouts and a `gen_pythia_events.py` with `--evis-min/--evis-max/--axis`; those live only in
  the FASRC copy and are not in this snapshot.
* Exploratory notebooks, the earlier single-DOM study and personal tooling.
