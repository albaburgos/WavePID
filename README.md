# WavePID photon arrival times: CC-like vs NC-like

OMSim's WavePID study shoots a single primary past a DOM and records every photon
reaching the photocathode with its arrival time and origin. It does not simulate
neutrino interactions, so NC/CC is mapped onto primaries:

| primary | what it models | neutrino-event analogue |
|---|---|---|
| `e-`  | electromagnetic cascade | EM part of a CC nu_e event |
| `pi-` (needs `--hadronic`) | hadronic cascade | NC event of any flavour / hadronic vertex |
| `mu-` | minimum-ionising track | CC nu_mu (OMSim's reference case) |

A real CC nu_e event is EM + hadronic cascade at one vertex; approximate it by
combining `e-` and `pi-` samples with the appropriate energy split (y-distribution).

## Setup (done on this Mac, 2026-09-20)
* Geant4 11.3.2, ROOT, Boost, spdlog from conda-forge in micromamba env `omsim`.
* OMSim at `~/IceCube/OMSim`, built in `build/`. Local patches (originals in `*.orig`):
  CMake OpenSSL path conditional, `wavepid` enabled, VLA in `OMSimHitManager.cc`,
  `--hadronic` option (FTFP_BERT + hadron EM) in the wavepid physics list, and a mutex
  around `TTree::Fill` in `ROOTHitManager.cc` (the upstream code crashes in
  multithreaded mode under load), and Cherenkov `TrackSecondariesFirst` switched off
  by default (`--track_secondaries_first` restores it). With it on, the parent track is
  suspended after every Cherenkov step and Geant4 resets the hadronic interaction
  length, so a hadron primary never showers: 10 GeV pi- gave 38 photons/event instead
  of ~85. EM and muon results are unaffected by this switch.
* 2026-09-23 (`*.orig2` = state before): `--photon_tracks F` writes a `PhotonTracks` tree
  (step points in mm/ns of every non-photon track, every detected photon and a fraction F
  of the other photons; `OMSimTrackingAction` buffers the current track, `OMSimSteppingAction`
  appends steps, the sensitive detector flags detections). The tree is only written when
  non-empty, so production files are unchanged. Same patch fixes `deltaPos_x/y/z`: it used
  to be (|Δ|, 0, 0); now it is the vector from the hit to the photon's creation point
  (emission = globalPos + deltaPos; `generationDetectionDistance` = |deltaPos| as before).

## Run
```
source env.sh
./run_wavepid.sh 200 10 5 12          # N events, E [GeV], impact parameter [m], threads
../venv/bin/python plot_arrival_times.py     # -> out/arrival_times_E10_d5.png, out/origin_breakdown_E10_d5.png
```
Any primary Geant4 knows works with `--hadronic`, e.g. `-p proton`, `-p kaon-`, `-p pi+`.
Output: `out/<tag>_E<E>_d<d>_hits.root`, tree `PhotonHits` (hitTime, photonOrigin,
parentType, parentProcess, wavelength, positions, ...), plus `<...>_args.json`.

## One GENIE event, full DOM, Geant4 light propagation
```
mkdir -p out/genie_nue     # inputs from the cluster (extract_genie_events.py --chunk-size 200)
rsync -avP fasrc:IceCube/wavepid/out/genie_nue/{nue_nc_chunk000.txt,nue_nc_truth.csv} out/genie_nue/
./run_genie_event.sh out/genie_nue/nue_nc_chunk000.txt 0 5 0.02    # file, event, d [m], photon fraction
```
`run_genie_event.sh` cuts one event out of the chunk (renumbered to 0), runs it single-threaded
with `--detector_type 3 --environment 2 --hadronic --photon_tracks F` and no photon kill, then
`plot_light_propagation.py` draws `out/vis/<file>_ev<K>_paths.png` (side and top view: cascade
by particle category, photon paths coloured by the time along the path, detected photons to the
DOM), `_snapshots.png` (positions of all recorded photons at fixed times, cascade drawn up to
t, hits accumulating on the DOM; `--times`), and with `--gif` an animation. A 1.4 GeV
NC-like event takes ~3 s in OMSim and ~10 s to plot; ~5% of its photons are emitted after
100 ns by neutron-capture gammas (the colour scales ignore them). Neutrinos are not drawn.
The Geant4 `-v` viewer is not available on this Mac (conda-forge Geant4 has no Qt/OpenGL).

## NC/CC benchmark for ../classifier.py
* `run_benchmark_pools.sh [N_EM] [N_HAD] [EMIN] [EMAX] [d] [threads]` simulates two pools
  with log-uniform energy spectra (GPS `Pow` alpha=-1): `out/benchmark/em_pool_hits.root`
  (e-) and `out/benchmark/pim_pool_hits.root` (pi-, `--hadronic`). Each file has the
  usual `PhotonHits` tree plus an `Events` tree (eventID, primaryEnergy [GeV],
  primaryParticle) with one row per simulated event, including zero-hit events.
* `python ../classifier.py --build` composes NC (label 1) = one pi- event and
  CC (label 0) = one pi- event + one e- event superposed, with shared E_nu and y
  distributions, and writes `../classifier_data/` (iwpc) plus
  `../classifier_data_truth.parquet` (truth, features, raw hit times) and
  `../classifier_data_templates.npz` (arrival-time templates for the mixture fit).
* `python ../classifier.py --train` trains the MLP and prints validation AUC.
* `--fixed-energy --em-pool out/em_E10_d5_hits.root --had-pool out/pim_E10_d5_hits.root`
  runs the same pipeline on the 10 GeV samples (separable mostly by photon count).
* Real GENIE events: `python ../classifier.py --build --genie out/nue_cc_E2-50_d5_hits.root out/nue_nc_E2-50_d5_hits.root`
  labels every complete CC/NC final state by its file (nothing superposed), pairs each usable NC event
  with the unused CC event closest in visible energy so both classes share the E_vis spectrum
  (`--no-match` keeps all usable events), keeps the mixture templates from the single-particle pools
  (`--templates-from genie` measures them from the two classes) and joins E_nu, y from `<tag>_truth.csv`
  in `--truth-dir` (default `out/genie_nue`, then `out/vis`). The dataset goes to `../classifier_data_genie/`
  (+ `_truth.parquet`, `_meta.json`, `_templates.npz`); train it with `python ../classifier.py --train --dataset-tag genie`.

## Large impact parameters and the FASRC cluster
* New OMSim options: `--world_radius R` (m, default 30; must exceed 1.7 d so the primary
  starts inside the ice) and `--photon_kill_distance K` (m, default 0 = off): optical
  photons farther than K from the module and moving away from it are killed. Validated
  at d = 5 m, K = 8 m: detected yield and timing unchanged, 1.65x faster; the gain grows
  with d. Use K well beyond the far end of the track (`1.4 d + 5` in the sbatch).
* Several modules, random vertex / rotation (training data for `pulse_regressor.py`): `--dom_layout fill:S[:R]`
  places a module on every lattice point of spacing S m within R m of the origin (default for the cluster: `fill:10:25`,
  81 DOMs, so every module near any vertex exists and records); `string:N:S` and `grid:NX:NY:NZ:S` give fixed
  small layouts (positions in `<out>_doms.json`,
  every hit carries `moduleID`); `--random_vertex R` draws each event's vertex in `[-R, R]^3` m and
  `--random_rotation` rotates each final state uniformly; both are written to the `Events` tree (`vx vy vz`,
  `nu_dx nu_dy nu_dz`). `--photon_tracks 1e-9` records all particle tracks (almost no photons) so
  `gamma_functions.py part_NNNN_hits.root --all-events` fits the shower axis and Gamma profile per event
  (`_profile_params.csv`). `cluster/wavepid_multidom_array.sbatch` runs this per chunk. The hits files carry the
  `PhotonTracks` tree (MBs per event); before copying them to the Mac drop it on the cluster with
  `rootcp part_NNNN_hits.root:PhotonHits part_NNNN_hits.root:Events slim/part_NNNN_hits.root` (pulse_regressor.py
  reads only `PhotonHits` and `Events`; ~70x smaller). `cluster/wavepid_slim_array.sbatch` does this per part as a
  Slurm array (and first fits the labels where `_profile_params.csv` is missing), writing `<class>_multidom_slim/`.
* `PulseShapeInference/` (scripts and all their outputs) infers the profile y = (ln a, ln b) of each event from its
  pulses, with the vertex known and the energy and axis as nuisances (maths in the docstrings; follows
  `Claude outputs/nc_cc_profile_inference.pdf`). From that directory:
  `python pulse_regressor.py --build ../out/nue_cc_multidom ../out/nue_nc_multidom` reads the parts (slim hits ROOT +
  `_doms.json` + `_profile_params.csv`) into `pulse_regressor_truth.parquet` with a fixed 20 % validation split;
  `--neptune` trains solution A (Neptune set transformer over the hit modules, tokens relative to the vertex, bivariate
  Gaussian head trained by conditional NLL, random rotations of the training events, best validation epoch kept) and
  writes `pulse_regressor_neptune_val.npz` (mu, Sigma per event); `--known-energy` and `--known-axis` hand the network
  the energy and the shower axis as well (kappa = (v, E, u) all known, only the profile left); `--mlp` is a flat-MLP
  baseline with the same head;
  `--evaluate CKPT` re-evaluates a checkpoint from `logs/`; `inspect_posterior.py` prints the resolution in bins of
  energy and hits, the pulls and the calibration of a prediction file and draws `figs/<stem>_posterior.png`.
  `pulse_regressor_inference.tex/.pdf` (2026-09-30) is the write-up of the inference the script trains in its
  known-vertex, known-energy, known-axis configuration: data, labels, posterior with the class marginalised, NLL
  objective and its minimiser, the network with ln E and cos(module, axis) on every token, rotation augmentation,
  validation, and the results of the combined CC+NC run plus the stage-2 NC/CC numbers. The .tex compiles with
  pdflatex (none on the Mac; the PDF was rendered from the same source through MathJax and headless Chrome).
  `classify_gamma.py --source truth|pred --build --train` classifies NC/CC from exactly two inputs, (ln alpha, ln beta),
  read from `pulse_regressor_neptune_val.npz`: the true profile (`truth`, the ceiling of any profile-based method) or
  the Neptune posterior mean (`pred`), on the same events. Classes are energy-matched (CC has a 6x higher median E_vis and
  alpha grows with ln E); an iwpc MLP with class-balanced BCE learns P(NC | x); the printout gives the AUC overall and per
  energy bin next to QDA on the same events and, for `pred`, the note's Lambda with the per-event Sigma and the truth
  ceiling. `bokeh serve --show vis.py [--args pred]` shows P(NC) over the plane for the newest run;
  `plot_energy_weighted.py` draws the true and P-weighted energy spectra plus ROC for all events of the file.
  `profile_likelihood.py` is solution B (per-event
  profile-likelihood fit with an analytic photon kernel, E maximised out, `--calibrate-kernel` fits the kernel constants
  on labelled events): exact on synthetic events, not adequate on OMSim events with the analytic kernel (see its docstring).
  On 4316 NC events Neptune reached median errors 0.23 / 0.29 / 0.18 in ln a / ln b / ln(ab) (prior 0.30 / 0.31 / 0.36).
* `cluster/sync_to_fasrc.sh` ships the patched OMSim + these scripts (needs `ssh fasrc` open),
  `cluster/setup_fasrc.sh` builds the same micromamba env and OMSim on the cluster,
  `cluster/wavepid_array.sbatch` is a Slurm array (one chunk of events per task, unique seeds),
  `cluster/merge_parts.py` merges the chunks into one `<tag>_E<E>_d<d>_hits.root`
  that plot_arrival_times.py and classifier.py read directly.
