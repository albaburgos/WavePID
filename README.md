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


