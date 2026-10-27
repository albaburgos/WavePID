#!/usr/bin/env python
"""Local step after the download: build the GeV dataset, train the Neptune (ln a, ln b) posterior, then the Gamma
NC/CC classifier. Only passes the Settings fields the local pulse_regressor.py actually has (the repo and the Mac
copies differ: `tag`, `run`, the 8-parameter head), so it works with either.

  python run_gev_regressor.py                 # DeepCore-geometry GeV parts (nue_{cc,nc}_gev_deepcore, tag gev)
  python run_gev_regressor.py --parts-tag gev_multidom --with-1tev --augment full
                                              # lattice-geometry GeV + 1 TeV parts: one regressor over 5 GeV - 3 TeV (tag gev1tev)

Rotation augmentation (--augment): the 1 TeV lattice is isotropic, so pulse_regressor's random 3-D rotation of the
token positions about the vertex is a symmetry there ("full"). The DeepCore layout is not: strings are vertical and
7 m apart in z against 45-72 m in x, y, so a 3-D rotation produces detector geometries that do not exist. Default
"none" (the fixed real geometry is what the network should learn; the vertex is already uniform over a unit cell);
"z" rotates about the vertical axis only (exact only for the 60-degree steps of the hexagon, a mild augmentation).
  python run_gev_regressor.py --epochs 300 --accelerator mps
  python run_gev_regressor.py --evaluate logs/neptune_gev/version_0/checkpoints/best-epoch=NN.ckpt

Then (truth-level and predicted-profile classifiers, matched in E_vis, flux-weighted evaluation as in classify_gamma.py):
  python classify_gamma.py --source truth --tag gev --build --train
  python classify_gamma.py --source pred  --tag gev --build --train
  python plot_energy_weighted.py --tag gev
"""
import argparse
import dataclasses

import numpy as np
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pulse_regressor as pr  # noqa: E402
from pulse_regressor import Settings, main  # noqa: E402


def z_rotation(rng):
    """Random rotation about the vertical axis (replaces pulse_regressor.random_rotation for --augment z)."""
    th = rng.uniform(0.0, 2.0 * np.pi); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


def settings(**kw):
    fields = {f.name for f in dataclasses.fields(Settings)}
    dropped = sorted(set(kw) - fields)
    if dropped:
        print(f"(local pulse_regressor.Settings has no {dropped}; ignored)")
    return Settings(**{k: v for k, v in kw.items() if k in fields})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts-tag", default="gev_deepcore", help="local part dirs PulseShapeInference/nue_{cc,nc}_<tag>/")
    ap.add_argument("--with-1tev", action="store_true", help="also build from nue_{cc,nc}_1tev_multidom/ (lattice geometry!)")
    ap.add_argument("--augment", choices=["none", "z", "full"], default="none", help="rotation augmentation, see above")
    ap.add_argument("--tag", default=None, help="dataset/run tag (default gev, or gev1tev with --with-1tev)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--evaluate", type=Path, default=None)
    ap.add_argument("--classify", action="store_true", help="run classify_gamma.py (truth and pred) afterwards")
    a = ap.parse_args()

    tag = a.tag or ("gev1tev" if a.with_1tev else "gev")
    if a.with_1tev and "deepcore" in a.parts_tag:
        print("WARNING: mixing DeepCore-geometry GeV parts with the lattice-geometry 1 TeV parts; the token geometry "
              "differs, so the regressor sees two detectors. Use --parts-tag gev_multidom for a mixed sample.")
    if a.augment == "z":
        pr.random_rotation = z_rotation          # EventSet looks the function up in the module at call time
    parts = [ROOT / f"nue_{c}_{a.parts_tag}" for c in ("cc", "nc")]
    if a.with_1tev:
        parts += [ROOT / f"nue_{c}_1tev_multidom" for c in ("cc", "nc")]
    missing = [p for p in parts if not p.is_dir()]
    if missing:
        raise SystemExit(f"missing part directories (run `run_gev.sh download` on the cluster first): {missing}")
    for p in parts:
        n = len(list(p.glob("part_*_hits.root"))); m = len(list(p.glob("part_*_profile_params.csv")))
        print(f"{p.name}: {n} hits parts, {m} profile-label parts")

    main(settings(build=parts, tag=tag, run="none", train=not a.no_train and a.evaluate is None,
                  evaluate=a.evaluate, epochs=a.epochs, accelerator=a.accelerator, min_hits=a.min_hits,
                  known_energy=False, known_axis=False, augment=a.augment != "none"))
    if a.classify:
        for src in ("truth", "pred"):
            subprocess.run([sys.executable, str(ROOT / "classify_gamma.py"), "--source", src, "--tag", tag,
                            "--build", "--train"], check=True)
        subprocess.run([sys.executable, str(ROOT / "plot_energy_weighted.py"), "--tag", tag], check=True)
