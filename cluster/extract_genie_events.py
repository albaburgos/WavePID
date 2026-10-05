#!/usr/bin/env python
"""Extract nu_e CC / NC final states from GENIE IceCube MC (e.g. dataset 23257)
into the text format read by OMSim_WavePID_study --event_file.

Needs the icetray python (cluster):
  eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.3/setup.sh)
  /cvmfs/icecube.opensciencegrid.org/py3-v4.4.3/icetray-env icetray/v1.19.1 \\
      python extract_genie_events.py --files "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/23257/*/*.i3.zst" \\
      --out-dir ~/IceCube/wavepid/out/genie_nue --max-files 20

Per Q frame: the neutrino primary's non-neutrino daughters (the lepton) plus the
daughters of the struck nucleus (hadrons, gammas, nuclear fragments) are the final
state. Directions are rotated into the frame where the neutrino travels along +x
(OMSim shoots along +x from (-1.7 d, 0, d)). Nuclei heavier than alpha, neutrinos
and particles below --emin are dropped. Output:
  <out-dir>/nue_cc_events.txt, nue_nc_events.txt   (event index restarts at 0 per class)
  <out-dir>/nue_cc_truth.csv, nue_nc_truth.csv     (per event: E_nu, y, n particles, weights ...)
"""
import argparse, csv, glob, math, os, sys
import numpy as np
from icecube import dataio, dataclasses, icetray  # noqa

# I3 ParticleType name -> Geant4 particle name (None = drop)
G4NAME = {
    "EMinus": "e-", "EPlus": "e+", "Gamma": "gamma", "MuMinus": "mu-", "MuPlus": "mu+",
    "PiPlus": "pi+", "PiMinus": "pi-", "Pi0": "pi0", "KPlus": "kaon+", "KMinus": "kaon-",
    "K0": "kaon0L", "K0Bar": "kaon0L", "K0_Long": "kaon0L", "K0_Short": "kaon0S", "Eta": "eta",
    "PPlus": "proton", "PMinus": "anti_proton", "Neutron": "neutron", "NeutronBar": "anti_neutron",
    "Lambda": "lambda", "LambdaBar": "anti_lambda", "SigmaPlus": "sigma+", "SigmaMinus": "sigma-", "Sigma0": "sigma0",
    "H2Nucleus": "deuteron", "H3Nucleus": "triton", "He3Nucleus": "He3", "He4Nucleus": "alpha",
    "TauMinus": "tau-", "TauPlus": "tau+",
    "D0": "D0", "D0Bar": "anti_D0", "DPlus": "D+", "DMinus": "D-", "DsPlus": "Ds+", "DsMinusBar": "Ds-",
    "LambdacPlus": "lambda_c+", "SigmaPlusBar": "anti_sigma+", "SigmaMinusBar": "anti_sigma-",
    "XiMinus": "xi-", "Xi0": "xi0", "OmegaMinus": "omega-",
}
SKIP_PREFIX = ("Nu", "unknown")


def rotation_to_x(nu_dir):
    """Rotation matrix R with R @ nu_dir = (1,0,0)."""
    a = np.asarray(nu_dir, float); a /= np.linalg.norm(a)
    b = np.array([1.0, 0.0, 0.0])
    v = np.cross(a, b); c = float(np.dot(a, b)); s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def final_state(tree):
    """(neutrino, [(type name, kinetic E GeV, dir)], y) or None.

    GENIE trees: neutrino -> lepton; nucleus -> Dark intermediates (Hadrons,
    resonances, pre-FSI hadrons) -> leaves with shape Cascade = the particles that
    actually leave the nucleus. Take every non-Dark leaf; skip neutrinos, the
    recoiling nucleus and its <MeV "Hadrons" remnant. NuGen trees have a single
    "Hadrons" leaf, handed through so the caller can substitute a stand-in.
    """
    nus = [p for p in tree.get_primaries() if p.is_neutrino]
    if not nus:
        return None
    nu = nus[0]
    primary_ids = {p.id for p in tree.get_primaries()}
    parts, e_lep = [], 0.0
    for d in tree:
        if len(tree.get_daughters(d)) > 0 or d.id in primary_ids:
            continue
        shape = str(d.shape).split(".")[-1]
        if shape == "Dark":
            continue
        name = str(d.type).split(".")[-1]
        if name.startswith(SKIP_PREFIX) or d.is_neutrino:
            continue
        if name in ("EMinus", "EPlus", "MuMinus", "MuPlus", "TauMinus", "TauPlus"):
            e_lep += d.energy
        parts.append((name, d.energy, np.array([d.dir.x, d.dir.y, d.dir.z])))
    y = 1.0 - e_lep / nu.energy if nu.energy > 0 else float("nan")
    return nu, parts, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=os.environ.get("GENIE_FILES"),
                    help="glob of .i3(.zst) files; icetray-env re-expands quoted globs, so prefer "
                         "exporting GENIE_FILES='<glob>' and omitting this flag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--emin", type=float, default=0.005, help="drop particles below this kinetic energy [GeV]")
    ap.add_argument("--hadrons-as", default="pi-", help="G4 name used for NuGen 'Hadrons' pseudo-particles")
    ap.add_argument("--enu-min", type=float, default=0.0, help="keep events with E_nu >= this [GeV]")
    ap.add_argument("--enu-max", type=float, default=1e9, help="keep events with E_nu <= this [GeV]")
    ap.add_argument("--max-per-class", type=int, default=0, help="stop a class after this many events (0 = all)")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="also write <tag>_chunkNNN.txt files of this many events (indices restart per chunk) "
                         "for the Slurm array; the truth csv gets chunk/global_index columns")
    args = ap.parse_args()
    if not args.files:
        raise SystemExit("give --files or export GENIE_FILES")
    files = sorted(glob.glob(os.path.expanduser(args.files)))
    if args.max_files:
        files = files[: args.max_files]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(files)} files")

    HEADER = "# <event index> <G4 name> <kinetic E GeV> <dx> <dy> <dz>   (neutrino along +x)\n"
    writers, truth, counts, chunks, tags = {}, {}, {1: 0, 2: 0}, {}, {1: "nue_cc", 2: "nue_nc"}
    for it, tag in tags.items():
        writers[it] = open(os.path.join(args.out_dir, f"{tag}_events.txt"), "w")
        writers[it].write(HEADER)
        truth[it] = csv.writer(open(os.path.join(args.out_dir, f"{tag}_truth.csv"), "w"))
        truth[it].writerow(["global_index", "chunk", "index", "file", "run", "event", "interaction", "e_nu_gev", "y",
                            "n_particles", "e_visible_gev", "one_weight", "n_events_gen", "particles"])
        chunks[it] = None

    def chunk_writer(it, gidx):
        if not args.chunk_size:
            return None
        k = gidx // args.chunk_size
        if chunks[it] is None or chunks[it][0] != k:
            if chunks[it] is not None:
                chunks[it][1].close()
            w = open(os.path.join(args.out_dir, f"{tags[it]}_chunk{k:03d}.txt"), "w"); w.write(HEADER)
            chunks[it] = (k, w)
        return chunks[it][1]
    dropped = {}
    for fi, path in enumerate(files):
        f = dataio.I3File(path)
        while f.more():
            fr = f.pop_frame()
            if fr is None:
                break
            if fr.Stop != icetray.I3Frame.DAQ or "I3MCTree" not in fr or "I3MCWeightDict" not in fr:
                continue
            wd = fr["I3MCWeightDict"]
            it = int(wd["InteractionType"]) if "InteractionType" in wd else 0
            if it not in (1, 2):
                continue
            if args.max_per_class and counts[it] >= args.max_per_class:
                if all(args.max_per_class and counts[c] >= args.max_per_class for c in (1, 2)):
                    break
                continue
            fs = final_state(fr["I3MCTree"])
            if fs is None:
                continue
            nu, parts, y = fs
            if not (args.enu_min <= nu.energy <= args.enu_max):
                continue
            R = rotation_to_x([nu.dir.x, nu.dir.y, nu.dir.z])
            gidx = counts[it]
            idx = gidx % args.chunk_size if args.chunk_size else gidx
            lines, names, e_vis = [], [], 0.0
            for name, e, d in parts:
                if name == "Hadrons" and e < 0.05:      # nuclear-remnant pseudo particle (GENIE)
                    continue
                g4 = args.hadrons_as if name == "Hadrons" else G4NAME.get(name)
                if g4 is None:
                    dropped[name] = dropped.get(name, 0) + 1
                    continue
                if e < args.emin:
                    continue
                dr = R @ d
                lines.append(f"{idx} {g4} {e:.6g} {dr[0]:.6f} {dr[1]:.6f} {dr[2]:.6f}\n")
                names.append(g4); e_vis += e
            if not lines:                    # nothing visible: still keep the event (empty) for bookkeeping
                lines.append(f"{idx} geantino 0.001 1 0 0\n"); names.append("geantino")
            writers[it].writelines(l.replace(f"{idx} ", f"{gidx} ", 1) for l in lines)
            cw = chunk_writer(it, gidx)
            if cw is not None:
                cw.writelines(lines)
            hdr = fr["I3EventHeader"] if "I3EventHeader" in fr else None
            truth[it].writerow([gidx, gidx // args.chunk_size if args.chunk_size else 0, idx,
                                os.path.basename(path), hdr.run_id if hdr else -1, hdr.event_id if hdr else -1,
                                it, f"{nu.energy:.6g}", f"{y:.5f}", len(names), f"{e_vis:.6g}",
                                f"{wd.get('OneWeight', float('nan')):.6g}", f"{wd.get('NEvents', float('nan')):.6g}",
                                " ".join(names)])
            counts[it] += 1
        f.close()
        if args.max_per_class and all(counts[c] >= args.max_per_class for c in (1, 2)):
            print(f"  reached {args.max_per_class} events per class after {fi+1} files"); break
        if (fi + 1) % 10 == 0:
            print(f"  {fi+1} files: CC {counts[1]}, NC {counts[2]}", flush=True)
    for w in writers.values():
        w.close()
    for c in chunks.values():
        if c is not None:
            c[1].close()
    print(f"done: CC {counts[1]} events, NC {counts[2]} events -> {args.out_dir}")
    if dropped:
        print("dropped particle types (no Geant4 mapping):", dict(sorted(dropped.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
