#!/usr/bin/env python
"""Generate nu_e (or nu_mu) deep-inelastic CC / NC final states with Pythia 8 and write
them in the text format read by OMSim_WavePID_study --event_file.

Same outputs as extract_genie_events.py (GENIE), so the Slurm array, merge and
analysis scripts are shared:
  <out-dir>/nue_cc_events.txt, nue_nc_events.txt      one particle per line,
                                                      "<event idx> <G4 name> <E_kin GeV> <dx> <dy> <dz>",
                                                      neutrino along +x
  <out-dir>/nue_cc_chunkNNN.txt ...                   --chunk-size events per file, indices restart at 0
  <out-dir>/nue_cc_truth.csv, nue_nc_truth.csv        one row per event (same 14 columns as GENIE + extras)

Physics: Pythia 8 W exchange (CC) or gamma*/Z exchange (NC) on a free proton or neutron
at rest (p:n = 10:8 as in water), Q^2 > --q2min. Valid where DIS dominates (E_nu above
~10 GeV); no nuclear effects, no QE/RES. Resonances (rho, K*, Delta, eta, pi0 ...)
are decayed by Pythia because Geant4 cannot transport them as primaries; hadrons with
c*tau > --tau0max-mm (pi+-, K+-, K0L, K0S, Lambda, n ...) are handed to Geant4.
--print-undecayed also lists the primary hadrons before those decays.

Pythia cannot change the beam energy inside one instance for DIS processes, so a fresh
Pythia is built per event (~60 ms). 12k events ~ 12 min on one core.

Example (10 + 10 events at 100 TeV, one event per chunk file, i.e. per Slurm task):
  python gen_pythia_events.py --out-dir out/pythia_nue_100TeV --n-per-class 10 \
      --enu-min 1e5 --enu-max 1e5 --chunk-size 1 --print-undecayed --seed 1
"""
import argparse, csv, math, os, sys, time
import numpy as np
import pythia8

MB_TO_CM2 = 1e-27
NEUTRINOS = {12, 14, 16}
TARGETS = ((2212, "p", 10.0), (2112, "n", 8.0))            # water: 10 protons, 8 neutrons

# PDG code -> Geant4 particle name (None / missing = dropped and reported)
G4NAME = {
    11: "e-", -11: "e+", 22: "gamma", 13: "mu-", -13: "mu+", 15: "tau-", -15: "tau+",
    211: "pi+", -211: "pi-", 111: "pi0", 221: "eta", 331: "eta_prime",
    321: "kaon+", -321: "kaon-", 130: "kaon0L", 310: "kaon0S", 311: "kaon0", -311: "anti_kaon0",
    2212: "proton", -2212: "anti_proton", 2112: "neutron", -2112: "anti_neutron",
    3122: "lambda", -3122: "anti_lambda", 3222: "sigma+", -3222: "anti_sigma+",
    3112: "sigma-", -3112: "anti_sigma-", 3212: "sigma0", -3212: "anti_sigma0",
    3322: "xi0", -3322: "anti_xi0", 3312: "xi-", -3312: "anti_xi-", 3334: "omega-", -3334: "anti_omega-",
    411: "D+", -411: "D-", 421: "D0", -421: "anti_D0", 431: "Ds+", -431: "Ds-",
    4122: "lambda_c+", -4122: "anti_lambda_c+", 4222: "sigma_c++", 4212: "sigma_c+", 4112: "sigma_c0",
    521: "B+", -521: "B-", 511: "B0", -511: "anti_B0", 531: "Bs0", -531: "anti_Bs0", 5122: "lambda_b",
    1000010020: "deuteron", 1000010030: "triton", 1000020030: "He3", 1000020040: "alpha",
}


def pythia_settings(a, flavour, target_id, enu, process, seed):
    return [
        "Beams:frameType = 2", f"Beams:idA = {flavour}", f"Beams:idB = {target_id}",
        f"Beams:eA = {enu:.10g}", "Beams:eB = 0.",                      # target at rest
        f"{process} = on",
        f"PhaseSpace:Q2Min = {a.q2min}", f"PhaseSpace:mHatMin = {a.mhatmin}",
        "PhaseSpace:pTHatMinDiverge = 1.0",
        "SigmaProcess:factorScale2 = 6", "SigmaProcess:renormScale2 = 6",   # Q^2 scales (DIS)
        "PDF:lepton = off", "SpaceShower:dipoleRecoil = on", "PartonLevel:MPI = off",
        "ParticleDecays:limitTau0 = on", f"ParticleDecays:tau0Max = {a.tau0max_mm}",
        "Random:setSeed = on", f"Random:seed = {seed % 900000000}",
        "Print:quiet = on", "Next:numberCount = 0",
    ]


def make_pythia(a, flavour, target_id, enu, process, seed):
    p = pythia8.Pythia("", False)
    for s in pythia_settings(a, flavour, target_id, enu, process, seed):
        p.readString(s)
    if not p.init():
        raise RuntimeError(f"Pythia init failed for E={enu} target={target_id} {process}")
    return p


def outgoing_lepton(ev):
    """Final copy of the hard-process outgoing lepton (e-/nu_e for nu_e beams)."""
    for i in range(ev.size()):
        q = ev[i]
        if abs(q.status()) == 23 and abs(q.id()) in (11, 12, 13, 14, 15, 16):
            return ev[q.iBotCopyId()]
    raise RuntimeError("no outgoing lepton with status 23 found")


def analyse_event(ev, enu, m_target):
    lep = outgoing_lepton(ev)
    y = 1.0 - lep.e() / enu
    # q = p_nu - p_lep with p_nu = (enu; 0, 0, enu) in Pythia's frame (beam A along +z)
    q0, qx, qy, qz = enu - lep.e(), -lep.px(), -lep.py(), enu - lep.pz()
    q2 = -(q0 * q0 - qx * qx - qy * qy - qz * qz)
    w2 = m_target * m_target + 2.0 * m_target * q0 - q2
    final, undecayed = [], []
    for i in range(ev.size()):
        q = ev[i]
        st = abs(q.status())
        if 81 <= st <= 89:                                   # primary hadrons from string fragmentation
            undecayed.append((q.name(), q.e()))
        if not q.isFinal() or abs(q.id()) in NEUTRINOS:
            continue
        final.append(q)
    undecayed.append((lep.name(), lep.e()))
    return lep, y, q2, math.sqrt(max(w2, 0.0)), final, undecayed


def sample_energy(rng, a):
    if a.enu_max <= a.enu_min:
        return a.enu_min, 1.0                                # fixed energy: p_gen = 1 by convention
    lo, hi, g = a.enu_min, a.enu_max, a.spectral_index
    if abs(g - 1.0) < 1e-9 or g == 0.0:                     # log-uniform (E^-1)
        e = lo * (hi / lo) ** rng.random()
        return e, 1.0 / (e * math.log(hi / lo))
    u = rng.random()                                          # E^-g via inverse CDF
    e = ((1 - u) * lo ** (1 - g) + u * hi ** (1 - g)) ** (1.0 / (1 - g))
    norm = (hi ** (1 - g) - lo ** (1 - g)) / (1 - g)
    return e, e ** (-g) / norm


def sigma_table(a, flavour, process, energies, base_seed):
    """sigma [cm^2] per nucleon type on a grid of energies (Pythia's own estimate)."""
    tab = {}
    for target_id, tname, _ in TARGETS:
        vals = []
        for k, e in enumerate(energies):
            p = make_pythia(a, flavour, target_id, e, process, base_seed + 7919 * k + target_id)
            n_ok = 0
            for _ in range(a.sigma_events):
                n_ok += bool(p.next())
            vals.append(p.infoPython().sigmaGen() * MB_TO_CM2)
        tab[tname] = np.array(vals)
    return tab


def interp_sigma(tab, tname, energies, e):
    if len(energies) == 1:
        return float(tab[tname][0])
    return float(np.exp(np.interp(math.log(e), np.log(energies), np.log(tab[tname]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--enu-min", type=float, default=1e5, help="GeV; equal to --enu-max -> fixed energy")
    ap.add_argument("--enu-max", type=float, default=1e5)
    ap.add_argument("--spectral-index", type=float, default=0.0,
                    help="0 or 1 = log-uniform in E, otherwise sample E^-gamma")
    ap.add_argument("--flavour", type=int, default=12, choices=(12, 14, 16, -12, -14, -16), help="beam PDG id")
    ap.add_argument("--classes", default="cc,nc")
    ap.add_argument("--chunk-size", type=int, default=1, help="events per chunk file (0 = no chunk files)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--q2min", type=float, default=1.0, help="GeV^2")
    ap.add_argument("--mhatmin", type=float, default=1.0, help="GeV (Pythia default 4 is above sqrt(s) at 10 GeV)")
    ap.add_argument("--emin", type=float, default=0.005, help="drop particles below this kinetic energy [GeV]")
    ap.add_argument("--tau0max-mm", type=float, default=10.0,
                    help="Pythia decays particles with c*tau below this; longer-lived ones go to Geant4")
    ap.add_argument("--print-undecayed", action="store_true",
                    help="print the primary-hadron list (before Pythia decays) + lepton for every event")
    ap.add_argument("--sigma-grid", type=int, default=8, help="energies for the cross-section table")
    ap.add_argument("--sigma-events", type=int, default=1000, help="Pythia events per table point")
    a = ap.parse_args()

    tag_root = {12: "nue", 14: "numu", 16: "nutau"}[abs(a.flavour)] + ("bar" if a.flavour < 0 else "")
    procs = {"cc": ("WeakBosonExchange:ff2ff(t:W)", 1), "nc": ("WeakBosonExchange:ff2ff(t:gmZ)", 2)}
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fixed = a.enu_max <= a.enu_min
    grid = np.array([a.enu_min]) if fixed else np.geomspace(a.enu_min, a.enu_max, a.sigma_grid)
    pdata = pythia8.Pythia("", False).particleData
    m_target = {tid: pdata.m0(tid) for tid, _, _ in TARGETS}
    t_prob = np.array([w for _, _, w in TARGETS]); t_prob /= t_prob.sum()
    HEADER = "# <event index> <G4 name> <kinetic E GeV> <dx> <dy> <dz>   (neutrino along +x)\n"
    TRUTH_COLS = ["global_index", "chunk", "index", "file", "run", "event", "interaction", "e_nu_gev", "y",
                  "n_particles", "e_visible_gev", "one_weight", "n_events_gen", "particles",
                  "target", "q2", "w_had", "sigma_cm2", "n_retries", "undecayed"]
    dropped, t_start = {}, time.time()

    for cls in classes:
        process, itype = procs[cls]
        tag = f"{tag_root}_{cls}"
        print(f"== {tag}: cross-section table on {len(grid)} energies ...", flush=True)
        sig = sigma_table(a, a.flavour, process, grid, a.seed * 1000 + itype)
        for tname in sig:
            print("   " + "  ".join(f"E={e:.3g}: {s:.3e}" for e, s in zip(grid, sig[tname])) + f" cm^2 ({tname})")

        events_f = open(os.path.join(a.out_dir, f"{tag}_events.txt"), "w"); events_f.write(HEADER)
        truth_f = open(os.path.join(a.out_dir, f"{tag}_truth.csv"), "w", newline="")
        truth = csv.writer(truth_f); truth.writerow(TRUTH_COLS)
        chunk = None
        ys, nch, evis_frac, retries_total = [], [], [], 0

        for gidx in range(a.n_per_class):
            enu, p_gen = sample_energy(rng, a)
            tid, tname, _ = TARGETS[rng.choice(len(TARGETS), p=t_prob)]
            n_retry = 0
            while True:
                seed = a.seed * 100000 + 10 * gidx + itype + 1000 * n_retry
                p = make_pythia(a, a.flavour, tid, enu, process, seed)
                if p.next():
                    break
                n_retry += 1
                if n_retry > 50:
                    raise RuntimeError(f"event {gidx}: Pythia failed 50 times at E={enu}")
            retries_total += n_retry
            lep, y, q2, w_had, final, undecayed = analyse_event(p.event, enu, m_target[tid])

            idx = gidx % a.chunk_size if a.chunk_size else gidx
            lines, names, e_vis, n_charged = [], [], 0.0, 0
            for q in final:
                g4 = G4NAME.get(q.id())
                if g4 is None:
                    dropped[q.name()] = dropped.get(q.name(), 0) + 1
                    continue
                ekin = q.e() - q.m()
                if ekin < a.emin:
                    continue
                pabs = math.sqrt(q.px() ** 2 + q.py() ** 2 + q.pz() ** 2)
                dx, dy, dz = q.pz() / pabs, q.px() / pabs, q.py() / pabs      # Pythia +z -> OMSim +x
                lines.append(f"{idx} {g4} {ekin:.6g} {dx:.6f} {dy:.6f} {dz:.6f}\n")
                names.append(g4); e_vis += ekin; n_charged += q.isCharged()
            if not lines:
                lines.append(f"{idx} geantino 0.001 1 0 0\n"); names.append("geantino")

            events_f.writelines(l.replace(f"{idx} ", f"{gidx} ", 1) for l in lines)
            if a.chunk_size:
                k = gidx // a.chunk_size
                if chunk is None or chunk[0] != k:
                    if chunk is not None:
                        chunk[1].close()
                    w = open(os.path.join(a.out_dir, f"{tag}_chunk{k:03d}.txt"), "w"); w.write(HEADER)
                    chunk = (k, w)
                chunk[1].writelines(lines)

            sigma = interp_sigma(sig, tname, grid, enu)
            undec = " ".join(f"{n}:{e:.6g}" for n, e in undecayed)
            truth.writerow([gidx, gidx // a.chunk_size if a.chunk_size else 0, idx, f"pythia{pythia8.__version__}"
                            if hasattr(pythia8, "__version__") else "pythia8", a.seed, gidx, itype,
                            f"{enu:.6g}", f"{y:.5f}", len(names), f"{e_vis:.6g}",
                            f"{sigma / p_gen:.6g}", a.n_per_class, " ".join(names),
                            tname, f"{q2:.6g}", f"{w_had:.6g}", f"{sigma:.6g}", n_retry, undec])
            ys.append(y); nch.append(n_charged); evis_frac.append(e_vis / enu)
            if a.print_undecayed:
                print(f"-- {tag} event {gidx}: E_nu={enu:.6g} GeV on {tname}, y={y:.4f}, Q2={q2:.3g} GeV^2, "
                      f"W={w_had:.3g} GeV, retries={n_retry}")
                for n, e in sorted(undecayed, key=lambda t: -t[1]):
                    print(f"      {n:14s} {e:14.6g} GeV")
                print(f"   -> to Geant4 ({len(names)}): " + " ".join(names))
            elif (gidx + 1) % 500 == 0:
                print(f"   {gidx + 1}/{a.n_per_class}", flush=True)

        events_f.close(); truth_f.close()
        if chunk is not None:
            chunk[1].close()
        print(f"== {tag}: {a.n_per_class} events, <y>={np.mean(ys):.3f}, <n_charged>={np.mean(nch):.2f}, "
              f"<E_vis/E_nu>={np.mean(evis_frac):.3f}, retries={retries_total} -> {a.out_dir}")
    if dropped:
        print("dropped particle types (no Geant4 mapping):", dict(sorted(dropped.items(), key=lambda kv: -kv[1])))
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
