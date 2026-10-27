#!/usr/bin/env python
"""How does the Pythia 8 DIS setup of gen_pythia_events.py behave at 5-300 GeV?

For each E_nu and class it builds Pythia with exactly `pythia_settings` from gen_pythia_events.py (Q^2 > 1 GeV^2,
mHat > 1 GeV, ...), generates N events on a proton and on a neutron, and reports
  sigma / E                vs the high-energy DIS asymptote (nu isoscalar CC 0.677e-38 cm^2/GeV, NC = 0.31 x that),
  failure rate             of p.next() (phase-space problems: the generator retries these),
  <y>, <W>, <n_charged>    of the hadronic final state.
Where sigma/E sits well below the asymptote the DIS-only model is missing most of the cross-section
(QE + RES + shallow-inelastic), and the final states it does produce are string fragments of a W ~ 2 GeV system,
not the single-nucleon / Delta states GENIE gives there. Run in the omsim env (needs pythia8):
    python pythia_lowE_check.py --energies 5 10 20 50 100 300 --n 300
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythia8  # noqa: E402
from gen_pythia_events import MB_TO_CM2, TARGETS, analyse_event, make_pythia  # noqa: E402

SIGMA_E_ASYMPT = {"cc": 0.677e-38, "nc": 0.31 * 0.677e-38}     # cm^2 / GeV, nu on an isoscalar target
PROC = {"cc": "WeakBosonExchange:ff2ff(t:W)", "nc": "WeakBosonExchange:ff2ff(t:gmZ)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energies", nargs="+", type=float, default=[5, 10, 20, 50, 100, 300])
    ap.add_argument("--n", type=int, default=300, help="events per (E, class, target)")
    ap.add_argument("--flavour", type=int, default=12)
    ap.add_argument("--q2min", type=float, default=1.0)
    ap.add_argument("--mhatmin", type=float, default=1.0)
    ap.add_argument("--tau0max-mm", type=float, default=10.0)
    a = ap.parse_args()
    pdata = pythia8.Pythia("", False).particleData
    m_t = {tid: pdata.m0(tid) for tid, _, _ in TARGETS}
    t_w = {tid: w for tid, _, w in TARGETS}
    print(f"{'E [GeV]':>8} {'cls':>3} {'sigma/E water':>14} {'/asympt':>8} {'fail%':>6} {'<y>':>6} {'<W> GeV':>8} {'<n_ch>':>7}  (time)")
    rows = []
    for e in a.energies:
        for cls in ("cc", "nc"):
            t0 = time.time()
            sig_w, ys, ws, nch, nfail, ntry = 0.0, [], [], [], 0, 0
            for tid, tname, w in TARGETS:
                p = make_pythia(a, a.flavour, tid, e, PROC[cls], 12345 + int(e) + tid)
                got = 0
                while got < a.n and ntry < 20 * a.n:
                    ntry += 1
                    if not p.next():
                        nfail += 1; continue
                    got += 1
                    lep, y, q2, w_had, final, _ = analyse_event(p.event, e, m_t[tid])
                    ys.append(y); ws.append(w_had); nch.append(sum(q.isCharged() for q in final))
                sig_w += p.infoPython().sigmaGen() * MB_TO_CM2 * t_w[tid]
            sig_w /= sum(t_w.values())
            ratio = sig_w / e / SIGMA_E_ASYMPT[cls]
            rows.append((e, cls, sig_w / e, ratio, nfail / max(ntry, 1), np.mean(ys), np.mean(ws), np.mean(nch)))
            print(f"{e:8g} {cls:>3} {sig_w / e:14.3e} {ratio:8.2f} {100 * nfail / max(ntry, 1):6.1f} {np.mean(ys):6.3f} "
                  f"{np.mean(ws):8.2f} {np.mean(nch):7.2f}  ({time.time() - t0:.0f} s)", flush=True)
    print("\nsigma/E in cm^2/GeV, water-averaged (p:n = 10:8); '/asympt' = ratio to the high-energy DIS value.")
    print("Pythia is usable where /asympt is ~0.8-1 and fail% is small; below that GENIE (QE+RES+DIS) is needed.")


if __name__ == "__main__":
    main()
