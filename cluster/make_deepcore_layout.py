#!/usr/bin/env python
"""Write the DeepCore DOM layout for OMSim's `--dom_layout file:<path>` (one module per line, metres).

Two sources, in order of preference:

1. The real geometry from a GCD file (lab store: data/2020/GCD). Dump it once with icetray (on a machine with icetray):

       from icecube import dataio, dataclasses, icetray
       f = dataio.I3File("GeoCalibDetectorStatus_<...>.i3.gz")
       while f.more():
           fr = f.pop_frame()
           if fr.Stop == icetray.I3Frame.Geometry:
               geo = fr["I3Geometry"]; break
       with open("gcd_doms.csv", "w") as o:
           o.write("string,om,x,y,z\\n")
           for k, g in geo.omgeo.items():
               if g.omtype == dataclasses.I3OMGeo.IceCube:
                   o.write(f"{k.string},{k.om},{g.position.x},{g.position.y},{g.position.z}\\n")

   then   python cluster/make_deepcore_layout.py --gcd-csv gcd_doms.csv
   Keeps strings 79-86 (the 8 DeepCore strings), DOMs 11-60 (the 50 DOMs at 7 m spacing in the clear ice below the
   dust layer; the 10 veto-cap DOMs above the dust layer are dropped so a uniform ice model is right), recentres the
   array on the mean (x, y) and mean z and records the offset. `--ic-strings 26,27,35,36,37,45,46` adds the seven
   surrounding standard strings (17 m DOM spacing) that are part of the usual DeepCore fiducial volume.

2. An idealised 8-string DeepCore (`--ideal`, used when no GCD dump is given): six strings on a hexagon of radius
   72 m (72 m nearest-neighbour spacing, as strings 79-84) plus two infill strings at 36 m from the centre (as 85/86),
   50 DOMs per string at 7 m spacing, z in [-171.5, 171.5] m. Nearest-neighbour string distances 36-72 m. Good enough
   for the production and the feasibility question; swap in the GCD layout before anything that is compared to data.

Output: `cluster/deepcore_layout.csv` (default), lines `x y z qe_scale string om`, '#' comments. qe_scale = 1.35 for
the DeepCore strings (Hamamatsu R7081-02MOD high-QE PMT, ~35 % more sensitive than the standard R7081-02 that
`--efficiency_cut` models), 1.0 for standard strings. The file is read by the `file:` branch of the layout parser in
OMSim_WavePID_study.cc (next to the `fill:` branch): moduleID = running index over the non-comment lines, position in
metres, and a per-module QE multiplier applied after the photocathode QE cut (or ignore the column and apply 1.35 as a
hit weight downstream).

The script also prints what the geometry implies for the run: bounding radius -> WORLD, the unit-cell box -> VERTEX,
and the distribution of the distance from a vertex to the nearest DOM.
"""
import argparse
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DC_STRINGS = list(range(79, 87))
DC_OMS = list(range(11, 61))            # 50 DOMs at 7 m spacing, clear ice below the dust layer
HQE = 1.35                              # DeepCore R7081-02MOD relative to the standard R7081-02


def ideal_layout():
    """Idealised 8-string DeepCore: hexagon of radius 72 m + two infill strings at 36 m; 50 DOMs per string, 7 m."""
    strings = []
    for s, ang in enumerate(range(0, 360, 60)):                      # 79-84: hexagon, 72 m spacing
        a = math.radians(ang)
        strings.append((79 + s, 72.0 * math.cos(a), 72.0 * math.sin(a)))
    strings += [(85, 0.0, 36.0), (86, 0.0, -36.0)]                   # 85/86: infill
    z = -171.5 + 7.0 * np.arange(50)
    rows = [(x, y, zz, HQE, s, 11 + k) for s, x, y in strings for k, zz in enumerate(z)]
    return rows, {"source": "ideal", "offset_m": (0.0, 0.0, 0.0)}


def gcd_layout(csv, strings, oms, ic_strings):
    import pandas as pd
    d = pd.read_csv(csv)
    sel = d[d.string.isin(strings) & d.om.isin(oms)].copy()
    sel["qe"] = HQE
    if ic_strings:
        ic = d[d.string.isin(ic_strings)].copy()
        zlo, zhi = sel.z.min(), sel.z.max()
        ic = ic[(ic.z >= zlo - 10) & (ic.z <= zhi + 10)]            # same depth range as the DeepCore DOMs
        ic["qe"] = 1.0
        sel = pd.concat([sel, ic])
    if sel.empty:
        raise SystemExit(f"{csv}: no DOMs for strings {strings} / oms {oms}")
    off = (sel.x.mean(), sel.y.mean(), sel.z.mean())
    rows = [(r.x - off[0], r.y - off[1], r.z - off[2], r.qe, int(r.string), int(r.om)) for r in sel.itertuples()]
    return rows, {"source": str(csv), "offset_m": tuple(float(v) for v in off)}


def describe(rows, vertex_half):
    P = np.array([r[:3] for r in rows])
    rmax = np.linalg.norm(P, axis=1).max()
    xy = np.unique(np.round(P[:, :2], 1), axis=0)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    rng = np.random.default_rng(0)
    V = rng.uniform(-vertex_half, vertex_half, (20000, 3))
    dmin = np.sqrt(((V[:, None, :] - P[None, :, :]) ** 2).sum(2)).min(1)
    print(f"{len(rows)} modules on {len(xy)} strings; z in [{P[:,2].min():.1f}, {P[:,2].max():.1f}] m; "
          f"farthest module {rmax:.0f} m from the origin")
    print(f"nearest-neighbour string spacing: min {nn.min():.0f} m, median {np.median(nn):.0f} m, max {nn.max():.0f} m")
    print(f"vertex box [-{vertex_half}, {vertex_half}]^3 m: distance to the nearest DOM median {np.median(dmin):.1f} m, "
          f"90 % {np.quantile(dmin, 0.9):.1f} m, max {dmin.max():.1f} m")
    print(f"-> run with  WORLD={math.ceil(rmax / 10) * 10 + 20}  VERTEX={vertex_half}  (LAYOUT=file:cluster/deepcore_layout.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcd-csv", type=Path, default=None, help="string,om,x,y,z dump of the GCD (see docstring)")
    ap.add_argument("--ideal", action="store_true", help="idealised 8-string layout (default when no --gcd-csv)")
    ap.add_argument("--strings", default="79-86", help="DeepCore strings to keep from the GCD dump")
    ap.add_argument("--oms", default="11-60", help="DOM numbers to keep (11-60 = clear-ice part, 7 m spacing)")
    ap.add_argument("--ic-strings", default="", help="also add these standard IceCube strings, e.g. 26,27,35,36,37,45,46")
    ap.add_argument("--vertex-half", type=float, default=36.0, help="half-width of the vertex cube to describe [m]")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "deepcore_layout.csv")
    a = ap.parse_args()

    def span(s):
        lo, hi = (int(v) for v in s.split("-")) if "-" in s else (int(s), int(s))
        return list(range(lo, hi + 1))

    if a.gcd_csv is None or a.ideal:
        rows, meta = ideal_layout()
    else:
        ic = [int(v) for v in a.ic_strings.split(",") if v.strip()]
        rows, meta = gcd_layout(a.gcd_csv, span(a.strings), span(a.oms), ic)

    with open(a.out, "w") as o:
        o.write("# DeepCore DOM layout for OMSim --dom_layout file:<this>; columns: x_m y_m z_m qe_scale string om\n")
        o.write(f"# source: {meta['source']}; array recentred, offset (x, y, z) = {meta['offset_m']} m\n")
        for x, y, z, qe, s, om in rows:
            o.write(f"{x:.2f} {y:.2f} {z:.2f} {qe:.2f} {s} {om}\n")
    print(f"wrote {a.out} ({meta['source']})")
    describe(rows, a.vertex_half)
