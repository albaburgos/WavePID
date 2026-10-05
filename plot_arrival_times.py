#!/usr/bin/env python
"""Compare photon arrival-time distributions at a DOM for different primaries.

Reads the OMSim WavePID ROOT output (<tag>_E<E>_d<D>_hits.root) produced by
run_wavepid.sh (single primaries) or merged from the cluster with
cluster/merge_parts.py (GENIE final states, E is then a range like E2-50) and
overlays, per tag, the time-residual PDF, the normalised CDF, and a per-tag
breakdown by photon origin. Runs with the same E and d are plotted together, so
the E10_d5 single-particle set and the E2-50_d5 GENIE set give separate figures.

  em     e-  -> electromagnetic cascade  (proxy for the EM part of a CC nu_e event)
  pim    pi- -> hadronic cascade         (proxy for an NC event / hadronic vertex)
  mum    mu- -> minimum-ionising track   (proxy for a CC nu_mu event, OMSim reference)
  nue_cc     -> GENIE nu_e CC final state (e- + hadrons), all from the vertex
  nue_nc     -> GENIE nu_e NC final state (hadrons only)

Time residual follows the OMSim notebook: dt = t_hit - (t_c + t_0), where t_c is
the direct-Cherenkov photon travel time from the track to the DOM surface and
t_0 the primary's travel time to the emission point. dt = 0 is a direct, unscattered
Cherenkov photon from a primary moving along the beam axis (+x); delays come from
the cascade's longitudinal development, scattering, and late secondaries. GENIE
final states are not collinear with the beam, so a hadron or electron heading
towards the DOM can give dt down to about -7 ns (it travels at c, the photon at
c/n_group); --tmin covers that.

usage: python plot_arrival_times.py [--out out] [--tmin -2] [--tmax 40] [--bin 0.5]
                                    [--tags nue_cc nue_nc]
"""
import argparse, glob, json, os, re
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (never cycled): slot1 blue, slot2 orange, slot3 aqua.
# CC-like tags share the blue, NC-like the orange; sets with different E/d are
# plotted in separate figures so the colours never collide within one panel.
PRIMARIES = {                         # tag: (label, colour)
    "em":     (r"$e^-$ EM cascade (CC $\nu_e$-like)",          "#2a78d6"),
    "pim":    (r"$\pi^-$ hadronic cascade (NC-like)",           "#eb6834"),
    "mum":    (r"$\mu^-$ track (CC $\nu_\mu$-like)",            "#1baf7a"),
    "nue_cc": (r"GENIE $\nu_e$ CC final state",                 "#2a78d6"),
    "nue_nc": (r"GENIE $\nu_e$ NC final state",                 "#eb6834"),
}
ORIGIN_ORDER = ["Cerenkov from Muon", "Cerenkov from Electron", "Cerenkov from Other",
                "Bremsstrahlung", "Scintillation", "Other"]
ORIGIN_COLORS = ["#1baf7a", "#2a78d6", "#eb6834", "#eda100", "#e87ba4", "#9a9a92"]

C_VAC, N_GROUP, THETA_C, DOM_R = 299.792458, 1.32, np.radians(42.0), 0.1651  # mm/ns, -, rad, m

# <tag>_E<E>_d<d>_hits.root; E is a number (E10) or a range (E2-50) for GENIE sets
FNAME = re.compile(r"(\w+?)_E([\d.]+(?:-[\d.]+)?)_d([\d.]+)_hits\.root")


def t_reference(d):
    """Arrival time (ns) of a direct Cherenkov photon from a primary that starts at
    (-1.7 d, 0, d) and moves along +x at c (OMSimPrimaryGeneratorAction): t_0 to
    reach the emission point x_e = -d / tan(theta_c), plus t_c for the photon to
    cover d / sin(theta_c) minus the DOM radius at the group velocity."""
    x_e = -d / np.tan(THETA_C)
    t0 = (1.7 * d + x_e) * 1e3 / C_VAC
    tc = (d / np.sin(THETA_C) - DOM_R) * 1e3 * N_GROUP / C_VAC
    return t0 + tc

def energy_key(E):
    """Sort key / numeric value of an energy label: '10' -> 10.0, '2-50' -> 2.0."""
    return float(E.split("-")[0])

def load(path):
    m = FNAME.match(os.path.basename(path))
    if m is None:
        return None          # smoke tests / other files
    tag, E, d = m.group(1), m.group(2), float(m.group(3))
    f = uproot.open(path)
    a = f["PhotonHits"].arrays(["eventID", "hitTime", "photonOrigin"], library="np")
    # number of simulated events: the args json OMSim / merge_parts.py write next to
    # the output, else the Events tree (one row per event, present in newer files),
    # else the largest eventID
    n_ev = None
    for cand in glob.glob(path.replace("_hits.root", "*.json")):
        try:
            n_ev = int(json.load(open(cand)).get("numevents"))
        except Exception:
            pass
    if n_ev is None and "Events" in f:
        n_ev = int(f["Events"].num_entries)
    if n_ev is None:
        n_ev = int(a["eventID"].max()) + 1 if len(a["eventID"]) else 0
    return dict(tag=tag, E=E, d=d, n_ev=n_ev, ev=a["eventID"],
                dt=a["hitTime"] - t_reference(d), origin=a["photonOrigin"].astype(str))

def per_event_quantiles(dt, ev, q):
    """Median over events of the per-event quantiles q of dt (one sort, no per-event masks)."""
    if len(dt) == 0:
        return np.full(len(q), np.nan)
    order = np.argsort(ev, kind="stable")
    ev_s, dt_s = ev[order], dt[order]
    bounds = np.flatnonzero(np.diff(ev_s)) + 1
    out = [np.quantile(chunk, q) for chunk in np.split(dt_s, bounds)]
    return np.median(out, axis=0)

def style(ax):
    ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#5d5d57", labelsize=9)

def energy_label(E):
    """'10' -> '10', '2-50' -> '2$-$50' (mathtext minus so it reads as a range)."""
    return E.replace("-", r"$-$")

def plot_group(runs, args):
    """Summary table + the two figures for one set of runs sharing E and d."""
    E, d = runs[0]["E"], runs[0]["d"]
    # lower edge: -2 ns for beam-collinear primaries, extended (whole bins) to the
    # earliest hit for GENIE final states unless --tmin fixes it
    tmin = args.tmin
    if tmin is None:
        lo = min(r["dt"].min() for r in runs if len(r["dt"]))
        tmin = min(-2.0, np.floor(lo / args.bin) * args.bin)
    edges = np.arange(tmin, args.tmax + args.bin, args.bin)
    title_Ed = rf"$E = {energy_label(E)}$ GeV $\cdot$ impact parameter $d = {d:g}$ m"

    # ---- summary table -------------------------------------------------------
    print(f"\n{'tag':<8}{'events':>7}{'hits/ev':>9}{'median dt':>11}{'t10':>7}{'t50':>7}{'t90':>7}"
          f"{'f(<5ns)':>9}{'f(<tmin)':>10}   E={E} GeV d={d:g} m")
    for r in runs:
        q = per_event_quantiles(r["dt"], r["ev"], [0.1, 0.5, 0.9])
        print(f"{r['tag']:<8}{r['n_ev']:>7}{len(r['dt'])/max(r['n_ev'],1):>9.1f}"
              f"{np.median(r['dt']):>11.2f}{q[0]:>7.2f}{q[1]:>7.2f}{q[2]:>7.2f}"
              f"{np.mean(r['dt'] < 5):>9.3f}{np.mean(r['dt'] < tmin):>10.4f}")

    # ---- Fig 1: residual-time PDF (log) and normalised CDF ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for r in runs:
        label, col = PRIMARIES[r["tag"]]
        c, _ = np.histogram(r["dt"], bins=edges)
        pdf = c / max(c.sum() * args.bin, 1)
        axes[0].stairs(pdf, edges, color=col, lw=2, label=label)
        cdf = np.searchsorted(np.sort(r["dt"]), edges, side="right") / max(len(r["dt"]), 1)
        axes[1].plot(edges, cdf, color=col, lw=2, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$p(\Delta t_\gamma)\;[\mathrm{ns}^{-1}]$")
    axes[0].set_title("Residual-time PDF", fontsize=10, loc="left")
    axes[1].set_ylabel(r"$P(\Delta t_\gamma' \leq \Delta t_\gamma)$")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Normalised cumulative distribution", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("Residual Time [ns]"); ax.set_xlim(edges[0], edges[-1]); style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(rf"Photon arrival times at a standard DOM $\cdot$ {title_Ed} $\cdot$ SPICE ice",
                 fontsize=11, x=0.01, ha="left")
    f1 = os.path.join(args.out, f"arrival_times_E{E}_d{d:g}.png"); fig.savefig(f1, dpi=160)
    plt.close(fig)

    # ---- Fig 2: per-tag stacked breakdown by photon origin --------------------
    fig, axes = plt.subplots(1, len(runs), figsize=(4.3 * len(runs), 3.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        stacks, names, cols = [], [], []
        for name, col in zip(ORIGIN_ORDER, ORIGIN_COLORS):
            sel = r["origin"] == name
            if sel.any():
                stacks.append(np.histogram(r["dt"][sel], bins=edges)[0] / max(r["n_ev"], 1) / args.bin)
                names.append(f"{name}  ({sel.mean()*100:.0f}%)"); cols.append(col)
        if stacks:
            ax.stackplot(edges[:-1], *stacks, step="post", colors=cols, labels=names, lw=0)
        ax.set_title(PRIMARIES[r["tag"]][0], fontsize=10, loc="left")
        ax.set_xlabel("Residual Time [ns]"); ax.set_xlim(edges[0], edges[-1]); style(ax)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("photons per event per ns")
    fig.suptitle(rf"Photon origin vs arrival time $\cdot$ {title_Ed}", fontsize=11, x=0.01, ha="left")
    f2 = os.path.join(args.out, f"origin_breakdown_E{E}_d{d:g}.png"); fig.savefig(f2, dpi=160)
    plt.close(fig)
    print("wrote", f1, "\nwrote", f2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--tmin", type=float, default=None,
                    help="lower edge of the residual-time axis [ns]; default -2, or lower if a set has earlier hits")
    ap.add_argument("--tmax", type=float, default=40.0)
    ap.add_argument("--bin", type=float, default=1.0)
    ap.add_argument("--tags", nargs="+", choices=list(PRIMARIES), default=None,
                    help="only these tags (default: every known tag found in --out)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.out, "*_hits.root")))
    runs = [r for r in (load(p) for p in paths) if r is not None and r["tag"] in PRIMARIES]
    if args.tags:
        runs = [r for r in runs if r["tag"] in args.tags]
    if not runs:
        raise SystemExit(f"no *_hits.root with known tags in {args.out}")

    # one figure set per (E, d): runs of different energies are not comparable
    groups = {}
    for r in runs:
        groups.setdefault((r["E"], r["d"]), []).append(r)
    plt.rcParams["mathtext.fontset"] = "cm"
    for key in sorted(groups, key=lambda k: (k[1], energy_key(k[0]))):
        grp = sorted(groups[key], key=lambda r: list(PRIMARIES).index(r["tag"]))
        plot_group(grp, args)

if __name__ == "__main__":
    main()
