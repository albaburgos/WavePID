#!/usr/bin/env python
"""Draw the Geant4 light propagation of one OMSim WavePID event.

Reads <prefix>_hits.root written with --photon_tracks (tree PhotonTracks: step points
of every charged track, every detected Cherenkov photon and a sampled fraction of the
other photons; mm and ns) and draws

  <prefix>_paths.png      side view (x-z) and top view (x-y) of the cascade leaving the
                          vertex at (-1.7 d, 0, d), the sampled photon paths coloured
                          by emission time, and the detected photons' paths to the DOM
  <prefix>_snapshots.png  the same scene at fixed times: where every recorded photon is
                          at t, the cascade drawn up to t, hits accumulated on the DOM
  <prefix>_propagation.gif  (--gif) the snapshots as an animation

The frame is OMSim's: DOM centre at the origin, neutrino along +x. Times are Geant4
global times (t = 0 at the vertex).

usage: python plot_light_propagation.py out/vis/nue_nc_chunk000_ev0 [--event-file F]
                                        [--truth nue_nc_truth.csv --chunk 0 --event 0]
                                        [--times 5 15 25 35 45 60] [--gif]
"""
import argparse, csv, json, os
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb
from matplotlib.patches import Circle

DOM_R = 0.1651                                   # m
# cascade particles are drawn by category: (label, colour, linestyle, linewidth)
CATEGORIES = {
    "e":   (r"$e^\pm$",                    "#2a78d6", "-",  0.9),
    "mu":  (r"$\mu^\pm$",                  "#1baf7a", "-",  1.4),
    "gam": (r"$\gamma$",                   "#9a9a92", "--", 0.5),
    "n":   ("neutrons",                    "#c3c2b7", ":",  0.7),
    "had": (r"charged hadrons, nuclei",    "#eb6834", "-",  1.1),
}
INVISIBLE = ("nu_", "anti_nu", "geantino")       # no light, not drawn

def category(name):
    if name in ("e-", "e+"): return "e"
    if name in ("mu-", "mu+"): return "mu"
    if name == "gamma": return "gam"
    if name in ("neutron", "anti_neutron"): return "n"
    return "had"

def is_charged(name):
    """Light-emitting (charged) cascade particles, used for the view box."""
    return name[-1] in "+-" or name in ("proton", "anti_proton", "deuteron", "triton", "alpha", "He3") or name[0].isupper()


def time_cmap(n=256):
    """Steamshovel's I3TimeColorMap (copied from ~/IceCube/shovel.py): red early, violet late."""
    v = np.linspace(0.0, 1.0, n)
    return ListedColormap(hsv_to_rgb(np.stack([0.7 * v, np.ones_like(v), np.ones_like(v)], 1)), name="I3TimeColorMap")

def style(ax):
    ax.grid(color="#e6e5df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#5d5d57", labelsize=8.5)
    ax.set_aspect("equal")

class Event:
    """One event's tracks, in metres and nanoseconds, split into cascade and photons."""

    def __init__(self, prefix):
        self.prefix = prefix
        f = uproot.open(prefix + "_hits.root")
        if "PhotonTracks" not in f:
            raise SystemExit(f"{prefix}_hits.root has no PhotonTracks tree; rerun OMSim with --photon_tracks")
        tr = f["PhotonTracks"].arrays()
        self.hits = f["PhotonHits"].arrays(["hitTime", "globalPos_x", "globalPos_y", "globalPos_z",
                                            "deltaPos_x", "deltaPos_y", "deltaPos_z", "photonOrigin"], library="np")
        self.args = {}
        if os.path.exists(prefix + "_args.json"):
            self.args = json.load(open(prefix + "_args.json"))
        self.d = float(self.args.get("impact_parameter", 5.0))
        self.fraction = float(self.args.get("photon_tracks", 0.0))
        self.vertex = np.array([-1.7 * self.d, 0.0, self.d])

        names = ak.to_numpy(tr.particle).astype(str)
        is_photon = names == "opticalphoton"
        invisible = np.array([n.startswith(INVISIBLE) for n in names])
        ph, ch = tr[is_photon], tr[~is_photon & ~invisible]
        self.ph_detected = ak.to_numpy(ph.detected == 1)
        self.ph = self._flatten(ph)
        self.ch = self._flatten(ch)
        self.ch_names = names[~is_photon & ~invisible]
        self.ch_cat = np.array([category(n) for n in self.ch_names])
        self.ch_energy = ak.to_numpy(ch.energy) / 1e3          # MeV -> GeV
        self.ch_parent = ak.to_numpy(ch.parentID)
        self.ph_t0 = self.ph["t"][self.ph["start"]]              # emission time per photon
        self.ph_t1 = self.ph["t"][self.ph["stop"] - 1]           # end time per photon
        self.n_primaries = int(np.sum(self.ch_parent == 0))

    @staticmethod
    def _flatten(tracks):
        """Flat point arrays + per-track [start, stop) offsets (fast vectorised slicing)."""
        n = ak.to_numpy(ak.num(tracks.x))
        stop = np.cumsum(n); start = stop - n
        return dict(x=ak.to_numpy(ak.flatten(tracks.x)) / 1e3, y=ak.to_numpy(ak.flatten(tracks.y)) / 1e3,
                    z=ak.to_numpy(ak.flatten(tracks.z)) / 1e3, t=ak.to_numpy(ak.flatten(tracks.t)),
                    start=start, stop=stop, n=n)

    # --- geometry helpers ---------------------------------------------------------
    @staticmethod
    def polylines(flat, sel, a, b):
        """List of (n_i, 2) arrays of coordinates a, b for the selected tracks."""
        return [np.column_stack([flat[a][s:e], flat[b][s:e]])
                for s, e in zip(flat["start"][sel], flat["stop"][sel])]

    @staticmethod
    def segments(flat, sel, a, b, tmax=None):
        """Segments of the selected tracks for a LineCollection.

        Returns (seg (N,2,2), track index per segment, time at the segment start).
        With tmax only the part of each track travelled before tmax is kept, the last
        segment being cut by interpolation."""
        seg, idx, tseg = [], [], []
        for i in np.flatnonzero(sel):
            s, e = flat["start"][i], flat["stop"][i]
            pa, pb, tt = flat[a][s:e], flat[b][s:e], flat["t"][s:e]
            if tmax is not None:
                k = np.searchsorted(tt, tmax, side="right")   # points with t <= tmax
                if k < 2:
                    continue
                if k < len(tt):                              # interpolate the running end
                    w = (tmax - tt[k - 1]) / max(tt[k] - tt[k - 1], 1e-9)
                    pa = np.append(pa[:k], pa[k - 1] + w * (pa[k] - pa[k - 1]))
                    pb = np.append(pb[:k], pb[k - 1] + w * (pb[k] - pb[k - 1]))
                    tt = np.append(tt[:k], tmax)
                else:
                    pa, pb, tt = pa[:k], pb[:k], tt[:k]
            pts = np.column_stack([pa, pb])
            seg.append(np.stack([pts[:-1], pts[1:]], 1)); idx.append(np.full(len(pts) - 1, i)); tseg.append(tt[:-1])
        if not seg:
            return np.zeros((0, 2, 2)), np.zeros(0, int), np.zeros(0)
        return np.concatenate(seg), np.concatenate(idx), np.concatenate(tseg)

    def positions_at(self, flat, t):
        """Position of every track at time t (NaN where the track has not started or has ended)."""
        # count the points with time <= t inside each [start, stop) with a cumulative sum
        tt = flat["t"]
        cum = np.concatenate([[0], np.cumsum(tt <= t)])
        k = cum[flat["stop"]] - cum[flat["start"]]
        n = flat["n"]
        alive = (k >= 1) & (k < n)
        i0 = flat["start"] + np.clip(k - 1, 0, None); i1 = np.minimum(i0 + 1, flat["stop"] - 1)
        t0, t1 = tt[i0], tt[i1]
        w = np.where(t1 > t0, (t - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
        pos = np.full((len(n), 3), np.nan)
        for j, a in enumerate("xyz"):
            pos[:, j] = np.where(alive, flat[a][i0] + w * (flat[a][i1] - flat[a][i0]), np.nan)
        return pos, alive

    def bounds(self):
        """Square view box (m) holding the charged cascade, the vertex and the DOM, clipped to the world."""
        charged = np.array([is_charged(n) for n in self.ch_names], bool)
        pts = [self.vertex, [0, 0, 0]]
        for s, e in zip(self.ch["start"][charged], self.ch["stop"][charged]):
            pts.append(np.column_stack([self.ch["x"][s:e], self.ch["y"][s:e], self.ch["z"][s:e]]))
        pts = np.vstack([np.atleast_2d(p) for p in pts])
        lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
        span = np.max(hi - lo)                         # same scale on every axis
        cen = (lo + hi) / 2
        lo, hi = cen - span / 2, cen + span / 2
        wr = float(self.args.get("world_radius", 30.0))
        return np.clip(lo, -wr, wr), np.clip(hi, -wr, wr)


def read_primaries(path):
    """Particles of the (single) event in an OMSim --event_file: [(name, E_GeV, dir)]."""
    out = []
    for line in open(path):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        out.append((p[1], float(p[2]), np.array(p[3:6], float)))
    return out

def read_truth(path, chunk, index):
    for row in csv.DictReader(open(path)):
        if int(row["chunk"]) == chunk and int(row["index"]) == index:
            return row
    return None


def draw_scene(ax, ev, view, tmax=None, norm=None, cmap=None, photon_alpha=0.12, lo=None, hi=None):
    """One projection ('xz' or 'xy').

    Without tmax: complete paths, every photon segment coloured by the time the light
    passes there (the propagation itself). With tmax: snapshot at that time, photon
    positions coloured by emission time, cascade drawn up to tmax, hits accumulated."""
    a, b = view[0], view[1]
    ia, ib = "xyz".index(a), "xyz".index(b)
    if tmax is None:
        seg, idx, ts = ev.segments(ev.ph, ~ev.ph_detected, a, b)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=cmap(norm(ts)), lw=0.35, alpha=photon_alpha, zorder=1))
        seg, idx, ts = ev.segments(ev.ph, ev.ph_detected, a, b)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=cmap(norm(ts)), lw=1.0, alpha=0.95, zorder=3))
    else:
        pos, alive = ev.positions_at(ev.ph, tmax)
        live = alive & ~ev.ph_detected
        ax.scatter(pos[live, ia], pos[live, ib], s=1.2, c=cmap(norm(ev.ph_t0[live])), lw=0, alpha=0.6, zorder=2)
        live = alive & ev.ph_detected
        ax.scatter(pos[live, ia], pos[live, ib], s=9, c=cmap(norm(ev.ph_t0[live])), edgecolors="black", lw=0.3, zorder=4)
        # hits already on the DOM
        done = ev.ph_detected & (ev.ph_t1 <= tmax)
        if done.any():
            hx = ev.ph[a][ev.ph["stop"][done] - 1]; hy = ev.ph[b][ev.ph["stop"][done] - 1]
            ax.scatter(hx, hy, s=14, c=cmap(norm(ev.ph_t0[done])), marker="o", edgecolors="black", lw=0.4, zorder=6)
    # cascade tracks, by category
    for cat, (_, col, ls, lw) in CATEGORIES.items():
        seg, idx, ts = ev.segments(ev.ch, ev.ch_cat == cat, a, b, tmax=tmax)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=col, lw=lw, ls=ls, alpha=0.9, zorder=5))
    # DOM and vertex
    ax.add_patch(Circle((0, 0), DOM_R, facecolor="white", edgecolor="black", lw=1.0, zorder=7))
    ax.plot(ev.vertex[ia], ev.vertex[ib], marker="*", ms=9, color="black", zorder=8, lw=0)
    ax.set_xlim(lo[ia], hi[ia]); ax.set_ylim(lo[ib], hi[ib])
    ax.set_xlabel(f"{a} [m]"); ax.set_ylabel(f"{b} [m]")
    style(ax)

def legend_handles(ev):
    present = [c for c in CATEGORIES if np.any(ev.ch_cat == c)]
    h = [plt.Line2D([], [], color=CATEGORIES[c][1], ls=CATEGORIES[c][2], lw=1.6, label=CATEGORIES[c][0]) for c in present]
    h += [plt.Line2D([], [], color="black", marker="*", lw=0, ms=9, label="vertex"),
          plt.Line2D([], [], color="black", marker="o", mfc="white", lw=0, ms=8, label="DOM")]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="output prefix, e.g. out/vis/nue_nc_chunk000_ev0")
    ap.add_argument("--event-file", default=None, help="the one-event OMSim input (primaries for the caption)")
    ap.add_argument("--truth", default=None, help="<class>_truth.csv from extract_genie_events.py")
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--event", type=int, default=0)
    ap.add_argument("--times", type=float, nargs="+", default=[5, 15, 25, 35, 45, 60], help="snapshot times [ns]")
    ap.add_argument("--tmax-colour", type=float, default=None, help="upper end of the time colour scale [ns]")
    ap.add_argument("--gif", action="store_true", help="also write an animation (needs Pillow)")
    ap.add_argument("--frames", type=int, default=64); ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    ev = Event(args.prefix)
    lo, hi = ev.bounds()
    t_hit = ev.hits["hitTime"]
    tmax_c = args.tmax_colour or (np.percentile(t_hit, 95) + 5 if len(t_hit) else 60.0)
    cmap = time_cmap()
    norm = Normalize(0, tmax_c)                                   # paths: time along the photon path
    # snapshots: emission time (cascade development). Late light from neutron captures
    # (microseconds) would stretch the scale, so it is set on the prompt photons only.
    prompt = ev.ph_t0[ev.ph_t0 < 100.0]
    norm_em = Normalize(0, max(np.percentile(prompt, 98) if len(prompt) else 1.0, 1.0))

    # ---- caption --------------------------------------------------------------
    primaries = read_primaries(args.event_file) if args.event_file else []
    e_vis = sum(e for _, e, _ in primaries)
    lines = [f"vertex (-1.7 d, 0, d), d = {ev.d:g} m; DOM at the origin; neutrino along +x"]
    if primaries:
        lines.append("primaries: " + ", ".join(f"{n} {e:.3g} GeV" for n, e, _ in primaries) + f"  (E_vis = {e_vis:.3g} GeV)")
    truth = read_truth(args.truth, args.chunk, args.event) if args.truth else None
    if truth:
        lines.append(f"GENIE: E_nu = {float(truth['e_nu_gev']):.3g} GeV, y = {float(truth['y']):.3f}, "
                     f"{'NC' if truth['interaction'] == '2' else 'CC'}, file {truth['file']} run {truth['run']} event {truth['event']}")
    n_ph = len(ev.ph["n"]); n_det = int(ev.ph_detected.sum()); n_samp = n_ph - n_det
    est_total = n_samp / ev.fraction if ev.fraction > 0 else float("nan")
    lines.append(f"{len(ev.ch['n'])} cascade tracks, {n_det} photons detected, {n_samp} of ~{est_total:,.0f} "
                 f"other photons drawn (fraction {ev.fraction:g})")
    caption = "\n".join(lines)

    # ---- summary ----------------------------------------------------------------
    print(caption)
    if len(t_hit):
        print(f"hit times: first {t_hit.min():.1f} ns, median {np.median(t_hit):.1f} ns, last {t_hit.max():.1f} ns")
        emis = np.column_stack([ev.hits["globalPos_x"] + ev.hits["deltaPos_x"], ev.hits["globalPos_y"] + ev.hits["deltaPos_y"],
                                ev.hits["globalPos_z"] + ev.hits["deltaPos_z"]]) / 1e3
        print(f"emission points of detected photons: x {emis[:, 0].min():.2f}..{emis[:, 0].max():.2f} m, "
              f"median distance from vertex {np.median(np.linalg.norm(emis - ev.vertex, axis=1)):.2f} m")
    u, c = np.unique(ev.ch_names, return_counts=True)
    print("cascade tracks:", ", ".join(f"{n} x{k}" for n, k in zip(u, c)), "(neutrinos not drawn)")
    late = np.mean(ev.ph_t0 > 100.0)
    print(f"photon emission: 98% of prompt light within {norm_em.vmax:.1f} ns; {late*100:.2f}% emitted after 100 ns "
          f"(neutron-induced gammas), latest {ev.ph_t0.max():.0f} ns")

    plt.rcParams["mathtext.fontset"] = "cm"
    # ---- Fig 1: complete paths --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    draw_scene(axes[0], ev, "xz", norm=norm, cmap=cmap, lo=lo, hi=hi)
    draw_scene(axes[1], ev, "xy", norm=norm, cmap=cmap, lo=lo, hi=hi)
    axes[0].set_title("side view (x-z)", fontsize=10, loc="left"); axes[1].set_title("top view (x-y)", fontsize=10, loc="left")
    axes[1].legend(handles=legend_handles(ev), frameon=False, fontsize=8, loc="upper right")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("time along the photon path [ns]", fontsize=9)
    fig.suptitle("Geant4 light propagation, OMSim WavePID, standard DOM, SPICE ice", fontsize=11.5, x=0.01, ha="left")
    fig.text(0.01, -0.01, caption, fontsize=8, va="top", color="#5d5d57")
    f1 = args.prefix + "_paths.png"; fig.savefig(f1, dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Fig 2: snapshots -------------------------------------------------------
    times = args.times
    ncol = 3; nrow = int(np.ceil(len(times) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), constrained_layout=True, squeeze=False)
    for ax, t in zip(axes.flat, times):
        draw_scene(ax, ev, "xz", tmax=t, norm=norm_em, cmap=cmap, lo=lo, hi=hi)
        n_alive = int(ev.positions_at(ev.ph, t)[1].sum()); n_hit = int(np.sum(t_hit <= t))
        ax.set_title(f"t = {t:g} ns   {n_alive} photons drawn, {n_hit} hits", fontsize=9.5, loc="left")
    for ax in axes.flat[len(times):]:
        ax.axis("off")
    axes.flat[0].legend(handles=legend_handles(ev), frameon=False, fontsize=7.5, loc="lower left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_em, cmap=cmap), ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("photon emission time [ns]", fontsize=9)
    fig.suptitle("Light propagation snapshots (side view), photons coloured by emission time", fontsize=11.5, x=0.01, ha="left")
    f2 = args.prefix + "_snapshots.png"; fig.savefig(f2, dpi=160); plt.close(fig)
    print("wrote", f1, "\nwrote", f2)

    # ---- optional animation -----------------------------------------------------
    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        t_end = max(times[-1], (t_hit.max() + 10) if len(t_hit) else 60)
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        def frame(i):
            ax.cla()
            t = t_end * i / (args.frames - 1)
            draw_scene(ax, ev, "xz", tmax=t, norm=norm_em, cmap=cmap, lo=lo, hi=hi)
            ax.set_title(f"t = {t:5.1f} ns   hits {int(np.sum(t_hit <= t))}", fontsize=10, loc="left")
        anim = FuncAnimation(fig, frame, frames=args.frames)
        f3 = args.prefix + "_propagation.gif"; anim.save(f3, writer=PillowWriter(fps=args.fps)); plt.close(fig)
        print("wrote", f3)

if __name__ == "__main__":
    main()
