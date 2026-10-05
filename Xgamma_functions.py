"""Longitudinal shower profile of one OMSim event and its Gamma-function fit.

Input: a `<prefix>_hits.root` written with `--photon_tracks F` (tree PhotonTracks:
step points x, y, z [mm], t [ns] of every non-photon track) and its `_args.json`.
The interaction vertex is known from the generator: (-1.7 d, 0, d) m with
d = impact_parameter, neutrino along +x.

1. Principal axis (shower direction)
   The axis passes through the vertex v. With step segments of midpoint r_i and
   length w_i, the direction u maximising the length-weighted second moment along it,

        u = argmax_{|u|=1}  sum_i w_i ((r_i - v) . u)^2  =  argmax  u^T M u,
        M = sum_i w_i (r_i - v)(r_i - v)^T,

   is the eigenvector of M with the largest eigenvalue (uncentred PCA, because the
   axis is pinned to the vertex). The sign is chosen so the shower develops at s > 0.

2. Longitudinal profile
   s = (r - v) . u is the displacement along the axis. Each track projects onto an
   interval [s_lo, s_hi]; the particle count at depth s is the number of tracks whose
   interval contains s:   N(s) = sum_tracks 1[s_lo <= s <= s_hi].

3. Gamma model    g(s; a, b) = s^(a-1) exp(-s/b) / (Gamma(a) b^a)   (mean a b, variance a b^2)
   This is the parametrisation PPC uses for cascade light (a = p1 + p2 ln E, b fixed).

   (a) Maximum likelihood on the depth samples. Every metre of track is one sample of
       s (weights w_i = segment length, s_i = segment midpoint). The log-likelihood
           l(a, b) = sum_i w_i [ (a-1) ln s_i - s_i/b - ln Gamma(a) - a ln b ]
       gives   dl/db = 0  ->  b = <s>/a
       and     dl/da = 0  ->  ln a - psi(a) = ln <s> - <ln s>      (psi = digamma),
       a one-dimensional root that is solved with brentq.

   (b) L1 fit of the binned count profile. Minimise
           L1(a, b, A) = sum_k | N(s_k) - A g(s_k; a, b) |
       over the bin centres s_k. L1 is the maximum-likelihood estimator when the
       residuals are Laplace distributed (median regression), so it is robust to the
       spiky bins produced by a few long tracks. Solved with Nelder-Mead from the MLE.

Usage
    python gamma_functions.py out/vis/nue_cc_chunk000_ev0_hits.root [--all-particles] [--bin 0.1] [--smax 12]
    python gamma_functions.py part_0000_hits.root --all-events        # every event -> part_0000_profile_params.csv

Batch mode (--all-events) fits every event of a multi-event file and writes one CSV row per
event: eventID, n_tracks, track_len_m, ux uy uz (principal axis), a_mle b_mle, a_l1 b_l1,
plus the vertex (vx vy vz) and neutrino direction (nu_dx ..) copied from the Events tree when
the file was made with --random_vertex / --random_rotation (else the args.json vertex).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln

MM = 1e-3          # mm -> m
CHARGED = {"e-", "e+", "mu-", "mu+", "pi+", "pi-", "proton", "anti_proton", "kaon+", "kaon-", "alpha"}


def load_tracks(hits_root: Path, charged_only: bool, event_id: int | None = None):
    """List of [n_steps, 3] arrays [m], one per non-photon track (of one event if given), plus names."""
    t = uproot.open(hits_root)["PhotonTracks"].arrays(["eventID", "particle", "x", "y", "z"], library="np")
    keep = t["particle"] != "opticalphoton"
    if charged_only:
        keep &= np.isin(t["particle"], list(CHARGED))
    if event_id is not None:
        keep &= t["eventID"] == event_id
    tracks = [np.stack([x, y, z], 1) * MM for x, y, z, k in zip(t["x"], t["y"], t["z"], keep) if k and len(x) >= 2]
    return tracks, t["particle"][keep]


def load_event_truth(hits_root: Path):
    """Events tree as a DataFrame indexed by eventID (vx.. nu_dz present for --random_vertex/--random_rotation runs)."""
    import pandas as pd
    ev = uproot.open(hits_root)["Events"]
    df = pd.DataFrame(ev.arrays(library="np")).set_index("eventID")
    return df


def fit_event(tracks, vertex, bin_m=0.1):
    """Principal axis and both Gamma fits for one event; None when there are too few tracks."""
    if len(tracks) < 5:
        return None
    mid, w = segments(tracks)
    u, var_frac = principal_axis(mid, w, vertex)
    s = (mid - vertex) @ u
    if (s > 0).sum() < 5:
        return None
    smax = max(weighted_quantile(s, w, 0.995), 5 * bin_m)
    edges = np.arange(0.0, smax + bin_m, bin_m)
    centres, n = particle_count_profile(tracks, vertex, u, edges)
    a_mle, b_mle = gamma_mle(s, w)
    a_l1, b_l1, A_l1, l1 = gamma_l1_fit(centres, n, a_mle, b_mle)
    return dict(n_tracks=len(tracks), track_len_m=float(w.sum()), ux=u[0], uy=u[1], uz=u[2],
                axis_var_frac=float(var_frac[0]), a_mle=a_mle, b_mle=b_mle, a_l1=a_l1, b_l1=b_l1, l1=l1)


def vertex_from_args(args_json: Path) -> np.ndarray:
    d = float(json.loads(args_json.read_text())["impact_parameter"])
    return np.array([-1.7 * d, 0.0, d])           # OMSimPrimaryGeneratorAction convention [m]


def segments(tracks):
    """Segment midpoints r_i [n, 3] and lengths w_i [n] of all step-to-step segments."""
    mid = np.concatenate([0.5 * (tr[1:] + tr[:-1]) for tr in tracks])
    w = np.concatenate([np.linalg.norm(tr[1:] - tr[:-1], axis=1) for tr in tracks])
    keep = w > 0
    return mid[keep], w[keep]


def principal_axis(mid, w, vertex):
    """Top eigenvector of M = sum_i w_i (r_i - v)(r_i - v)^T, oriented along the shower."""
    r = mid - vertex
    M = (r * w[:, None]).T @ r
    vals, vecs = np.linalg.eigh(M)
    u = vecs[:, np.argmax(vals)]
    if (r @ u * w).sum() < 0:
        u = -u
    return u, vals[np.argsort(vals)[::-1]] / vals.sum()


def particle_count_profile(tracks, vertex, u, edges):
    """N(s_k): number of tracks whose projected interval covers bin centre s_k."""
    centres = 0.5 * (edges[1:] + edges[:-1])
    proj = [(tr - vertex) @ u for tr in tracks]
    lo = np.array([p.min() for p in proj])
    hi = np.array([p.max() for p in proj])
    n = ((lo[:, None] <= centres) & (centres <= hi[:, None])).sum(0)
    return centres, n


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, q * cw[-1])])


def gamma_pdf(s, a, b):
    s = np.clip(s, 1e-9, None)
    return np.exp((a - 1) * np.log(s) - s / b - gammaln(a) - a * np.log(b))


def gamma_mle(s, w):
    """Weighted Gamma MLE: ln a - psi(a) = ln<s> - <ln s>,  b = <s>/a."""
    keep = s > 0
    s, w = s[keep], w[keep]
    mean_s = np.average(s, weights=w)
    mean_ln = np.average(np.log(s), weights=w)
    rhs = np.log(mean_s) - mean_ln                 # > 0 by Jensen
    a = brentq(lambda a: np.log(a) - digamma(a) - rhs, 1e-3, 1e4)
    return a, mean_s / a


def gamma_l1_fit(centres, n, a0, b0):
    """Minimise sum_k |N_k - A g(s_k; a, b)| (Laplace-likelihood / median regression)."""
    def loss(p):
        a, b, A = np.exp(p)                          # positivity through log-parameters
        return np.abs(n - A * gamma_pdf(centres, a, b)).sum()
    A0 = n.sum() * (centres[1] - centres[0])
    res = minimize(loss, np.log([a0, b0, A0]), method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000))
    a, b, A = np.exp(res.x)
    return a, b, A, res.fun


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hits", type=Path, nargs="?", default=Path("out/vis/nue_cc_chunk000_ev0_hits.root"))
    ap.add_argument("--args-json", type=Path, default=None, help="default: <hits>_args.json")
    ap.add_argument("--all-particles", action="store_true",
                    help="include neutral tracks (gammas, neutrons); by default only charged tracks, the "
                         "Cherenkov emitters, are used - far-wandering neutrals otherwise dominate the axis fit")
    ap.add_argument("--bin", type=float, default=0.1, help="depth bin [m]")
    ap.add_argument("--smax", type=float, default=None, help="last depth [m]; default: 99.5% of the track length")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all-events", action="store_true", help="fit every event, write <prefix>_profile_params.csv, no plot")
    ap.add_argument("--event", type=int, default=None, help="event to plot (default: the first one in the file)")
    args = ap.parse_args()
    args_json = args.args_json or Path(str(args.hits).replace("_hits.root", "_args.json"))
    args.charged_only = not args.all_particles

    truth = load_event_truth(args.hits)
    has_vertex = "vx" in truth.columns

    def event_vertex(eid):
        return truth.loc[eid, ["vx", "vy", "vz"]].to_numpy(float) if has_vertex else vertex_from_args(args_json)

    if args.all_events:
        import pandas as pd
        rows = []
        for eid in truth.index:
            tracks, _ = load_tracks(args.hits, args.charged_only, event_id=int(eid))
            fit = fit_event(tracks, event_vertex(eid), args.bin)
            if fit is None:
                continue
            row = dict(eventID=int(eid), E_vis_gev=float(truth.loc[eid, "primaryEnergy"]))
            v = event_vertex(eid); row.update(vx=v[0], vy=v[1], vz=v[2])
            for k in ("nu_dx", "nu_dy", "nu_dz"):
                row[k] = float(truth.loc[eid, k]) if k in truth.columns else np.nan
            row.update(fit); rows.append(row)
        out = args.out or Path(str(args.hits).replace("_hits.root", "_profile_params.csv"))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{args.hits.name}: fitted {len(rows)} of {len(truth)} events -> {out}")
        return

    eid = args.event if args.event is not None else int(truth.index.min())
    tracks, names = load_tracks(args.hits, args.charged_only, event_id=eid)
    vertex = event_vertex(eid)
    mid, w = segments(tracks)
    u, var_frac = principal_axis(mid, w, vertex)
    s = (mid - vertex) @ u
    smax = args.smax or weighted_quantile(s, w, 0.995)
    edges = np.arange(0.0, smax + args.bin, args.bin)
    centres, n = particle_count_profile(tracks, vertex, u, edges)

    a_mle, b_mle = gamma_mle(s, w)
    a_l1, b_l1, A_l1, l1 = gamma_l1_fit(centres, n, a_mle, b_mle)
    A_mle = n.sum() * args.bin                      # same normalisation for the MLE curve
    zen = np.degrees(np.arccos(u[2])); azi = np.degrees(np.arctan2(u[1], u[0]))
    print(f"{args.hits.name}: {len(tracks)} tracks, {len(w)} segments, {w.sum():.1f} m of track")
    print(f"vertex {vertex} m; principal axis u = ({u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}), "
          f"{np.degrees(np.arccos(u[0])):.2f} deg from +x, variance fractions {np.round(var_frac, 3)}")
    print(f"Gamma MLE (track-length weighted depth): a = {a_mle:.3f}, b = {b_mle:.3f} m, mean a b = {a_mle*b_mle:.2f} m")
    print(f"Gamma L1 fit of N(s):                    a = {a_l1:.3f}, b = {b_l1:.3f} m, mean a b = {a_l1*b_l1:.2f} m, L1 = {l1:.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.step(edges[:-1], n, where="post", color="#0b0b0b", lw=1.5, label=f"particles crossing depth s ({len(tracks)} tracks)")
    ss = np.linspace(edges[0] + 1e-3, edges[-1], 400)
    ax.plot(ss, A_mle * gamma_pdf(ss, a_mle, b_mle), color="#2a78d6", lw=2,
            label=f"Gamma MLE: a = {a_mle:.2f}, b = {b_mle:.2f} m")
    ax.plot(ss, A_l1 * gamma_pdf(ss, a_l1, b_l1), color="#eb6834", lw=2, ls="--",
            label=f"Gamma L1 fit: a = {a_l1:.2f}, b = {b_l1:.2f} m")
    ax.set_xlabel("displacement from the vertex along the principal axis s [m]")
    ax.set_ylabel(f"number of {'charged ' if args.charged_only else ''}particle tracks")
    ax.set_title(f"{args.hits.name.replace('_hits.root', '')}: axis {np.degrees(np.arccos(u[0])):.1f} deg from the nu direction",
                 fontsize=10)
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or args.hits.with_name(args.hits.name.replace("_hits.root", "_profile.png"))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
