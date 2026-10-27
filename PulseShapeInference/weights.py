"""Per-event weights for the NC/CC studies: matching in visible energy, flux weights, and the class-prior shift.

Two weightings, used at two different stages:

  matching weights   remove visible energy (or the total charge Q_tot) as a trivial NC/CC discriminant for *training*
                     and for energy-blind metrics.  Per class c, w_c(x) = p_target(x) / p_c(x) in bins of x = log10 E_vis
                     or log10 Q_tot.  Targets: the harmonic mean of the two class densities, 2 p_CC p_NC / (p_CC + p_NC)
                     (w_CC = 2 p_NC / (p_CC + p_NC) <= 2 and w_NC <= 2, bounded, zero only where a class is absent, no
                     event discarded; classify_gamma.py used to subsample both classes to the min count per bin instead);
                     or one class as the reference ("nc": CC reweighted onto the NC histogram, w_CC = N_NC / N_CC per
                     bin and NC kept at 1, flux_weighting.ipynb section 9; unbounded, so the ratio is clipped).
                     Flux weights cannot do this: they act on E_nu, and the class difference in E_vis is the convolution
                     with kappa = E_dep/E_nu (~1 for CC, F*y for NC), which survives any flux.
  flux weights       the physical class mix for *evaluation*: expected events per year for a flux, from
                     pythia_nue_1tev/weights_1tev.parquet (flux_weighting.ipynb), joined on (cls, part, eventID).
                     The energy-matched posterior is turned into the physical one by the prior shift
                         P(NC | x, E) = sigmoid( logit P_matched(NC | x) + ln[pi_NC(E) / pi_CC(E)] ),
                     pi_c(E) being the flux-weighted class spectra in E_vis.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WEIGHT_TABLE = ROOT / "pythia_nue_1tev" / "weights_1tev.parquet"     # written by flux_weighting.ipynb
MATCH_TARGETS = ("harmonic", "min", "nc", "cc")


def n_eff(w):
    """Kish effective sample size (sum w)^2 / sum w^2 of a weight vector (0 for an all-zero vector)."""
    w = np.asarray(w, float)
    s2 = (w ** 2).sum()
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def match_weights(log10_x, is_nc, n_bins=40, edges=None, target="harmonic", min_count=5, max_weight=None):
    """Weights that give both classes the same distribution of x (log10 E_vis or log10 Q_tot) in bins, keeping every event.

    Returns (w, edges).  Events in bins where either class has fewer than min_count events get w = 0 (no overlap to match to).
    target: "harmonic"  both classes to 2 p_CC p_NC / (p_CC + p_NC): weights bounded by 2, mean 1 over the non-zero events of
                        each class so the classes keep their relative size (recommended when neither class is the reference);
            "min"       both classes to min(p_CC, p_NC), the weighted analogue of min-count subsampling, weights bounded by 1;
            "nc"        CC reweighted onto the NC histogram: w_CC = N_NC(bin) / N_CC(bin), NC kept at 1, so the weighted CC
                        counts equal the NC counts bin by bin (unbounded: max_weight clips the ratio; no renormalisation);
            "cc"        the mirror image, NC onto CC.
    """
    if target not in MATCH_TARGETS:
        raise ValueError(f"target must be one of {MATCH_TARGETS}, got {target!r}")
    x, nc = np.asarray(log10_x, float), np.asarray(is_nc, bool)
    if edges is None:
        edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    counts = np.stack([np.bincount(b[~nc], minlength=len(edges) - 1),
                       np.bincount(b[nc], minlength=len(edges) - 1)]).astype(float)        # [class, bin], 0 = CC, 1 = NC
    overlap = (counts >= min_count).all(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if target in ("harmonic", "min"):
            p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)                   # per-class densities
            tgt = 2 * p[0] * p[1] / (p[0] + p[1]) if target == "harmonic" else np.minimum(p[0], p[1])
            tgt = np.where(overlap, np.nan_to_num(tgt), 0.0)
            ratio = np.where(p > 0, tgt[None, :] / np.where(p > 0, p, 1.0), 0.0)               # [class, bin]
        else:
            ref, other = (1, 0) if target == "nc" else (0, 1)                                  # ref keeps weight 1
            ratio = np.ones_like(counts)
            ratio[other] = counts[ref] / np.where(counts[other] > 0, counts[other], 1.0)
            ratio = np.where(overlap[None, :], ratio, 0.0)
    if max_weight is not None:
        ratio = np.minimum(ratio, float(max_weight))
    w = ratio[nc.astype(int), b]
    if target in ("harmonic", "min"):
        for k in (False, True):                                                                # mean 1 per class (non-zero)
            m = (nc == k) & (w > 0)
            if m.any():
                w[m] /= w[m].mean()
    return w, edges


def evis_match_weights(log10_evis, is_nc, n_bins=40, edges=None, target="harmonic", min_count=5):
    """match_weights on log10 E_vis (kept for older callers)."""
    return match_weights(log10_evis, is_nc, n_bins=n_bins, edges=edges, target=target, min_count=min_count)


def weight_tables():
    """Every per-dataset weight table pythia_*/weights_*.parquet (flux_weighting.ipynb, make_weights.py); the (cls, part,
    eventID) keys name the dataset, so the tables can be concatenated."""
    return sorted(ROOT.glob("pythia_*/weights_*.parquet"))


def load_flux_weights(cls, part, event_id, flux="astro_g2", path=WEIGHT_TABLE, column=None):
    """Per-event weight of the weight table joined on (cls, part, eventID): the flux weight w_<flux> (expected events per
    year in the generation volume) or, with column, any column of the table (e.g. "w_qmatch").  With the default path every
    table of weight_tables() that has the column is used, so any dataset tag resolves.

    cls, part, eventID are as pulse_regressor.py writes them (directory name, part_%04d_hits.root, event index).
    Returns a float array with NaN where no row of the weight table matches; None if the table does not exist.
    """
    col = column or f"w_{flux}"
    tabs = weight_tables() if Path(path) == WEIGHT_TABLE else [Path(path)]       # default: every table (one per dataset tag)
    tabs = [t for t in tabs if t.exists() and col in pd.read_parquet(t).columns]
    if not tabs:
        return None
    tab = pd.concat([pd.read_parquet(t, columns=["cls", "part", "eventID", col]) for t in tabs], ignore_index=True)
    keys = pd.DataFrame({"cls": np.asarray(cls).astype(str), "part": np.asarray(part).astype(str),
                         "eventID": np.asarray(event_id).astype(int)})
    out = keys.merge(tab, on=["cls", "part", "eventID"], how="left")
    return out[col].to_numpy(float)


def wpercentile(x, w, qs):
    """Weighted percentiles qs (in %) along axis 0 of x [N] or [N, M] with weights w [N]: the smallest value whose cumulative
    weight reaches q % of the total (the lower unweighted percentile when all weights are equal).
    Returns one entry per q: a float for 1-D x, an array [M] for 2-D x.
    """
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not w.sum() > 0:
        raise ValueError("wpercentile: the weights sum to zero")
    x2 = x[:, None] if x.ndim == 1 else x
    o = np.argsort(x2, axis=0, kind="stable")
    xs, cw = np.take_along_axis(x2, o, 0), np.cumsum(w[o], axis=0)
    cw = cw / cw[-1]
    out = []
    for q in qs:
        idx = np.minimum((cw < q / 100.0).sum(0), len(x2) - 1)
        r = np.take_along_axis(xs, idx[None, :], 0)[0]
        out.append(float(r[0]) if x.ndim == 1 else r)
    return out


def log_prior_ratio(log10_evis, is_nc, w_flux, edges, floor=1e-12):
    """ln[N_NC(E) / N_CC(E)] per log10 E_vis bin from flux-weighted counts; returns (table, interpolator).

    The interpolator evaluates the log prior ratio at any log10 E_vis (piecewise constant per bin, clamped at the ends).
    Bins where either class has no weight get the floor, i.e. an extreme prior; use the edges of the matching.
    """
    e, nc, w = np.asarray(log10_evis, float), np.asarray(is_nc, bool), np.nan_to_num(np.asarray(w_flux, float))
    h_nc = np.histogram(e[nc], edges, weights=w[nc])[0]
    h_cc = np.histogram(e[~nc], edges, weights=w[~nc])[0]
    table = np.log(np.maximum(h_nc, floor)) - np.log(np.maximum(h_cc, floor))

    def lpr(x):
        b = np.clip(np.digitize(np.asarray(x, float), edges) - 1, 0, len(edges) - 2)
        return table[b]

    return table, lpr


def prior_shift(p_matched, log10_evis, lpr, eps=1e-7):
    """Physical posterior from an energy-matched one: sigmoid(logit p + ln[pi_NC(E)/pi_CC(E)])."""
    p = np.clip(np.asarray(p_matched, float), eps, 1 - eps)
    z = np.log(p / (1 - p)) + lpr(log10_evis)
    return 1.0 / (1.0 + np.exp(-z))
