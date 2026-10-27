#!/usr/bin/env python3
"""WavePID -> Pynu connector: classifier response table from classify_gamma's validation predictions.

Reads classify_gamma_<source>_val_preds.npz (keys: y = is_nc, p = P(NC), log10_evis), applies ONE
threshold t on P(NC) to sort every event into
    NC-like  : P(NC) >= t          (default t = 0.5)
    CC-like  : P(NC) <  t
and tabulates, per true class (nc, cc) and per energy bin, R(E) = P(NC-like | class, E) with its
binomial error. This is what Pynu's 3-morphology switch ({NC-like, CC-like, tracks}) applies to the
release MC in the mixed PID bin: a true-NC event of reconstructed energy E enters NC-like with
weight w * R_nc(E) and CC-like with w * (1 - R_nc(E)); a nu_e CC event with R_cc(E).

Energy variable: the OMSim E_vis (deposited light energy) is matched to the release's reco_energy,
so the default edges are the release's reconstructed-energy bins. Beyond the sample's E_vis range
the class's average response over all its events is carried over (flagged in the JSON).

Usage (in WavePID/PulseShapeInference, after classify_gamma.py --source pred --train):
    python classifier_response.py --source pred --tag gev_evisz_s85_both_none_nomatch   # all ~2.4k regressor-validation events, t = 0.5
    python classifier_response.py --source pred                   # untagged build, t = 0.5
    python classifier_response.py --source pred --scan            # efficiency/purity for a grid of t
    python classifier_response.py --source truth --out response_truth.json
"""
import argparse, json
from pathlib import Path
import numpy as np

# IceCube DeepCore 9-yr release reconstructed-energy bin edges [GeV] (the analysis binning)
E_EDGES = [6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363, 27.3136977,
           36.61429921, 49.08185342, 65.79474104, 88.19854278, 158.49]
CATS = ["nc_like", "cc_like"]


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["y"].astype(int), d["p"].astype(float), d["log10_evis"].astype(float)


def categorise(p, t):
    return np.where(p >= t, 0, 1)                 # 0 = NC-like, 1 = CC-like


def response(y, p, log10e, t, edges):
    e = 10.0 ** log10e
    cat = categorise(p, t)
    ib = np.digitize(e, edges) - 1
    out = {}
    for cls, lab in (("nc", 1), ("cc", 0)):
        sel = (y == lab)
        eps = np.full((len(edges) - 1, 2), np.nan)
        err = np.full((len(edges) - 1, 2), np.nan)
        n = np.zeros(len(edges) - 1, int)
        for b in range(len(edges) - 1):
            m = sel & (ib == b)
            n[b] = m.sum()
            if n[b] == 0:
                continue
            for k in range(2):
                f = np.mean(cat[m] == k)
                eps[b, k] = f
                err[b, k] = np.sqrt(f * (1 - f) / n[b])
        # carry the average response of the class (over all its events, i.e. the populated bins weighted by their
        # counts) into empty bins: the response is flat in energy where it was measured, so the mean is the best guess
        extrapolated = []
        if sel.sum():
            avg = np.array([np.mean(cat[sel] == k) for k in range(2)])
            avg_err = np.sqrt(avg * (1 - avg) / sel.sum())
            for b in range(len(edges) - 1):
                if n[b] == 0:
                    eps[b], err[b] = avg, avg_err
                    extrapolated.append(b)
        tot = {k: float(np.mean(cat[sel] == i)) for i, k in enumerate(CATS)}
        out[cls] = {"n_per_bin": n.tolist(), "eps": eps.tolist(), "err": err.tolist(),
                    "extrapolated_bins": extrapolated, "overall": tot}
    # separation summary
    nc_like_purity = np.sum((cat == 0) & (y == 1)) / max(np.sum(cat == 0), 1)
    cc_like_purity = np.sum((cat == 1) & (y == 0)) / max(np.sum(cat == 1), 1)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = None
    return out, {"auc": auc, "nc_like_purity": float(nc_like_purity), "cc_like_purity": float(cc_like_purity)}


def print_table(resp, edges):
    print(f"{'E_reco [GeV]':>16s} | {'n_nc':>5s} {'R_nc=P(NC-like|NC)':>19s} | {'n_cc':>5s} {'R_cc=P(NC-like|CC)':>19s}")
    for b in range(len(edges) - 1):
        row = f"{edges[b]:6.1f}-{edges[b+1]:6.1f}  |"
        for cls in ("nc", "cc"):
            r = resp[cls]
            flag = "*" if b in r["extrapolated_bins"] else " "
            row += f" {r['n_per_bin'][b]:5d}      {r['eps'][b][0]:6.3f} ± {r['err'][b][0]:5.3f}{flag}  |"
        print(row)
    print("  (* = no events in this bin; the class's average response over all its events carried over)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("truth", "pred"), default="pred")
    ap.add_argument("--tag", default="", help="dataset tag of the classify_gamma build, e.g. gev_evisz_s85_both_none_nomatch "
                                              "(file classify_gamma_<source>_<tag>_all_preds.npz)")
    ap.add_argument("--held-out-only", action="store_true", help="use the classifier's held-out half (_val_preds) instead of all events (_all_preds)")
    ap.add_argument("--npz", default=None, help="override: path to the predictions npz")
    ap.add_argument("--t", type=float, default=0.5, help="threshold on P(NC) for NC-like")
    ap.add_argument("--edges", type=float, nargs="+", default=E_EDGES, help="energy bin edges [GeV]")
    ap.add_argument("--scan", action="store_true", help="print purity/efficiency for a grid of thresholds and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    stem = f"classify_gamma_{a.source}" + (f"_{a.tag}" if a.tag else "")
    kind = "val" if a.held_out_only else "all"
    npz = Path(a.npz) if a.npz else here / f"{stem}_{kind}_preds.npz"
    if not npz.exists() and kind == "all":
        npz = here / f"{stem}_val_preds.npz"                    # older classify_gamma runs wrote the held-out half only
        print(f"(no _all_preds file for this tag: falling back to the held-out half {npz.name}; retrain with classify_gamma.py to get all events)")
    y, p, log10e = load(npz)
    print(f"{npz.name}: {y.size} events, {int(y.sum())} NC / {int((1-y).sum())} CC, "
          f"E_vis {10**log10e.min():.1f}-{10**log10e.max():.1f} GeV")

    if a.scan:
        print(f"{'t':>5s} | {'NC eff':>7s} {'NC-like pur':>11s} | {'CC eff':>7s} {'CC-like pur':>11s}")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            resp, s = response(y, p, log10e, t, a.edges)
            print(f"{t:5.2f} | {resp['nc']['overall']['nc_like']:7.3f} {s['nc_like_purity']:11.3f} | "
                  f"{resp['cc']['overall']['cc_like']:7.3f} {s['cc_like_purity']:11.3f}")
        return

    resp, summary = response(y, p, log10e, a.t, a.edges)
    print(f"threshold t={a.t}: AUC={summary['auc']}, NC-like purity {summary['nc_like_purity']:.3f}, "
          f"CC-like purity {summary['cc_like_purity']:.3f}\n")
    print_table(resp, a.edges)

    out = Path(a.out) if a.out else here / f"classifier_response_{a.source}{'_' + a.tag if a.tag else ''}_t{a.t}.json"
    json.dump({"source": str(npz.name), "classifier": f"classify_gamma --source {a.source}",
               "threshold": a.t, "morphologies": ["nc_like", "cc_like", "tracks"],
               "energy_variable": "reco_energy (release) <-> E_vis (OMSim)",
               "energy_edges_gev": list(a.edges), "categories": CATS,
               "classes": {"nc": "true NC (any flavour)", "cc": "true nu_e CC"},
               "summary": summary, "response": resp,
               "assumptions_for_other_classes": {
                   "nutau_cc": "0.65*nc + 0.35*cc (hadronic vs leptonic tau decays) until an OMSim nu_tau class exists",
                   "numu_cc_in_mixed_bin": "cc response (short track + hadrons) until an OMSim nu_mu class exists",
                   "atm_muons": "cc_like (elongated profile), own nuisance"}},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()