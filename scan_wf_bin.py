#!/usr/bin/env python
"""Scan the waveform bin width of the NC/CC benchmark and compare ROC curves.

For each --wf-bin value the benchmark is rebuilt (classifier.py --build) with the
number of bins chosen so that the waveform always covers the same time window
(--reach, default 16 ns), the MLP is retrained (classifier.py --train), and the
validation predictions are collected. Each setting is repeated for several seeds
(different photon subsample / event pairing / network init) and the ROC curves
are averaged on a common false-positive-rate grid.

usage: python scan_wf_bin.py [--bins 4 2 1 0.5 0.1] [--seeds 3] [--epochs 100] [--reach 16]
"""
import argparse, os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
CLASSIFIER = os.path.join(HERE, "classifier.py")
PRED = os.path.join(PROJECT, "classifier_val_preds.npz")
# fixed categorical order, coarse -> fine
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def style(ax):
    ax.grid(color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#5d5d57", labelsize=9)


def run(cmd, env):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=float, nargs="+", default=[4, 2, 1, 0.5, 0.1], help="waveform bin widths [ns]")
    ap.add_argument("--reach", type=float, default=16.0, help="time window covered by the waveform [ns]")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-subsample", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    plt.rcParams["mathtext.fontset"] = "cm"
    env = dict(os.environ, DISABLE_IWPC_WELCOME="1")
    store = os.path.join(args.out, "scan_wf_bin"); os.makedirs(store, exist_ok=True)

    fpr_grid = np.linspace(0, 1, 201)
    results = {}
    for wf_bin in args.bins:
        nbins = int(round(args.reach / wf_bin))
        tprs, aucs, aucs_mle = [], [], []
        for seed in range(args.seeds):
            env["CLASSIFIER_TORCH_SEED"] = str(1234 + seed)
            run([sys.executable, CLASSIFIER, "--build", "--wf-bin", str(wf_bin), "--wf-nbins", str(nbins),
                 "--n-subsample", str(args.n_subsample), "--seed", str(1234 + seed), "--n-files", "4"], env)
            run([sys.executable, CLASSIFIER, "--train", "--epochs", str(args.epochs),
                 "--accelerator", "cpu", "--batch-size", "32"], env)
            d = np.load(PRED)
            y, p = d["y"], d["p"]
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(fpr_grid, fpr, tpr)); aucs.append(roc_auc_score(y, p))
            shutil.copy(PRED, os.path.join(store, f"preds_wfbin{wf_bin:g}_seed{seed}.npz"))
            print(f"wf_bin={wf_bin:g} ns ({nbins} bins) seed={seed}: AUC={aucs[-1]:.3f}  ({len(y)} val events)")
        results[wf_bin] = dict(nbins=nbins, tpr=np.mean(tprs, 0), tpr_sd=np.std(tprs, 0),
                               auc=np.mean(aucs), auc_sd=np.std(aucs), aucs=aucs)

    # ---- ROC curves on the common grid + AUC vs bin width ------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for (wf_bin, r), col in zip(results.items(), COLORS):
        axes[0].plot(fpr_grid, r["tpr"], color=col, lw=2,
                     label=rf"{wf_bin:g} ns $\times$ {r['nbins']} bins, AUC = {r['auc']:.2f} $\pm$ {r['auc_sd']:.2f}")
        axes[0].fill_between(fpr_grid, r["tpr"] - r["tpr_sd"], r["tpr"] + r["tpr_sd"], color=col, alpha=0.10, lw=0)
    axes[0].plot([0, 1], [0, 1], color="#9a9a92", lw=1, ls="--")
    axes[0].set_xlabel("false positive rate  (CC called NC)"); axes[0].set_ylabel("true positive rate  (NC called NC)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_title(rf"ROC, mean $\pm$ s.d. over {args.seeds} seeds", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    xs = list(results); means = [results[b]["auc"] for b in xs]; sds = [results[b]["auc_sd"] for b in xs]
    axes[1].errorbar(xs, means, yerr=sds, color="#2a78d6", lw=2, marker="o", ms=6, capsize=3)
    for b in xs:
        axes[1].scatter([b] * len(results[b]["aucs"]), results[b]["aucs"], color="#2a78d6", alpha=0.35, s=14, zorder=3)
    axes[1].axhline(0.5, color="#9a9a92", lw=1, ls="--")
    axes[1].set_xscale("log"); axes[1].set_xticks(xs); axes[1].set_xticklabels([f"{b:g}" for b in xs])
    axes[1].set_xlabel("waveform bin width [ns]"); axes[1].set_ylabel("validation AUC")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title(f"AUC vs bin width ({args.reach:g} ns window, {args.n_subsample} photons/event)", fontsize=10, loc="left")
    for ax in axes: style(ax)
    fig.suptitle("NC/CC shape-only classifier vs waveform resolution", fontsize=11, x=0.01, ha="left")
    f = os.path.join(args.out, "roc_wf_bin.png"); fig.savefig(f, dpi=160); print("wrote", f)
    np.savez(os.path.join(store, "summary.npz"), fpr=fpr_grid,
             **{f"tpr_{b:g}": results[b]["tpr"] for b in results}, **{f"auc_{b:g}": results[b]["aucs"] for b in results})


if __name__ == "__main__":
    main()
