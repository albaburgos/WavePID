"""Visible-energy spectra weighted by the classifier's P(NC) and P(CC) against the true NC and CC spectra, and ROC.

For every event the (ln alpha, ln beta) classifiers of classify_gamma.py give P(NC | x), once with the true
profile x = y (classifier trained with --source truth) and once with the light-inferred one x = mu (--source pred),
both from pulse_regressor_neptune_val.npz and the latest checkpoint under logs/classify_gamma/<source>.  Summing
P(NC) and P(CC) = 1 - P(NC) per energy bin gives the "soft-counted" class spectra; if the classifier is calibrated
for the population it is applied to, these sums reproduce the true class counts bin by bin:
    sum_i P(NC | x_i) 1[E_i in bin] ~ N_NC(bin).

Population: all events of the prediction file, unmatched.  CC has a 6x higher median E_vis than NC, so here the
equal-prior posterior learnt on energy-matched classes is applied where the class prior depends on energy, and
the soft counts show what that costs.

Figure (2 x 3): one column per class with the true spectrum (green, solid) and the two soft-counted spectra
(dotted: red = true (alpha, beta) as input, purple = inferred (alpha, beta) as input) on top and the ratio
soft / true per bin below; right column: ROC of P(NC) with the AUC and the AUC per energy bin for both inputs.

Run from this directory:  python plot_energy_weighted.py    -> figs/classify_gamma_energy.png
Settings are the constants below (checkpoints, bins, output).
"""
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from classify_gamma import ENERGY, FEATURES, LABEL, LOG_DIR, GammaClassifier, best_checkpoint, load_events, paths

ROOT = Path(__file__).resolve().parent
SOURCES = ("truth", "pred")                   # the classify_gamma.py classifiers to apply: on y and on mu
CKPT = {"truth": None, "pred": None}          # None: latest checkpoint of the newest run under logs/classify_gamma/<source>
BINS = 24                                     # log10 E_vis bins for the spectra
E_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # log10 E_vis bins for the AUC: 1-3-10-32-100 GeV
OUT = ROOT / "figs" / "classify_gamma_energy.png"
GREEN, RED, PURPLE = "#008300", "#e34948", "#4a3aa7"     # true spectrum; input = true (alpha, beta); input = inferred
COLOR = {"truth": RED, "pred": PURPLE}
NAME = {"truth": "input: true (alpha, beta)", "pred": "input: inferred (alpha, beta)"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def latest_model(source):
    """The classifier of CKPT[source], or of the latest checkpoint of the newest run of that source."""
    runs = list((LOG_DIR / source / "lightning_logs").glob("version_*"))
    if not runs and not CKPT.get(source):
        raise SystemExit(f"no run under {LOG_DIR / source}: run classify_gamma.py --source {source} --build --train first")
    ckpt = Path(CKPT[source]) if CKPT.get(source) else best_checkpoint(max(runs, key=lambda p: int(p.name.split("_")[1])))
    return GammaClassifier.load_from_checkpoint(ckpt, map_location="cpu"), ckpt


def validation_split(source):
    """The validation half of the iwpc dataset (PandasDirDataModule: first ceil(N/2) files train, rest validation)."""
    data_dir = paths(source)[0]
    files = sorted(data_dir.glob("file_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no dataset files in {data_dir}: run classify_gamma.py --source {source} --build first")
    return pd.concat(map(pd.read_pickle, files[math.ceil(len(files) * 0.5):]), ignore_index=True)


def predict(model, df, features):
    model.eval().cpu()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(df[features].to_numpy(np.float32)))).numpy()[:, 0]


def draw_class(ax_top, ax_ratio, lab, name, samples, edges):
    """One class: true spectrum (green) and the soft-counted spectra of every classifier in `samples` (dotted),
    ratio soft / true per bin below.  samples = {source: (df, P(NC))}; returns {source: (true, soft)}."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    df0 = next(iter(samples.values()))[0]
    e, y = df0[ENERGY].to_numpy(), df0[LABEL].to_numpy()
    truth = np.histogram(e[y == lab], edges)[0]
    ax_top.stairs(truth, edges, fill=True, color=GREEN, alpha=0.15, lw=0)
    ax_top.stairs(truth, edges, color=GREEN, lw=1.8, label=f"true {name} ({truth.sum():,})")
    out = {}
    for source, (df, p) in samples.items():
        w = p if lab == 1 else 1 - p
        soft = np.histogram(df[ENERGY].to_numpy(), edges, weights=w)[0]
        ax_top.stairs(soft, edges, color=COLOR[source], lw=2.2, ls=":", label=f"sum P({name}), {NAME[source]} ({soft.sum():,.0f})")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(truth > 0, soft / truth, np.nan)
            err = np.where(truth > 0, np.sqrt(truth) / truth, np.nan)          # Poisson error of the true count
        ax_ratio.errorbar(centres, ratio, yerr=err, fmt="o", ms=3.5, color=COLOR[source], lw=1, label=NAME[source])
        out[source] = (truth, soft)
    ax_ratio.axhline(1, color=MUTED, lw=1, ls=":")
    ax_top.set_title(f"{name} events: true spectrum and soft counts", fontsize=9, color=INK2)
    ax_top.set_ylabel("events per bin"); ax_top.legend(fontsize=7.5, frameon=False)
    ax_ratio.set_ylabel("sum P / true"); ax_ratio.set_ylim(0, 2.5); ax_ratio.legend(fontsize=7.5, frameon=False)
    ax_ratio.set_xlabel("log10 (E_vis / GeV)")
    for ax in (ax_top, ax_ratio):
        ax.set_xlim(edges[0], edges[-1]); style(ax)
    return out


def binned_auc(y, p, e, edges, min_events=40):
    """AUC of P(NC) per bin of log10 E (NaN where a bin has too few events or one class only)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e >= lo) & (e < hi)
        out.append(roc_auc_score(y[m], p[m]) if m.sum() >= min_events and 0 < y[m].mean() < 1 else np.nan)
    return np.array(out)


def draw_auc(ax_roc, ax_bins, samples):
    """ROC curves with AUC (ax_roc) and AUC per energy bin (ax_bins) for {source: (df, P(NC))}."""
    for source, (df, p) in samples.items():
        y, e = df[LABEL].to_numpy(), df[ENERGY].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=COLOR[source], lw=1.8, label=f"{NAME[source]}: AUC {roc_auc_score(y, p):.3f}")
        ax_bins.plot(0.5 * (E_EDGES[:-1] + E_EDGES[1:]), binned_auc(y, p, e, E_EDGES), "o-", color=COLOR[source], lw=1.6, label=NAME[source])
    ax_roc.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1); ax_roc.set_aspect("equal")
    ax_roc.set_xlabel("CC called NC (false positive rate)"); ax_roc.set_ylabel("NC called NC (true positive rate)")
    ax_roc.set_title("ROC of P(NC | alpha, beta)", fontsize=9, color=INK2); ax_roc.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax_bins.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax_bins.set_xticks(E_EDGES); ax_bins.set_xticklabels([f"{10 ** x:.0f}" for x in E_EDGES])
    ax_bins.set_xlabel("visible energy [GeV]"); ax_bins.set_ylabel("AUC per bin"); ax_bins.set_ylim(0.45, 0.9)
    ax_bins.legend(fontsize=7.5, frameon=False)
    for ax in (ax_roc, ax_bins):
        style(ax)


def main():
    matplotlib.use("Agg")                     # here, not at import: notebooks import this module and draw inline
    # every event of the prediction file, unmatched, scored with the truth-input and the inferred-input classifier
    samples = {}
    for source in SOURCES:
        meta = json.loads(paths(source)[1].read_text())
        model, ckpt = latest_model(source)
        df = load_events(Path(meta["pred"]), source)
        samples[source] = (df, predict(model, df, FEATURES))
        print(f"{source}: classifier {ckpt.relative_to(ROOT)} applied to {'y (true profile)' if source == 'truth' else 'mu (inferred profile)'}")
    any_df = next(iter(samples.values()))[0]
    edges = np.linspace(any_df[ENERGY].min(), any_df[ENERGY].max() + 1e-9, BINS + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 7.8), facecolor="#fcfcfb", gridspec_kw=dict(height_ratios=(3, 1.6)))
    fig.suptitle(f"all events of the prediction file, unmatched ({len(any_df):,}): true class spectra (green) against the "
                 "spectra soft-counted with P(NC), P(CC) (dotted); ROC of P(NC | alpha, beta)", fontsize=10.5, color=INK, y=0.995)
    hist = {}
    for j, (lab, name) in enumerate(((0, "CC"), (1, "NC"))):
        hist[name] = draw_class(axes[0, j], axes[1, j], lab, name, samples, edges)
    draw_auc(axes[0, 2], axes[1, 2], samples)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=fig.get_facecolor())

    for source in SOURCES:
        df, p = samples[source]
        print(f"{source}: AUC {roc_auc_score(df[LABEL].to_numpy(), p):.3f}; "
              f"true NC {hist['NC'][source][0].sum():,} vs sum P(NC) {hist['NC'][source][1].sum():,.0f}; "
              f"true CC {hist['CC'][source][0].sum():,} vs sum P(CC) {hist['CC'][source][1].sum():,.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
