#!/usr/bin/env python
"""Timing distributions of the NC/CC benchmark written by classifier.py --build.

Reads classifier_data_truth.parquet (one row per event, raw hit times included)
and draws, per class:
  1. residual-time PDF relative to the first detected photon (log scale)
  2. its normalised cumulative distribution
  3. the mean normalised 1 ns waveform the classifier sees (with 16-84% band)
  4. the distribution of the mixture-fit fraction y_hat_MLE

usage: python plot_benchmark.py [--truth ../classifier_data_truth.parquet] [--out out]
                                [--tmax 40] [--n-bins 40] [--y-bins 20] [--wf-bin X] [--wf-nbins N]
Panels 1 and 2 use the --wf-bin width by default; --n-bins overrides it (width = tmax / n_bins);
the waveform panel uses the classifier's own binning (wf_* columns, bin width from
classifier_data_meta.json).
Paths default relative to this file's location (wavepid/), so it runs from anywhere.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = {  # label: (name, colour)  -- hadronic = orange, EM+hadronic = blue as in the WavePID plots
    1.0: (r"NC: $\pi^-$ (hadronic)", "#eb6834"),
    0.0: (r"CC: $e^- + \pi^-$ (EM + hadronic)", "#2a78d6"),
}
WF_BIN_NS_DEFAULT = 1.0   # used only if classifier_data_meta.json is missing


def style(ax):
    ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#5d5d57", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--truth", default=os.path.join(here, os.pardir, "classifier_data_truth.parquet"),
                    help="truth parquet written by classifier.py --build")
    ap.add_argument("--out", default=os.path.join(here, "out"), help="directory for the PNG")
    ap.add_argument("--tmax", type=float, default=40.0, help="upper edge of the residual-time axis [ns]")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="residual-time bins between 0 and tmax for panels 1-2 "
                         "(default: same bin width as the waveform panel, i.e. --wf-bin)")
    ap.add_argument("--y-bins", type=int, default=20, help="bins for the y_hat_MLE histogram")
    ap.add_argument("--wf-bin", type=float, default=None,
                    help="waveform bin width [ns] to display (default: the dataset's own width). "
                         "An integer multiple of the stored width merges the classifier's bins exactly; "
                         "any other width is recomputed from the raw hit times of all photons")
    ap.add_argument("--wf-nbins", type=int, default=None,
                    help="waveform bins to show after rebinning (default: all available)")
    args = ap.parse_args()
    plt.rcParams["mathtext.fontset"] = "cm"

    df = pd.read_parquet(args.truth)
    WF = sorted(c for c in df.columns if c.startswith("wf_"))   # waveform columns written by classifier.py
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.truth)), "classifier_data_meta.json")
    stored_bin = WF_BIN_NS_DEFAULT
    if os.path.exists(meta_path):
        stored_bin = float(json.load(open(meta_path)).get("wf_bin_ns", stored_bin))
    wf_bin = args.wf_bin if args.wf_bin else stored_bin
    ratio = wf_bin / stored_bin
    if abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1:
        # merge k stored bins -> exactly what the classifier saw, coarser
        k = int(round(ratio))
        n_out = len(WF) // k
        wf_matrix = df[WF].to_numpy()[:, :n_out * k].reshape(len(df), n_out, k).sum(axis=2)
        recomputed = False
    else:
        # arbitrary width: rebuild the normalised waveform from the raw hit times (all photons)
        n_out = int(np.ceil(len(WF) * stored_bin / wf_bin))
        wf_matrix = np.zeros((len(df), n_out))
        for i, t in enumerate(df["times"]):
            t = np.asarray(t); h, _ = np.histogram(t - t.min(), bins=np.arange(n_out + 1) * wf_bin)
            wf_matrix[i] = h / max(len(t), 1)
        recomputed = True
    if args.wf_nbins is not None:
        if args.wf_nbins > n_out:
            print(f"  --wf-nbins {args.wf_nbins} > {n_out} bins available at {wf_bin:g} ns; showing all {n_out}")
        n_out = max(1, min(args.wf_nbins, n_out))
        wf_matrix = wf_matrix[:, :n_out]
    wf_edges = np.arange(n_out + 1) * wf_bin
    WF_BIN_NS = wf_bin
    # panels 1-2 follow the waveform bin width unless --n-bins is given explicitly
    if args.n_bins is None:
        edges = np.arange(0.0, args.tmax + wf_bin / 2, wf_bin)
    else:
        edges = np.linspace(0.0, args.tmax, args.n_bins + 1)
    bin_w = edges[1] - edges[0]
    # photons per event the classifier saw: a fixed subsample, or all of them (n_used == nhits)
    if recomputed:
        wf_note = f"\n(recomputed from all photons, {wf_bin:g} ns bins)"
    elif "n_used" in df and df["n_used"].nunique() == 1 and (df["n_used"] < df["nhits"]).any():
        wf_note = f"\n({int(df['n_used'].iloc[0])} photons per event, {wf_bin:g} ns bins)"
    else:
        wf_note = f"\n(all photons of each event, {wf_bin:g} ns bins)"
    print(f"{len(df)} events: " + ", ".join(
        f"{CLASSES[l][0].split(':')[0]} {int((df['is_nc']==l).sum())} (mean {df.loc[df['is_nc']==l,'nhits'].mean():.0f} photons)"
        for l in (1.0, 0.0)))

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    for label, (name, col) in CLASSES.items():
        sel = df[df["is_nc"] == label]
        dt = np.concatenate([np.asarray(t) - np.min(t) for t in sel["times"]])
        c, _ = np.histogram(dt, bins=edges)
        axes[0].stairs(c / max(c.sum() * bin_w, 1e-12), edges, color=col, lw=2, label=name)
        cdf = np.array([np.mean(dt <= t) for t in edges])
        axes[1].plot(edges, cdf, color=col, lw=2, label=name)
        wf = wf_matrix[(df["is_nc"] == label).to_numpy()]
        med, lo, hi = np.median(wf, 0), np.percentile(wf, 16, 0), np.percentile(wf, 84, 0)
        axes[2].stairs(med, wf_edges, color=col, lw=2, label=name)
        axes[2].fill_between(wf_edges, np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                             step="post", color=col, alpha=0.15, lw=0)
        axes[3].hist(sel["y_hat_mle"], bins=np.linspace(0, 1, args.y_bins + 1), histtype="step", color=col, lw=2,
                     density=True, label=name)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$p(\Delta t_\gamma)\;[\mathrm{ns}^{-1}]$")
    axes[0].set_title(f"Residual-time PDF\n(all photons, {bin_w:g} ns bins)", fontsize=10, loc="left")
    axes[1].set_ylabel(r"$P(\Delta t_\gamma' \leq \Delta t_\gamma)$"); axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Normalised cumulative\ndistribution", fontsize=10, loc="left")
    for ax in axes[:2]:
        ax.set_xlabel(r"Residual Time [ns]"); ax.set_xlim(edges[0], edges[-1])
    axes[2].set_xlabel(r"Residual Time [ns]"); axes[2].set_ylabel(f"fraction of photons per {WF_BIN_NS:g} ns bin")
    axes[2].set_xlim(wf_edges[0], wf_edges[-1])
    axes[2].set_title("Classifier waveform, median, 16–84% band" + wf_note, fontsize=10, loc="left")
    axes[3].set_xlabel(r"$\hat{y}_{\mathrm{MLE}}$"); axes[3].set_ylabel("density")
    axes[3].set_title("Mixture-fit fraction\n" + r"$f = \hat{y}\, f_{\pi^-} + (1-\hat{y})\, f_{e^-}$", fontsize=10, loc="left")
    for ax in axes: style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    e = df["e_nu"].mean()
    fig.suptitle(rf"NC/CC benchmark timing distributions $\cdot$ {len(df)} events $\cdot$ "
                 rf"residual time relative to first detected photon", fontsize=11, x=0.01, ha="left")
    os.makedirs(args.out, exist_ok=True)
    f = os.path.join(args.out, "benchmark_timing.png"); fig.savefig(f, dpi=160); print("wrote", f)


if __name__ == "__main__":
    main()
