"""ROC / AUC and training curves for one classifier.py run (a Lightning version dir).

classifier.py logs train/val loss and accuracy per epoch to
classifier/lightning_logs/version_N and writes the validation predictions of
its last training run to IceCube/classifier_val_preds<_tag>.npz. This script
draws, for one version,

    left    ROC curve of the saved validation predictions, AUC with a bootstrap
            68% interval in the legend
    middle  train / validation loss per epoch (from the TensorBoard events)
    right   train / validation accuracy per epoch

The predictions file is not tied to a version, so the script compares its
modification time with the events file and warns when they differ by more than
--max-skew-min (the file may then belong to another run).

Usage (from wavepid/):
    python plot_classifier_run.py --version 26
    python plot_classifier_run.py --version 25 --dataset-tag genie
    python plot_classifier_run.py --version 26 --preds ../classifier_val_preds.npz --out out/v26.png
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LOGDIR = ROOT / "classifier" / "lightning_logs"

# validated reference palette (dataviz skill): slot 1 blue, slot 2 orange; chrome inks
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def latest_version(logdir: Path) -> int:
    versions = [int(m.group(1)) for p in logdir.glob("version_*")
                if (m := re.fullmatch(r"version_(\d+)", p.name))]
    if not versions:
        raise SystemExit(f"no version_* directories in {logdir}")
    return max(versions)


def load_scalars(run_dir: Path) -> dict:
    """{tag: (epochs, values)} for the per-epoch metrics classifier.py logs."""
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out = {}
    for name in ("train_loss", "val_loss", "train_acc", "val_acc"):
        tag = f"{name}_epoch" if f"{name}_epoch" in tags else (f"{name}_step" if f"{name}_step" in tags else None)
        if tag is None:
            continue
        ev = ea.Scalars(tag)
        out[name] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    if not out:
        raise SystemExit(f"no loss/accuracy scalars in {run_dir}; tags: {sorted(tags)}")
    return out


def bootstrap_auc(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [16, 84])


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, default=None, help="Lightning version number (default: latest)")
    ap.add_argument("--logdir", type=Path, default=LOGDIR)
    ap.add_argument("--dataset-tag", default=None, help="picks ../classifier_val_preds_<tag>.npz")
    ap.add_argument("--preds", type=Path, default=None, help="validation predictions npz (overrides --dataset-tag)")
    ap.add_argument("--out", type=Path, default=None, help="default: out/classifier_run_v<N>.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-skew-min", type=float, default=10.0,
                    help="warn when predictions and events files are further apart in time than this")
    ap.add_argument("--bare", action="store_true",
                    help="ROC curve only: no title, legend, axis labels or annotations (tick numbers stay); "
                         "written to out/classifier_run_v<N>_roc_bare.png unless --out is given")
    args = ap.parse_args()

    version = latest_version(args.logdir) if args.version is None else args.version
    run_dir = args.logdir / f"version_{version}"
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise SystemExit(f"no events file in {run_dir}")
    preds = args.preds or PROJECT / f"classifier_val_preds{'_' + args.dataset_tag if args.dataset_tag else ''}.npz"
    if not preds.exists():
        raise SystemExit(f"{preds} not found; pass --preds or --dataset-tag")
    skew_min = abs(preds.stat().st_mtime - events[-1].stat().st_mtime) / 60.0
    stale = skew_min > args.max_skew_min
    if stale:
        print(f"warning: {preds.name} is {skew_min:.0f} min away from the version_{version} events file; "
              "the predictions may belong to another run")

    d = np.load(preds, allow_pickle=True)
    y, p = d["y"].astype(float), d["p"].astype(float)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc(y, p)
    scal = load_scalars(run_dir)
    n_epochs = int(max(e[-1] for e, _ in scal.values())) + 1
    best = None
    if "val_loss" in scal:
        e, v = scal["val_loss"]
        best = (int(e[np.argmin(v)]), float(v.min()))

    out = args.out or ROOT / "out" / f"classifier_run_v{version}{'_roc_bare' if args.bare else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bare:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), facecolor="#fcfcfb")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
        ax.plot(fpr, tpr, lw=2, color=BLUE)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        style(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
        print(f"version_{version}: validation AUC {auc:.3f} (68% bootstrap {lo:.3f}-{hi:.3f}) on {len(y)} events")
        print(f"wrote {out}")
        return
    when = datetime.fromtimestamp(events[-1].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    title = args.title or (f"classifier run version_{version} ({when}): {n_epochs} epochs, "
                           f"{len(y)} validation events, {int(y.sum())} NC / {int((1 - y).sum())} CC")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor="#fcfcfb",
                             gridspec_kw=dict(width_ratios=(1.0, 1.2, 1.2)))
    fig.suptitle(title, fontsize=10.5, color=INK, y=0.99)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="chance (0.50)")
    ax.plot(fpr, tpr, lw=2, color=BLUE, label=f"AUC {auc:.3f}  (68%: {lo:.2f}-{hi:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("false positive rate (CC called NC)")
    ax.set_ylabel("true positive rate (NC called NC)")
    ax.set_title(f"validation ROC, {preds.name}" + ("  [may not be this run]" if stale else ""),
                 fontsize=9, color=INK2)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False, borderaxespad=0.6)
    style(ax)

    for ax, metric, label in ((axes[1], "loss", "binary cross-entropy"), (axes[2], "acc", "accuracy")):
        for split, color in (("train", BLUE), ("val", ORANGE)):
            key = f"{split}_{metric}"
            if key in scal:
                e, v = scal[key]
                ax.plot(e + 1, v, lw=1.8, color=color, label=f"{split} ({v[-1]:.3f} at last epoch)")
        if metric == "loss" and best is not None:
            ax.axvline(best[0] + 1, color=MUTED, lw=1, ls=":")
            ax.annotate(f"min val loss {best[1]:.3f}\nepoch {best[0] + 1}", (best[0] + 1, best[1]),
                        xytext=(6, 10), textcoords="offset points", fontsize=7.5, color=INK2)
        if metric == "acc":
            ax.axhline(0.5, color=MUTED, lw=1, ls="--")
            ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} per epoch", fontsize=9, color=INK2)
        ax.legend(fontsize=7.5, frameon=False)
        style(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"version_{version}: {n_epochs} epochs; validation AUC {auc:.3f} (68% bootstrap {lo:.3f}-{hi:.3f}) "
          f"on {len(y)} events" + (f"; min val loss {best[1]:.3f} at epoch {best[0] + 1}" if best else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
