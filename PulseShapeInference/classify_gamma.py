"""NC/CC classifier from the two Gamma profile parameters (ln alpha, ln beta), true or inferred from the light.
Usage (from this directory; everything else is a constant in the settings block below)
    python classify_gamma.py --source truth --build --train
    python classify_gamma.py --source pred  --build --train
    python classify_gamma.py --source pred  --train                 # retrain on the dataset already built
    python classify_gamma.py --source pred --tag 1tev_none --match-var qtot --match-target nc --build --train
                                                                     # 1 TeV sample, CC reweighted onto the NC Q_tot histogram
    python classify_gamma.py --source pred --tag gev_evisz_s85_both_none --match-mode off --build --train
                                                                     # no energy matching at all (a sample generated flat in E_vis
                                                                     # for both classes); outputs classify_gamma_<source>_<tag>_nomatch_*
Outputs next to the script: classify_gamma_<source>_data/ (iwpc dataset), classify_gamma_<source>_meta.json,
classify_gamma_<source>_val_preds.npz (validation labels, P(NC), log10 E_vis, weights, keys), classify_gamma_<source>_all_preds.npz
(the same for every event of the prediction file, held_out = 1 for the validation half), logs in logs/classify_gamma/<source>/.

Visible energy (or the total charge Q_tot, --match-var qtot) is removed as a trivial discriminant by *weighting*
(MATCH_MODE = "weights", weights.py): every event is kept and each class is reweighted to the same log10 E_vis (log10 Q_tot)
distribution, so the classifier learns the profile shape only.  --match-target nc reweights CC onto the NC histogram and
keeps NC at 1 (flux_weighting.ipynb section 9); a Q_tot-matched build is named classify_gamma_<source>_<tag>_qtot_*.  The
physical NC/CC mix for a flux (weights.py flux weights, from flux_weighting.ipynb) enters only at evaluation, as a
flux-weighted AUC and as the prior shift P(NC | x, E) = sigmoid(logit P(NC | x) + ln pi_NC(E)/pi_CC(E)).
"""
import argparse
import json
import re
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.models.utils import basic_model_factory
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.visualisable import Visualisable

from weights import MATCH_TARGETS, load_flux_weights, log_prior_ratio, match_weights, n_eff, prior_shift

ROOT = Path(__file__).resolve().parent
PRED_IN = ROOT / "pulse_regressor_neptune_val.npz"   # Neptune prediction file (pulse_regressor.py; --tag picks pulse_regressor_<tag>_...)
LOG_DIR = ROOT / "logs" / "classify_gamma"
TAG = ""                                             # set by --tag: inputs from pulse_regressor_<tag>_neptune_val.npz, outputs *_<tag>

# settings (formerly command-line options)
MATCH_MODE = "weights"     # "weights": per-event E_vis matching weights (weights.py), every event kept;
                           # "subsample": min count per E_vis bin (the old behaviour); "off": no energy matching
MATCH_TARGET = "harmonic"  # matching target: "harmonic" (weights bounded by 2), "min" (weighted analogue of subsampling),
                           # "nc" (CC reweighted onto the NC histogram, NC kept at 1) or "cc" (the mirror); --match-target overrides
MATCH_VAR = "evis"         # variable matched between the classes: "evis" (log10 E_vis) or "qtot" (log10 Q_tot, photoelectrons
                           # summed over all modules = nhits of the prediction file); --match-var overrides
MATCH_BINS = 25            # bins of the matching (and of the prior-ratio table); 25 keeps >= 5 events per class per bin on a ~2k validation split
MATCH_MAX_WEIGHT = 10.0    # clip of the per-bin ratio for the "nc"/"cc" targets (unbounded otherwise); None: no clip
FLUX_EVAL = ("astro_g2", "atm_conv")   # flux weights of weights_1tev.parquet used at evaluation (needs `part` in the npz)
MIN_TRACKS = 0             # drop events whose label rests on fewer charged tracks (0: keep all)
N_FILES = 5                # files written at --build; the train/val split is by file (first half / second half)
EPOCHS = 10000
LR = 1e-3
BATCH_SIZE = 256
ACCELERATOR = "cpu"
SEED = 1234

FEATURES = ["ln_alpha", "ln_beta"]                 # the two inputs, from y (truth) or mu (pred)
TRUE = ["true_ln_alpha", "true_ln_beta"]           # carried along for the baselines
COV = ["S00", "S01", "S11"]                        # posterior covariance, for the note's Lambda
ENERGY, LABEL, WEIGHT = "log10_evis", "is_nc", "weight"
QTOT = "log10_qtot"                                # log10 of the photoelectrons summed over all modules (nhits), carried in the dataset
VAR_NAME = {"evis": "E_vis", "qtot": "Q_tot"}      # printed names of the matching variables
W_MATCH, KEYS = "w_match", ["part_idx", "eventID"]  # carried through the dataset (float32) next to the label
SCALARS = {"ln_alpha": (r"$$\ln\alpha$$", (-0.5, 2.5)), "ln_beta": (r"$$\ln(\beta/\mathrm{m})$$", (-1.5, 1.5))}   # axes of vis.py


def paths(source):
    """Dataset directory, meta file, prediction file and log directory of one source."""
    s = f"{source}_{TAG}" if TAG else source
    if MATCH_MODE == "off":
        s += "_nomatch"                            # an unmatched build lives next to the matched ones
    elif MATCH_VAR != "evis":
        s += f"_{MATCH_VAR}"                       # a Q_tot-matched build lives next to the E_vis-matched one
    return (ROOT / f"classify_gamma_{s}_data", ROOT / f"classify_gamma_{s}_meta.json",
            ROOT / f"classify_gamma_{s}_val_preds.npz", LOG_DIR / s)


def best_checkpoint(run_dir: Path) -> Path:
    """The checkpoint of a Lightning run directory (.../version_N): the one with the highest epoch in its name,
    whatever the prefix (best-epoch=K.ckpt from ModelCheckpoint here, epoch=K-step=S.ckpt from the default)."""
    ckpts = list(Path(run_dir).glob("checkpoints/*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint under {run_dir}")
    return max(ckpts, key=lambda p: int(m.group(1)) if (m := re.search(r"epoch=(\d+)", p.name)) else -1)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_events(pred_path: Path, source: str) -> pd.DataFrame:
    """One row per event of the prediction file: the two inputs from y (truth) or mu (pred), plus truth, Sigma, E, class."""
    d = np.load(pred_path, allow_pickle=True)
    x = d["y"] if source == "truth" else d["mu"]
    cls = d["cls"].astype(str)
    is_nc = np.array(["nc" in c.lower() for c in cls])
    if is_nc.all() or not is_nc.any():
        raise SystemExit(f"{pred_path.name} holds one class only ({set(cls)}): pulse_regressor.py must be built and trained "
                         "on both the CC and the NC directory")
    S = d["Sigma"]
    df = pd.DataFrame({"ln_alpha": x[:, 0], "ln_beta": x[:, 1], "true_ln_alpha": d["y"][:, 0], "true_ln_beta": d["y"][:, 1],
                       "S00": S[:, 0, 0], "S01": S[:, 0, 1], "S11": S[:, 1, 1],
                       ENERGY: np.log10(np.clip(d["E_vis_gev"], 1e-6, None)), "n_tracks": d["n_tracks"], "nhits": d["nhits"],
                       QTOT: np.log10(np.clip(d["nhits"], 1, None)),
                       "eventID": d["eventID"], "cls": cls, LABEL: is_nc.astype(np.float32),
                       "part": d["part"].astype(str) if "part" in d.files else ""})
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  {pred_path.name}, source {source}: {n0:,} events ({int(is_nc.sum()):,} NC, {int((~is_nc).sum()):,} CC), {len(df):,} finite")
    return df


def match_energy(df, n_bins=40, seed=0):
    """Subsample both classes to the same log10 E_vis histogram (min count per bin)."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(df[ENERGY].min(), df[ENERGY].max() + 1e-9, n_bins + 1)
    df = df.assign(_bin=np.digitize(df[ENERGY], edges) - 1)
    keep = []
    for _, g in df.groupby("_bin"):
        counts = g[LABEL].value_counts()
        if len(counts) < 2:
            continue
        for _, gl in g.groupby(LABEL):
            keep.append(gl.sample(n=int(counts.min()), random_state=int(rng.integers(1 << 31))))
    out = pd.concat(keep).drop(columns="_bin").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"  energy matching in {n_bins} log10 E bins: {len(df):,} -> {len(out):,} events ({int(out[LABEL].sum()):,} per class); "
          f"median E_vis {10 ** out[ENERGY].median():.2f} GeV")
    return out


def note_statistic(train, ev, use_sigma=True):
    """Lambda = ln N(x; m_CC, S_CC + Sigma) - ln N(x; m_NC, S_NC + Sigma), class Gaussians from the true y of `train`."""
    x = ev[FEATURES].to_numpy()
    Sig = np.stack([ev.S00, ev.S01, ev.S01, ev.S11], 1).reshape(-1, 2, 2) if use_sigma else np.zeros((len(ev), 2, 2))

    def log_normal(lab):
        t = train.loc[train[LABEL] == lab, TRUE].to_numpy()
        C = np.cov(t.T)[None] + Sig
        r = x - t.mean(0)
        return -0.5 * (np.einsum("ni,nij,nj->n", r, np.linalg.inv(C), r) + np.log(np.linalg.det(C)))

    return log_normal(0) - log_normal(1)


def baselines(df, source, seed=0):
    """Closed-form references on a random 80/20 split (validation AUC, NC = 1), with the matching weights of the events
    (so that, like the classifier, they are judged on the profile shape and not on the energy)."""
    rng = np.random.default_rng(seed)
    val = rng.random(len(df)) < 0.2
    tr, ev = df[~val & (df[WEIGHT] > 0)], df[val]        # the fits see the matched (overlap) events only
    y, w = ev[LABEL].to_numpy(), ev[WEIGHT].to_numpy()
    print(f"  baselines (random 80/20 split, {VAR_NAME[MATCH_VAR] + '-matching weights' if MATCH_MODE == 'weights' else 'unweighted'}):")
    for k in FEATURES + ["ln_alpha_beta"]:
        v = (ev.ln_alpha + ev.ln_beta).to_numpy() if k == "ln_alpha_beta" else ev[k].to_numpy()
        auc = roc_auc_score(y, v, sample_weight=w)
        print(f"    {k:>14}: {max(auc, 1 - auc):.3f}  ({'NC larger' if auc >= 0.5 else 'NC smaller'})")
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[FEATURES], tr[LABEL])
    print(f"    QDA on (ln alpha, ln beta): {roc_auc_score(y, qda.predict_proba(ev[FEATURES])[:, 1], sample_weight=w):.3f}")
    if source == "pred":
        print(f"    note's Lambda with the per-event Sigma: {roc_auc_score(y, -note_statistic(tr, ev), sample_weight=w):.3f}; with Sigma = 0: "
              f"{roc_auc_score(y, -note_statistic(tr, ev, False), sample_weight=w):.3f}")
        qda_t = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[TRUE], tr[LABEL])
        print(f"    truth ceiling on these events (QDA on the true y): {roc_auc_score(y, qda_t.predict_proba(ev[TRUE])[:, 1], sample_weight=w):.3f}")
        r = df[FEATURES].to_numpy() - df[TRUE].to_numpy()
        print(f"    prediction quality: median |mu - y| ln alpha {np.median(np.abs(r[:, 0])):.3f}, ln beta {np.median(np.abs(r[:, 1])):.3f}")
    for lab, name in ((0, "CC"), (1, "NC")):
        m = df[df[LABEL] == lab]
        print(f"    {name}: mean ln alpha {m.ln_alpha.mean():+.3f} (sd {m.ln_alpha.std():.3f}), ln beta {m.ln_beta.mean():+.3f} "
              f"(sd {m.ln_beta.std():.3f}), corr {m.ln_alpha.corr(m.ln_beta):+.2f}")


def flux_columns(df):
    """Flux weights of FLUX_EVAL joined on (cls, part, eventID), each normalised to mean 1 over the matched events
    (0 where unmatched); [] when the npz carries no `part` or the weight table is missing."""
    cols = []
    if "part" not in df or (df["part"] == "").all():
        print("  no `part` in the prediction file (re-run the pulse_regressor evaluation): no flux-weighted evaluation")
        return cols
    for flux in FLUX_EVAL:
        w = load_flux_weights(df.cls, df.part, df.eventID, flux)
        if w is None:
            print("  weights_1tev.parquet not found (run flux_weighting.ipynb): no flux-weighted evaluation")
            return cols
        ok = np.isfinite(w)
        w = np.where(ok, w, 0.0)
        if ok.sum() == 0:
            print(f"  flux {flux}: no event of the prediction file matched the weight table, skipped")
            continue
        df[f"w_{flux}"] = w / w[ok].mean()
        cols.append(f"w_{flux}")
        print(f"  flux {flux}: {ok.sum():,}/{len(df):,} events matched to the weight table, n_eff {n_eff(w):,.0f}")
    return cols


def build(source):
    data_dir, meta_file, _, _ = paths(source)
    mcol = ENERGY if MATCH_VAR == "evis" else QTOT
    print(f"building dataset: source {source}, features {FEATURES}, {VAR_NAME[MATCH_VAR]} matching: {MATCH_MODE}"
          + (f" (target {MATCH_TARGET}, {MATCH_BINS} bins)" if MATCH_MODE == "weights" else ""))
    if MATCH_VAR not in VAR_NAME or MATCH_TARGET not in MATCH_TARGETS:
        raise SystemExit(f"MATCH_VAR must be one of {tuple(VAR_NAME)} and MATCH_TARGET one of {MATCH_TARGETS}, got {MATCH_VAR!r}, {MATCH_TARGET!r}")
    if MATCH_MODE == "subsample" and MATCH_VAR != "evis":
        raise SystemExit("MATCH_MODE 'subsample' matches E_vis only; use MATCH_MODE 'weights' for Q_tot")
    df = load_events(PRED_IN, source)
    if MIN_TRACKS > 0:
        n0 = len(df)
        df = df[df.n_tracks >= MIN_TRACKS]
        print(f"  n_tracks >= {MIN_TRACKS}: {n0:,} -> {len(df):,} events (label quality cut)")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    edges = np.linspace(df[mcol].min(), df[mcol].max() + 1e-9, MATCH_BINS + 1)
    if MATCH_MODE == "subsample":
        df = match_energy(df, n_bins=MATCH_BINS, seed=SEED)
        df[W_MATCH] = 1.0
    elif MATCH_MODE == "weights":
        clip = MATCH_MAX_WEIGHT if MATCH_TARGET in ("nc", "cc") else None
        df[W_MATCH], edges = match_weights(df[mcol], df[LABEL] > 0.5, edges=edges, target=MATCH_TARGET, max_weight=clip)
        for lab, name in ((0, "CC"), (1, "NC")):
            w = df.loc[df[LABEL] == lab, W_MATCH].to_numpy()
            print(f"  {VAR_NAME[MATCH_VAR]} matching weights (target {MATCH_TARGET}), {name}: n_eff {n_eff(w):,.0f} of {len(w):,} events, "
                  f"{int((w == 0).sum()):,} with weight 0 (no {'CC' if lab else 'NC'} at that {VAR_NAME[MATCH_VAR]}), max weight {w.max():.2f}")
    elif MATCH_MODE == "off":
        df[W_MATCH] = 1.0
    else:
        raise SystemExit(f"MATCH_MODE must be weights, subsample or off, got {MATCH_MODE!r}")
    # class-balanced training weights: each class sums to N/4, so the mean weight is 1/2 as before
    df[WEIGHT] = df[W_MATCH] / df.groupby(LABEL)[W_MATCH].transform("sum") * len(df) / 4.0
    flux_cols = flux_columns(df)
    parts = sorted(df["part"].unique().tolist()) if "part" in df else [""]
    df["part_idx"] = df["part"].map({p: i for i, p in enumerate(parts)}).astype(float) if "part" in df else 0.0
    cls_names = {int(lab): str(df.loc[df[LABEL] == lab, "cls"].iloc[0]) for lab in (0, 1)}
    extra_cols = [W_MATCH, QTOT] + flux_cols + KEYS
    print(f"  {len(df):,} events, NC fraction {df[LABEL].mean():.3f}")
    baselines(df, source, seed=SEED)
    with PandasDirDataModuleBuilder(data_dir, force=True, file_size=int(max(100, len(df) // N_FILES))) as builder:
        builder.write(df[FEATURES + [LABEL, ENERGY, WEIGHT] + TRUE + COV + extra_cols].astype(np.float32))
    meta_file.write_text(json.dumps(dict(source=source, pred=str(PRED_IN), features=FEATURES, match_mode=MATCH_MODE,
                                         match_var=MATCH_VAR, match_target=MATCH_TARGET, match_bins=MATCH_BINS,
                                         match_max_weight=MATCH_MAX_WEIGHT, match_edges=[float(x) for x in edges],
                                         n_eff={name: float(n_eff(df.loc[df[LABEL] == lab, W_MATCH])) for lab, name in ((0, "CC"), (1, "NC"))},
                                         flux_eval=[c[2:] for c in flux_cols], extra_cols=extra_cols, parts=parts, cls_names=cls_names,
                                         min_tracks=MIN_TRACKS, n_events=len(df),
                                         center=[float(df[k].median()) for k in FEATURES]), indent=1))
    print(f"wrote {len(df):,} rows to {data_dir.name} (~{N_FILES} files) and {meta_file.name}")


# ---------------------------------------------------------------------------
# Model (iwpc MLP, Visualisable for vis.py)
# ---------------------------------------------------------------------------

def phi(x):
    return x["phi"]


def p_nc(x):
    return 1 / (1 + np.exp(-x["phi"]))


def p_cc(x):
    return 1 - p_nc(x)


class GammaClassifier(L.LightningModule, Visualisable):
    """Binary NC/CC classifier over (ln alpha, ln beta): phi(x) = logit P(NC | x)."""

    def __init__(self, features, center, lr: float = 1e-3, source: str = "truth"):
        super().__init__()
        self.lr, self.features, self.center, self.source = lr, list(features), list(center), source
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.model = basic_model_factory(TrivialEncoding(len(self.features)), TrivialEncoding(1), hidden_layer_sizes=(64, 64, 32))
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y, w = batch
        out = self.forward(x)
        y = y[:, :1].reshape_as(out)                  # column 0 is the label; the rest is carried for the evaluation
        w = w.reshape_as(out)
        loss = (self.criterion(out, y) * w).mean()
        acc = ((torch.sigmoid(out) > 0.5).float() == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches,
                                                    pct_start=0.05, anneal_strategy="cos")
        return [opt], [dict(scheduler=sched, interval="step")]

    # -- Visualisable ------------------------------------------------------
    def get_input_scalars(self):
        return [Scalar(k, SCALARS[k][0], bins=np.linspace(*SCALARS[k][1], 100)) for k in self.features]

    def get_output_scalars(self):
        return [ScalarFunction(phi, "phi", r"$$\phi = \mathrm{logit}\,P(\mathrm{NC}\,|\,x)$$"),
                ScalarFunction(p_nc, "p_NC", r"$$P(\mathrm{NC}\,|\,x)$$", bins=np.linspace(0, 1, 100)),
                ScalarFunction(p_cc, "p_CC", r"$$P(\mathrm{CC}\,|\,x)$$", bins=np.linspace(0, 1, 100))]

    def evaluate_for_visualiser(self, z):
        self.eval()
        self.cpu()
        with torch.no_grad():
            return {"phi": self.forward(torch.tensor(np.asarray(z), dtype=torch.float32)).cpu().numpy()[:, 0]}

    @property
    def center_point(self):
        return self.center


# ---------------------------------------------------------------------------

def train(source):
    data_dir, meta_file, pred_file, log_dir = paths(source)
    if not meta_file.exists():
        raise SystemExit(f"{meta_file.name} not found: run --source {source} --build first")
    meta = json.loads(meta_file.read_text())
    ycols = [LABEL, ENERGY] + TRUE + COV + meta.get("extra_cols", [])
    dm = PandasDirDataModule(dataset_dir=str(data_dir), feature_spec=[FEATURES, ycols], weight_col=[WEIGHT],
                             dataloader_kwargs={"num_workers": 0, "batch_size": BATCH_SIZE})
    model = GammaClassifier(FEATURES, meta["center"], lr=LR, source=source)
    mode = meta.get("match_mode", "subsample" if meta.get("match_energy") else "off")
    print(f"training on {FEATURES} from the {source} ({meta['n_events']:,} events, {VAR_NAME[meta.get('match_var', 'evis')]} matching: {mode}"
          + (f", target {meta['match_target']}" if "match_target" in meta and mode != "off" else "") + f", {EPOCHS} epochs)")
    best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}")   # lowest validation loss
    trainer = L.Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, log_every_n_steps=5,
                        logger=TensorBoardLogger(save_dir=str(log_dir)), num_sanity_val_steps=0, enable_progress_bar=False,
                        callbacks=[best])
    trainer.fit(model, datamodule=dm)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"last epoch: train loss {metrics.get('train_loss', float('nan')):.4f}, val loss {metrics.get('val_loss', float('nan')):.4f}; "
          f"best val loss {float(best.best_model_score):.4f} at {Path(best.best_model_path).stem} (evaluated below, kept in {log_dir.name}/)")
    model = GammaClassifier.load_from_checkpoint(best.best_model_path)
    evaluate(model, dm, source, pred_file, meta)


def evaluate(model, dm, source, pred_file, meta=None):
    """Validation AUC: with the E_vis-matching weights (shape only), per energy bin, and flux-weighted on the physical class
    mix with the prior-shift calibration check; closed-form baselines on the same validation events."""
    meta = meta or {}
    ycols = [LABEL, ENERGY] + TRUE + COV + meta.get("extra_cols", [])
    model.eval()
    model.cpu()
    xs, ys, ps, extra = [], [], [], []
    with torch.no_grad():
        for x, y, _ in dm.val_dataloader():
            xs.append(x.numpy()); ps.append(torch.sigmoid(model(x)).numpy()[:, 0]); ys.append(y[:, 0].numpy()); extra.append(y[:, 1:].numpy())
    if not ys:
        print("no validation data (need n_files >= 2 at --build)")
        return
    y, p, extra = np.concatenate(ys), np.concatenate(ps), np.concatenate(extra)
    col = {name: extra[:, i] for i, name in enumerate(ycols[1:])}
    e = col[ENERGY]
    mvar = meta.get("match_var", "evis") if QTOT in col else "evis"
    mcol = QTOT if mvar == "qtot" else ENERGY                # the matched variable: prior ratio and shift live in its bins
    v = col[mcol]
    wm = col.get(W_MATCH, np.ones(len(y)))
    matched = meta.get("match_mode", "weights") != "off"
    print(f"validation ({source}): {len(y):,} events, AUC = {roc_auc_score(y, p):.3f} unweighted"
          + (f", {roc_auc_score(y, p, sample_weight=wm):.3f} with the {VAR_NAME[mvar]}-matching weights (n_eff {n_eff(wm):,.0f})" if matched
             else " (no energy matching: the classes enter as generated)")
          + f"; accuracy = {np.mean((p > 0.5) == (y > 0.5)):.3f}  (NC = 1)")
    for lo, hi in zip(*(lambda b: (b[:-1], b[1:]))(np.linspace(np.percentile(e, 1), np.percentile(e, 99), 6))):
        m = (e >= lo) & (e < hi)
        if m.sum() >= 40 and 0 < y[m].mean() < 1:
            print(f"  E_vis {10 ** lo:7.1f}-{10 ** hi:7.1f} GeV: {m.sum():5d} events, AUC {roc_auc_score(y[m], p[m]):.3f}")
    if mvar == "qtot":
        for lo, hi in zip(*(lambda b: (b[:-1], b[1:]))(np.linspace(np.percentile(v, 1), np.percentile(v, 99), 6))):
            m = (v >= lo) & (v < hi)
            if m.sum() >= 40 and 0 < y[m].mean() < 1:
                print(f"  Q_tot {10 ** lo:7.0f}-{10 ** hi:7.0f} p.e.: {m.sum():5d} events, AUC {roc_auc_score(y[m], p[m]):.3f}, "
                      f"{roc_auc_score(y[m], p[m], sample_weight=wm[m]):.3f} weighted")
    val_files = set(map(str, dm.validation_files))
    tr = pd.concat([pd.read_pickle(f) for f in dm.all_files if str(f) not in val_files], ignore_index=True)
    val_df = pd.DataFrame(np.c_[np.concatenate(xs), y[:, None], extra], columns=FEATURES + ycols)
    fit = tr[tr[W_MATCH] > 0] if W_MATCH in tr else tr
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(fit[FEATURES], fit[LABEL])
    line = f"  same events ({'matching weights' if matched else 'unweighted'}): QDA on (ln alpha, ln beta) {roc_auc_score(y, qda.predict_proba(val_df[FEATURES])[:, 1], sample_weight=wm):.3f}"
    if source == "pred":
        qda_t = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(fit[TRUE], fit[LABEL])
        line += (f"; note's Lambda with Sigma {roc_auc_score(y, -note_statistic(fit, val_df), sample_weight=wm):.3f}; "
                 f"truth ceiling (QDA on the true y) {roc_auc_score(y, qda_t.predict_proba(val_df[TRUE])[:, 1], sample_weight=wm):.3f}")
    print(line)
    # physical class mix: flux-weighted AUC and the prior shift, prior ratio from the whole dataset (train + validation)
    out = dict(y=y, p=p, log10_evis=e, x=np.concatenate(xs), features=np.array(FEATURES), source=source, w_match=wm,
               eventID=col.get("eventID", np.full(len(y), -1.0)).astype(int),
               cls=np.array([meta.get("cls_names", {}).get(str(int(k)), "nc" if k else "cc") for k in y]))
    if "part_idx" in col and meta.get("parts"):
        out["part"] = np.array(meta["parts"], dtype=str)[col["part_idx"].astype(int)]
    edges = np.array(meta.get("match_edges", np.linspace(v.min(), v.max() + 1e-9, MATCH_BINS + 1)))
    allev = pd.concat([tr[ycols], val_df[ycols]], ignore_index=True)
    bins = np.linspace(np.percentile(e, 1), np.percentile(e, 99), 5)
    for flux in meta.get("flux_eval", []):
        wf = col[f"w_{flux}"]
        if wf.sum() <= 0 or y[wf > 0].min() == y[wf > 0].max():       # no weight, or one class only
            continue
        _, lpr = log_prior_ratio(allev[mcol], allev[LABEL] > 0.5, allev[f"w_{flux}"], edges)
        ps = prior_shift(p, v, lpr)
        print(f"  flux {flux} (physical mix, n_eff {n_eff(wf):,.0f}): AUC {roc_auc_score(y, p, sample_weight=wf):.3f}; "
              f"NC fraction {np.average(y, weights=wf):.3f}; sum P(NC)/true NC = {np.sum(wf * p) / np.sum(wf * y):.3f} as trained, "
              f"{np.sum(wf * ps) / np.sum(wf * y):.3f} with the prior shift")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (e >= lo) & (e < hi) & (wf > 0)
            if m.sum() >= 40 and np.sum(wf[m] * y[m]) > 0:
                print(f"      E_vis {10 ** lo:7.1f}-{10 ** hi:7.1f} GeV: true NC fraction {np.average(y[m], weights=wf[m]):.3f}, "
                      f"sum P(NC)/true NC {np.sum(wf[m] * p[m]) / np.sum(wf[m] * y[m]):.2f} -> {np.sum(wf[m] * ps[m]) / np.sum(wf[m] * y[m]):.2f} shifted")
        out[f"w_{flux}"], out[f"p_shift_{flux}"] = wf, ps
    np.savez(pred_file, **out)
    print(f"wrote validation predictions to {pred_file.name}")
    # the same for EVERY event of the prediction file (the classifier's training half included, flagged held_out = 0):
    # Pynu's response tables want all ~2k regressor-validation events, not only the classifier's held-out half
    allev = pd.concat([tr.assign(held_out=0.0), val_df.assign(held_out=1.0)], ignore_index=True)
    with torch.no_grad():
        p_all = torch.sigmoid(model(torch.tensor(allev[FEATURES].to_numpy(np.float32))))[:, 0].numpy()
    y_all = allev[LABEL].to_numpy()
    out_all = dict(y=y_all, p=p_all, log10_evis=allev[ENERGY].to_numpy(), x=allev[FEATURES].to_numpy(np.float32), features=np.array(FEATURES),
                   source=source, held_out=allev["held_out"].to_numpy(), w_match=allev[W_MATCH].to_numpy() if W_MATCH in allev else np.ones(len(allev)),
                   eventID=allev["eventID"].to_numpy().astype(int) if "eventID" in allev else np.full(len(allev), -1),
                   cls=np.array([meta.get("cls_names", {}).get(str(int(k)), "nc" if k else "cc") for k in y_all]))
    if "part_idx" in allev and meta.get("parts"):
        out_all["part"] = np.array(meta["parts"], dtype=str)[allev["part_idx"].to_numpy().astype(int)]
    all_file = pred_file.with_name(pred_file.name.replace("_val_preds", "_all_preds"))
    np.savez(all_file, **out_all)
    print(f"wrote predictions for all {len(allev):,} events ({int(allev.held_out.sum()):,} held out, AUC all {roc_auc_score(y_all, p_all):.3f}) to {all_file.name}")


def main():
    global TAG, PRED_IN, MATCH_VAR, MATCH_TARGET, MATCH_MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("truth", "pred"), required=True, help="which (ln alpha, ln beta) to classify on: y or mu")
    ap.add_argument("--build", action="store_true", help="write the dataset")
    ap.add_argument("--train", action="store_true", help="train the classifier")
    ap.add_argument("--tag", default="", help="dataset tag of pulse_regressor.py (e.g. 1tev): read pulse_regressor_<tag>_neptune_val.npz, "
                                              "write classify_gamma_<source>_<tag>_*")
    ap.add_argument("--match-var", choices=tuple(VAR_NAME), default=MATCH_VAR,
                    help=f"variable matched between the classes (default {MATCH_VAR}); qtot adds _qtot to the output names")
    ap.add_argument("--match-target", choices=MATCH_TARGETS, default=MATCH_TARGET,
                    help=f"matching target (default {MATCH_TARGET}); nc = CC reweighted onto the NC histogram, NC kept at 1")
    ap.add_argument("--match-mode", choices=("weights", "subsample", "off"), default=MATCH_MODE,
                    help=f"how the classes are matched in the --match-var variable (default {MATCH_MODE}); off = no matching, the two "
                         "inferred parameters alone, outputs *_nomatch_*")
    args = ap.parse_args()
    TAG, MATCH_VAR, MATCH_TARGET, MATCH_MODE = args.tag, args.match_var, args.match_target, args.match_mode
    if TAG:
        PRED_IN = ROOT / f"pulse_regressor_{TAG}_neptune_val.npz"
    if not (args.build or args.train):
        ap.error("pass --build and/or --train")
    if args.build:
        build(args.source)
    if args.train:
        train(args.source)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
