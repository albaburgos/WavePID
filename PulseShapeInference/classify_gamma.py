"""NC/CC classifier from the two Gamma profile parameters (ln alpha, ln beta), true or inferred from the light.
Usage (from this directory; everything else is a constant in the settings block below)
    python classify_gamma.py --source truth --build --train
    python classify_gamma.py --source pred  --build --train
    python classify_gamma.py --source pred  --train                 # retrain on the dataset already built
Outputs next to the script: classify_gamma_<source>_data/ (iwpc dataset), classify_gamma_<source>_meta.json,
classify_gamma_<source>_val_preds.npz (validation labels, P(NC), log10 E_vis), logs in logs/classify_gamma/<source>/.
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

ROOT = Path(__file__).resolve().parent
PRED_IN = ROOT / "pulse_regressor_neptune_val.npz"   # Neptune prediction file (pulse_regressor.py --neptune / --evaluate)
LOG_DIR = ROOT / "logs" / "classify_gamma"

# settings (formerly command-line options)
MATCH_ENERGY = True        # subsample both classes to the same E_vis histogram
MIN_TRACKS = 0             # drop events whose label rests on fewer charged tracks (0: keep all)
N_FILES = 5                # files written at --build; the train/val split is by file (first half / second half)
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 256
ACCELERATOR = "cpu"
SEED = 1234

FEATURES = ["ln_alpha", "ln_beta"]                 # the two inputs, from y (truth) or mu (pred)
TRUE = ["true_ln_alpha", "true_ln_beta"]           # carried along for the baselines
COV = ["S00", "S01", "S11"]                        # posterior covariance, for the note's Lambda
ENERGY, LABEL, WEIGHT = "log10_evis", "is_nc", "weight"
SCALARS = {"ln_alpha": (r"$$\ln\alpha$$", (-3,3)), "ln_beta": (r"$$\ln(\beta/\mathrm{m})$$", (0, 0.69))}


def paths(source):
    """Dataset directory, meta file, prediction file and log directory of one source."""
    return (ROOT / f"classify_gamma_{source}_data", ROOT / f"classify_gamma_{source}_meta.json",
            ROOT / f"classify_gamma_{source}_val_preds.npz", LOG_DIR / source)


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
                       "eventID": d["eventID"], "cls": cls, LABEL: is_nc.astype(np.float32)})
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
    """Closed-form references on a random 80/20 split (validation AUC, NC = 1)."""
    rng = np.random.default_rng(seed)
    val = rng.random(len(df)) < 0.2
    tr, ev = df[~val], df[val]
    y = ev[LABEL].to_numpy()
    print("  baselines (random 80/20 split):")
    for k in FEATURES + ["ln_alpha_beta"]:
        v = (ev.ln_alpha + ev.ln_beta).to_numpy() if k == "ln_alpha_beta" else ev[k].to_numpy()
        auc = roc_auc_score(y, v)
        print(f"    {k:>14}: {max(auc, 1 - auc):.3f}  ({'NC larger' if auc >= 0.5 else 'NC smaller'})")
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[FEATURES], tr[LABEL])
    print(f"    QDA on (ln alpha, ln beta): {roc_auc_score(y, qda.predict_proba(ev[FEATURES])[:, 1]):.3f}")
    if source == "pred":
        print(f"    note's Lambda with the per-event Sigma: {roc_auc_score(y, -note_statistic(tr, ev)):.3f}; with Sigma = 0: "
              f"{roc_auc_score(y, -note_statistic(tr, ev, False)):.3f}")
        qda_t = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[TRUE], tr[LABEL])
        print(f"    truth ceiling on these events (QDA on the true y): {roc_auc_score(y, qda_t.predict_proba(ev[TRUE])[:, 1]):.3f}")
        r = df[FEATURES].to_numpy() - df[TRUE].to_numpy()
        print(f"    prediction quality: median |mu - y| ln alpha {np.median(np.abs(r[:, 0])):.3f}, ln beta {np.median(np.abs(r[:, 1])):.3f}")
    for lab, name in ((0, "CC"), (1, "NC")):
        m = df[df[LABEL] == lab]
        print(f"    {name}: mean ln alpha {m.ln_alpha.mean():+.3f} (sd {m.ln_alpha.std():.3f}), ln beta {m.ln_beta.mean():+.3f} "
              f"(sd {m.ln_beta.std():.3f}), corr {m.ln_alpha.corr(m.ln_beta):+.2f}")


def build(source):
    data_dir, meta_file, _, _ = paths(source)
    print(f"building dataset: source {source}, features {FEATURES}, energy matching {'on' if MATCH_ENERGY else 'off'}")
    df = load_events(PRED_IN, source)
    if MIN_TRACKS > 0:
        n0 = len(df)
        df = df[df.n_tracks >= MIN_TRACKS]
        print(f"  n_tracks >= {MIN_TRACKS}: {n0:,} -> {len(df):,} events (label quality cut)")
    df = match_energy(df, seed=SEED) if MATCH_ENERGY else df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df[WEIGHT] = 1.0
    df[WEIGHT] = df[WEIGHT] / df.groupby(LABEL)[WEIGHT].transform("mean") / 2.0        # class-balanced weights
    print(f"  {len(df):,} events, NC fraction {df[LABEL].mean():.3f}")
    baselines(df, source, seed=SEED)
    with PandasDirDataModuleBuilder(data_dir, force=True, file_size=int(max(100, len(df) // N_FILES))) as builder:
        builder.write(df[FEATURES + [LABEL, ENERGY, WEIGHT] + TRUE + COV].astype(np.float32))
    meta_file.write_text(json.dumps(dict(source=source, pred=str(PRED_IN), features=FEATURES, match_energy=MATCH_ENERGY,
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
    dm = PandasDirDataModule(dataset_dir=str(data_dir), feature_spec=[FEATURES, [LABEL, ENERGY] + TRUE + COV], weight_col=[WEIGHT],
                             dataloader_kwargs={"num_workers": 0, "batch_size": BATCH_SIZE})
    model = GammaClassifier(FEATURES, meta["center"], lr=LR, source=source)
    print(f"training on {FEATURES} from the {source} ({meta['n_events']:,} events, energy matching {'on' if meta['match_energy'] else 'off'}, "
          f"{EPOCHS} epochs)")
    best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}")   # lowest validation loss
    trainer = L.Trainer(accelerator=ACCELERATOR, max_epochs=EPOCHS, log_every_n_steps=5,
                        logger=TensorBoardLogger(save_dir=str(log_dir)), num_sanity_val_steps=0, enable_progress_bar=False,
                        callbacks=[best])
    trainer.fit(model, datamodule=dm)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"last epoch: train loss {metrics.get('train_loss', float('nan')):.4f}, val loss {metrics.get('val_loss', float('nan')):.4f}; "
          f"best val loss {float(best.best_model_score):.4f} at {Path(best.best_model_path).stem} (evaluated below, kept in {log_dir.name}/)")
    model = GammaClassifier.load_from_checkpoint(best.best_model_path)
    evaluate(model, dm, source, pred_file)


def evaluate(model, dm, source, pred_file):
    """Validation AUC (overall, per energy bin) and the closed-form baselines on the same validation events."""
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
    e = extra[:, 0]
    print(f"validation ({source}): {len(y):,} events, AUC = {roc_auc_score(y, p):.3f}, accuracy = {np.mean((p > 0.5) == (y > 0.5)):.3f}  (NC = 1)")
    for lo, hi in ((-1, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)):
        m = (e >= lo) & (e < hi)
        if m.sum() >= 40 and 0 < y[m].mean() < 1:
            print(f"  E_vis {10 ** lo:6.2f}-{10 ** hi:6.2f} GeV: {m.sum():5d} events, AUC {roc_auc_score(y[m], p[m]):.3f}")
    val_files = set(map(str, dm.validation_files))
    tr = pd.concat([pd.read_pickle(f) for f in dm.all_files if str(f) not in val_files], ignore_index=True)
    val_df = pd.DataFrame(np.c_[np.concatenate(xs), y[:, None], extra], columns=FEATURES + [LABEL, ENERGY] + TRUE + COV)
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[FEATURES], tr[LABEL])
    line = f"  same events: QDA on (ln alpha, ln beta) {roc_auc_score(y, qda.predict_proba(val_df[FEATURES])[:, 1]):.3f}"
    if source == "pred":
        qda_t = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(tr[TRUE], tr[LABEL])
        line += (f"; note's Lambda with Sigma {roc_auc_score(y, -note_statistic(tr, val_df)):.3f}; "
                 f"truth ceiling (QDA on the true y) {roc_auc_score(y, qda_t.predict_proba(val_df[TRUE])[:, 1]):.3f}")
    print(line)
    np.savez(pred_file, y=y, p=p, log10_evis=e, x=np.concatenate(xs), features=np.array(FEATURES), source=source)
    print(f"wrote validation predictions to {pred_file.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("truth", "pred"), required=True, help="which (ln alpha, ln beta) to classify on: y or mu")
    ap.add_argument("--build", action="store_true", help="write the dataset")
    ap.add_argument("--train", action="store_true", help="train the classifier")
    args = ap.parse_args()
    if not (args.build or args.train):
        ap.error("pass --build and/or --train")
    if args.build:
        build(args.source)
    if args.train:
        train(args.source)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
