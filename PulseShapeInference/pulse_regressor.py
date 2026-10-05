"""Solution A: the longitudinal profile (ln a, ln b) of a cascade from its DOM pulses, with Neptune.

Data: OMSim WavePID multi-module events (../cluster/wavepid_multidom_array.sbatch), per part
    part_NNNN_hits.root           PhotonHits (eventID, moduleID, hitTime [ns]); Events (eventID, primaryEnergy [GeV],
                                  vx vy vz [m], nu_dx nu_dy nu_dz); the interaction is at t = 0
    part_NNNN_doms.json           module positions [{"moduleID", "x", "y", "z"}] [m]
    part_NNNN_profile_params.csv  labels from `gamma_functions.py --all-events`: shower axis ux uy uz and the Gamma fit
                                  a_mle b_mle (a_l1 b_l1) of the charged-track-length depth profile of the event

Problem (../Claude outputs/nc_cc_profile_inference.pdf, Sec. 2 and 3.1)
    known      the vertex v (and t_v = 0)
    target     y = (ln a, ln b) of   g(s; a, b) = s^(a-1) e^(-s/b) / (Gamma(a) b^a),   s = depth along the axis [m]
    nuisance   the shower axis u (every event is randomly rotated) and the energy E, which is not fixed and is not
               given to the network: it only sees the light itself.  known_energy appends ln E to every token
               (the note's known kappa = (E, v, t_v)); known_axis appends cos psi_d = (r_d - v).u / |r_d - v|, the
               angle of each module to the shower axis (the note's "direction also known" case), after which only
               the profile itself is unknown.  Both are invariant under the rotation augmentation below.
    data       D = per module d its photoelectron time profile (T_BIN_NS bins, last bin = overflow) and summaries

Model
    tokens     one per hit module:  xi_d = (r_d - v [km], t_first [us]),
               f_d = ( ln(1 + n_d), t_first, t10, t50, t90, std_t, t_res = t_first - n_g |r_d - v| / c, |r_d - v|, n_dk / n_d )
    encoder    NeptuneModel: point tokens -> transformer with 4-D rotary position encoding -> attention pool, then
               the pooled vector ++ [ln(1 + #tokens), ln(1 + Q_tot)]  (the total light survives the pooling)
    head       mu(D) in R^2 and a lower-triangular L(D) with L_ii = softplus(.) + eps;  Sigma = L L^T
    loss       -ln N(y; mu, Sigma) = 1/2 |L^-1 (y - mu)|^2 + ln L_11 + ln L_22 + ln 2 pi, averaged over the events.
               Its population minimiser is mu = E[y | D], Sigma = Cov[y | D] whatever the posterior's shape, so one
               forward pass returns the posterior mean and covariance of (ln a, ln b).
    training   AdamW with weight decay, one-cycle learning rate, gradient clipping; the training events are rotated
               at random about the vertex on the fly (isotropic ice: a rotated detector sees the same a, b), which
               removes the overfitting seen without it (augment = False); the epoch with the lowest validation NLL is kept.

Run on 4316 NC events (28 parts, 27-module grid, 3453 train / 863 validation), 40 epochs, 2026-09-28: validation NLL
2.01 vs 2.84 for the prior; median errors 0.23 (ln a), 0.29 (ln b), 0.18 (ln ab) vs 0.30, 0.31 for the prior;
coverage 0.70 / 0.93; still improving at the last epoch.  Without augmentation the best epoch was the 4th.

Usage (from this directory; inputs are the OMSim part directories, all outputs stay here).  There is no command line:
edit the `Settings` defaults below and run `python pulse_regressor.py`, or call `main(Settings(...))` from Python.
    build = [Path("../out/nue_cc_multidom"), Path("../out/nue_nc_multidom")]   # -> pulse_regressor_truth.parquet (+ _meta.json)
    train = True, epochs = 100                                                 # -> pulse_regressor_neptune_val.npz
    train = True, known_energy = True                                          # vertex and energy known
    train = True, known_energy = True, known_axis = True                       # vertex, energy and axis known
    evaluate = Path("logs/neptune/version_N/checkpoints/best-epoch=K.ckpt")   # summary of a saved checkpoint, no training
The build shuffles the events once and marks a fixed 20 % validation split (`is_val`) shared by every method.
Logs and checkpoints: logs/neptune/version_N.  Solution B (profile-likelihood fit with an analytic photon
kernel) is profile_likelihood.py in this directory and works on the same dataset.
"""
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import uproot
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parent                  # PulseShapeInference/: scripts and every output
NEPTUNE_SRC = next((p / "neptune" for p in ROOT.parents if (p / "neptune" / "neptune" / "__init__.py").exists()),
                   Path.home() / "IceCube" / "neptune")  # the neptune source checkout (not pip-installed)
if str(NEPTUNE_SRC) not in sys.path:
    sys.path.insert(0, str(NEPTUNE_SRC))
from neptune import NeptuneModel                        # noqa: E402

META_FILE = ROOT / "pulse_regressor_meta.json"
TRUTH_FILE = ROOT / "pulse_regressor_truth.parquet"
PRED_FILE = ROOT / "pulse_regressor_neptune_val.npz"
LOG_DIR = ROOT / "logs"

TARGETS = ["ln_a", "ln_b"]
VERTEX = ["vx", "vy", "vz"]
AXIS = ["ux", "uy", "uz"]
T_BIN_NS, T_NBINS = 4.0, 48                        
EDGES = np.arange(0.0, (T_NBINS + 1) * T_BIN_NS, T_BIN_NS)
HIST = [f"h_{i:02d}" for i in range(T_NBINS)]
SUMMARY = ["nhits", "t_first", "t10", "t50", "t90", "std_t"]
MODULE_FEATURES = SUMMARY + HIST
N_MODULES, FEATURES = 1, []
C_NS, N_GROUP = 0.299792458, 1.35                       
VAL_FRACTION = 0.2


# ----------------------------------------------------------------------------------------------------------
# settings: edit the defaults here (there is no command line) or pass a Settings to main() from Python
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Settings:
    build: list = field(default_factory=list)   # OMSim part directories (part_*_hits.root + _doms.json + _profile_params.csv)
                                                # or hits files -> dataset, e.g. [ROOT.parent / "out" / "nue_nc_multidom"]; [] skips
    target_fit: str = "mle"                     # which Gamma fit of the tracks is the label: "mle" or "l1"
    min_hits: int = 5                           # drop events with fewer hits in all modules
    train: bool = False                         # train the Neptune posterior q(ln a, ln b | hits)
    evaluate: Path = None                       # checkpoint to evaluate (after training, if both); None skips
    known_energy: bool = False                  # append ln E_vis to the inputs (the note's known-E case)
    known_axis: bool = False                    # append cos(angle between module direction and true shower axis) to every token
    augment: bool = True                        # random rotations of the training events about the vertex
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    token_dim: int = 64                         # Neptune token dimension (a multiple of 4 * 8)
    layers: int = 3                             # Neptune transformer layers
    accelerator: str = "cpu"
    seed: int = 1234                            # build shuffle, torch, and the rotation augmentation


def set_features(n_modules: int):
    """Flat feature names m<k>_<summary / histogram bin> for n modules, k in moduleID order."""
    global N_MODULES, FEATURES
    N_MODULES = int(n_modules)
    FEATURES = [f"m{k}_{f}" for k in range(N_MODULES) for f in MODULE_FEATURES]


# ----------------------------------------------------------------------------------------------------------
# 1. dataset from the OMSim files
# ----------------------------------------------------------------------------------------------------------
def module_features(t, prefix):
    """Pulse summary and arrival-time histogram (counts) of one module; all zero when it saw nothing."""
    t = np.sort(np.asarray(t, float))
    if len(t) == 0:
        return {prefix + f: 0.0 for f in MODULE_FEATURES}
    h, _ = np.histogram(np.clip(t, 0.0, EDGES[-1] - 1e-6), bins=EDGES)
    q = np.quantile(t, [0.1, 0.5, 0.9])
    row = dict(nhits=float(len(t)), t_first=t[0], t10=q[0], t50=q[1], t90=q[2], std_t=t.std())
    row.update(zip(HIST, h.astype(float)))
    return {prefix + k: v for k, v in row.items()}


def read_part(hits_root: Path):
    """Module positions, per-event truth, hit times per (event, module) and the profile labels of one part."""
    stem = str(hits_root)[: -len("_hits.root")]
    doms = json.loads(Path(stem + "_doms.json").read_text())
    positions = {int(d["moduleID"]): (float(d["x"]), float(d["y"]), float(d["z"])) for d in doms}
    params_csv = Path(stem + "_profile_params.csv")
    params = pd.read_csv(params_csv).set_index("eventID") if params_csv.exists() else None
    f = uproot.open(hits_root)
    try:
        ev = f["Events"].arrays(["eventID", "primaryEnergy"] + VERTEX + ["nu_dx", "nu_dy", "nu_dz"], library="np")
    except uproot.KeyInFileError as e:
        raise SystemExit(f"{hits_root}: {e}; the file needs the --random_vertex/--random_rotation Events branches")
    events = pd.DataFrame(ev).set_index("eventID")
    hits = f["PhotonHits"].arrays(["eventID", "moduleID", "hitTime"], library="np")
    order = np.lexsort((hits["moduleID"], hits["eventID"]))
    e, m, t = (hits[k][order] for k in ("eventID", "moduleID", "hitTime"))
    starts = np.flatnonzero(np.r_[True, (e[1:] != e[:-1]) | (m[1:] != m[:-1])])
    groups = {(int(e[i0]), int(m[i0])): t[i0:i1] for i0, i1 in zip(starts, np.r_[starts[1:], len(e)])}
    return positions, events, groups, params


def hits_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(p.glob("part_*_hits.root"))
            if not found:
                raise SystemExit(f"no part_*_hits.root in {p}")
            files += found
        elif p.is_file() and p.name.endswith("_hits.root"):
            files.append(p)
        else:
            raise SystemExit(f"{p}: not a directory of parts nor a *_hits.root file")
    return files


def build_dataframe(files, target_fit="mle", min_hits=5):
    """One row per event: per-module pulse features, vertex, axis, labels ln a, ln b and bookkeeping columns."""
    rows, positions = [], None
    for hits_root in files:
        pos, events, groups, params = read_part(hits_root)
        if positions is None:
            positions = pos
            set_features(len(positions))
        elif pos != positions:
            raise SystemExit(f"{hits_root.name}: module layout differs from {files[0].name}")
        if params is None:
            print(f"  {hits_root.name}: no _profile_params.csv (run gamma_functions.py --all-events), skipped")
            continue
        mids, n0 = sorted(positions), len(rows)
        for eid, ev in events.iterrows():
            if eid not in params.index:
                continue
            per_module = [groups.get((int(eid), k), np.empty(0)) for k in mids]
            n_tot = sum(len(t) for t in per_module)
            if n_tot < min_hits:
                continue
            row = {}
            for j, t in enumerate(per_module):
                row.update(module_features(t, f"m{j}_"))
            pr = params.loc[eid]
            row.update(ln_a=np.log(pr[f"a_{target_fit}"]), ln_b=np.log(pr[f"b_{target_fit}"]),
                       a_mle=pr["a_mle"], b_mle=pr["b_mle"], a_l1=pr["a_l1"], b_l1=pr["b_l1"],
                       vx=ev["vx"], vy=ev["vy"], vz=ev["vz"], ux=pr["ux"], uy=pr["uy"], uz=pr["uz"],
                       nu_dx=ev["nu_dx"], nu_dy=ev["nu_dy"], nu_dz=ev["nu_dz"], n_tracks=pr["n_tracks"],
                       nhits=n_tot, E_vis_gev=ev["primaryEnergy"], eventID=int(eid),
                       part=hits_root.name, cls=hits_root.parent.name)
            rows.append(row)
        print(f"  {hits_root.name}: {len(events)} events, {len(rows) - n0} kept")
    if not rows:
        raise SystemExit("no usable events")
    df = pd.DataFrame(rows)
    df.attrs["module_positions"] = [positions[k] for k in sorted(positions)]
    return df


def build(cfg):
    df = build_dataframe(hits_files(cfg.build), cfg.target_fit, cfg.min_hits)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df["is_val"] = np.arange(len(df)) < int(round(VAL_FRACTION * len(df)))
    train = df[~df.is_val]
    mean, std = train[TARGETS].mean(), train[TARGETS].std().clip(lower=1e-6)
    for k in TARGETS:                                    # standardised targets for the networks
        df[f"z_{k}"] = (df[k] - mean[k]) / std[k]
    META_FILE.write_text(json.dumps(dict(
        t_bin_ns=T_BIN_NS, t_nbins=T_NBINS, features=FEATURES, targets=TARGETS,
        target_mean=mean.to_dict(), target_std=std.to_dict(), sources=[str(Path(p).resolve()) for p in cfg.build],
        n_events=len(df), n_val=int(df.is_val.sum()), n_modules=N_MODULES,
        module_positions=df.attrs["module_positions"], target_fit=cfg.target_fit), indent=1))
    df.to_parquet(TRUTH_FILE)
    print(f"wrote {len(df):,} events ({int(df.is_val.sum())} validation) with {N_MODULES} modules to {TRUTH_FILE.name}")
    print(f"labels: ln a = {mean['ln_a']:.3f} +- {std['ln_a']:.3f}, ln b = {mean['ln_b']:.3f} +- {std['ln_b']:.3f} "
          f"(a = {np.exp(mean['ln_a']):.2f}, b = {np.exp(mean['ln_b']):.2f} m); "
          f"hits/event median {df.nhits.median():.0f}; classes {df.cls.value_counts().to_dict()}")


def load_dataset():
    if not META_FILE.exists():
        raise SystemExit(f"{META_FILE} not found: run the build first (Settings.build)")
    meta = json.loads(META_FILE.read_text())
    set_features(meta["n_modules"])
    return pd.read_parquet(TRUTH_FILE), meta


def module_counts(df, meta):
    """n_dk [N, M, K] and n_d [N, M] of the rows of df, from the flat feature columns."""
    X = df[FEATURES].to_numpy(float).reshape(len(df), len(meta["module_positions"]), len(MODULE_FEATURES))
    return X[:, :, len(SUMMARY):], X[:, :, 0]


# ----------------------------------------------------------------------------------------------------------
# 2. events as token sets
# ----------------------------------------------------------------------------------------------------------
GEOM_SCALE = 30.0        # Neptune's position encodings are tuned for km-scale detectors; these events span ~30 m
N_POINT_FEATURES = 8 + T_NBINS


def random_rotation(rng):
    """Uniformly distributed 3x3 rotation matrix (QR of a Gaussian matrix, signs fixed, det +1)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class EventSet(torch.utils.data.Dataset):
    """Per event: hit-module tokens (coords [n, 4] in km/us, features [n, F]) for Neptune and the z-scored target y.
    With augment=True the token positions (relative to the vertex) are rotated at random per access; distances and
    times are unchanged."""

    def __init__(self, df, meta, known_energy=False, augment=False, seed=0, known_axis=False):
        self.augment, self.rng, self.known_energy, self.known_axis = augment, np.random.default_rng(seed), known_energy, known_axis
        self.u = df[AXIS].to_numpy(np.float32)                                 # true shower axis (used with known_axis)
        pos = np.asarray(meta["module_positions"], float)                    # [M, 3]
        X = df[FEATURES].to_numpy(np.float32).reshape(len(df), len(pos), len(MODULE_FEATURES))
        self.n, self.summary, self.hist = X[:, :, 0], X[:, :, 1:len(SUMMARY)], X[:, :, len(SUMMARY):]
        self.rel = (pos[None] - df[VERTEX].to_numpy(np.float32)[:, None]).astype(np.float32)   # r_d - v [m]
        self.dist = np.linalg.norm(self.rel, axis=2)
        self.y = df[[f"z_{k}" for k in TARGETS]].to_numpy(np.float32)
        self.ln_e = np.log(df.E_vis_gev.to_numpy(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        hit = self.n[i] > 0
        n, s, dist = self.n[i, hit], self.summary[i, hit], self.dist[i, hit]   # s: t_first, t10, t50, t90, std_t
        t_res = s[:, 0] - N_GROUP * dist / C_NS                                # delay w.r.t. straight light from v
        feats = [np.log1p(n)[:, None], s / 100.0, t_res[:, None] / 100.0, dist[:, None] / 10.0, self.hist[i, hit] / n[:, None]]
        if self.known_energy:
            feats.append(np.full((hit.sum(), 1), self.ln_e[i], np.float32))
        if self.known_axis:                          # cos psi_d: module direction against the shower axis, rotation invariant
            feats.append(((self.rel[i, hit] @ self.u[i]) / dist)[:, None])
        rel = self.rel[i, hit] @ random_rotation(self.rng).T if self.augment else self.rel[i, hit]
        coords = np.c_[rel * 1e-3, s[:, :1] * 1e-3]                            # (r_d - v) [km], t_first [us]
        return (torch.from_numpy(coords.astype(np.float32)), torch.from_numpy(np.concatenate(feats, 1).astype(np.float32)),
                torch.from_numpy(self.y[i]), i)


def collate(batch):
    coords, feats, y, idx = zip(*batch)
    batch_ids = torch.cat([torch.full((len(c),), b, dtype=torch.long) for b, c in enumerate(coords)])
    return torch.cat(coords), torch.cat(feats), batch_ids, torch.stack(y), torch.tensor(idx)


# ----------------------------------------------------------------------------------------------------------
# 3. Gaussian posterior head, model, loss
# ----------------------------------------------------------------------------------------------------------
def gaussian_head(out):
    """mu [N, 2] and the Cholesky factor L [N, 2, 2] from the 5 head outputs (mu_1, mu_2, l_11, l_21, l_22)."""
    mu = out[:, :2]
    Lm = torch.zeros(len(out), 2, 2, dtype=out.dtype, device=out.device)
    Lm[:, 0, 0] = F.softplus(out[:, 2]) + 1e-3
    Lm[:, 1, 0] = out[:, 3]
    Lm[:, 1, 1] = F.softplus(out[:, 4]) + 1e-3
    return mu, Lm


def gaussian_nll(out, y):
    """-ln N(y; mu, L L^T) per event = 1/2 |L^-1 (y - mu)|^2 + ln L_11 + ln L_22 + ln 2 pi."""
    mu, Lm = gaussian_head(out)
    r = y - mu
    z1 = r[:, 0] / Lm[:, 0, 0]                                   # forward substitution: L z = r
    z2 = (r[:, 1] - Lm[:, 1, 0] * z1) / Lm[:, 1, 1]
    return 0.5 * (z1 ** 2 + z2 ** 2) + torch.log(Lm[:, 0, 0]) + torch.log(Lm[:, 1, 1]) + np.log(2 * np.pi)


class ProfilePosterior(L.LightningModule):
    """q(y | D) = N(mu(D), Sigma(D)) with the Neptune encoder"""

    def __init__(self, n_point_features, n_modules, lr=1e-3, token_dim=64, layers=3, heads=4,
                 dropout=0.1, known_energy=False, known_axis=False):      # the two flags are stored for evaluate
        super().__init__()
        self.save_hyperparameters()
        
        G = GEOM_SCALE
        self.net = NeptuneModel(
            in_channels=n_point_features, num_patches=max(int(n_modules), 8), token_dim=token_dim,
            num_layers=layers, num_heads=heads, hidden_dim=2 * token_dim, dropout=dropout, output_dim=5,
            pool_type="attention", attn_impl="padded", compile_encoder=False,
            tokenizer_kwargs=dict(mlp_layers=[token_dim, token_dim]),
            fourier_freq_min=3.0 * G, fourier_freq_max=180.0 * G, rope_scales=(180.0 * G, 180.0 * G, 180.0 * G, 40.0 * G))
    
    def forward(self, coords, feats, batch_ids, batch_size):
        return self.net(coords, feats, batch_ids, batch_size=batch_size)

    def _step(self, batch, stage):
        coords, feats, batch_ids, y, _ = batch
        loss = gaussian_nll(self(coords, feats, batch_ids, y.size(0)), y).mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, pct_start=0.1,
                                                    total_steps=self.trainer.estimated_stepping_batches)
        return [opt], [dict(scheduler=sched, interval="step")]

# ----------------------------------------------------------------------------------------------------------
# 4. training, prediction, evaluation
# ----------------------------------------------------------------------------------------------------------
def datasets(df, meta, cfg):
    """Training and validation EventSets, plus the validation rows as a dataframe."""
    train, val = df[~df.is_val], df[df.is_val]
    return (EventSet(train, meta, cfg.known_energy, augment=cfg.augment, seed=cfg.seed, known_axis=cfg.known_axis),
            EventSet(val, meta, cfg.known_energy, known_axis=cfg.known_axis), val, cfg.augment)


def predict(model, loader, meta):
    """Posterior mean [N, 2] and covariance [N, 2, 2] in (ln a, ln b) for every event of the loader."""
    model.eval().cpu()
    outs = []
    with torch.no_grad():
        for coords, feats, batch_ids, y, _ in loader:
            outs.append(model(coords, feats, batch_ids, y.size(0)))
    mu_z, L_z = gaussian_head(torch.cat(outs))
    s = np.array([meta["target_std"][k] for k in TARGETS]); m = np.array([meta["target_mean"][k] for k in TARGETS])
    return mu_z.numpy() * s + m, (L_z @ L_z.transpose(1, 2)).numpy() * np.outer(s, s)


def save_predictions(val, y, mu, Sigma):
    np.savez(PRED_FILE, y=y, mu=mu, Sigma=Sigma, E_vis_gev=val.E_vis_gev.to_numpy(), nhits=val.nhits.to_numpy(),
             n_tracks=val.n_tracks.to_numpy() if "n_tracks" in val else np.full(len(val), np.nan),
             eventID=val.eventID.to_numpy(), cls=val.cls.to_numpy().astype(str), targets=np.array(TARGETS))
    print(f"wrote {PRED_FILE.name}")


def fit_posterior(cfg):
    """Train the Neptune posterior; evaluate the best epoch on the validation split."""
    df, meta = load_dataset()
    ds_train, ds_val, val, augment = datasets(df, meta, cfg)
    kw = dict(batch_size=cfg.batch_size, collate_fn=collate, num_workers=0)
    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, **kw)
    dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False, **kw)
    model = ProfilePosterior(N_POINT_FEATURES + int(cfg.known_energy) + int(cfg.known_axis), meta["n_modules"],
                             lr=cfg.lr, token_dim=cfg.token_dim, layers=cfg.layers,
                             known_energy=cfg.known_energy, known_axis=cfg.known_axis)
    print(f"neptune: {sum(p.numel() for p in model.parameters()):,} parameters, {len(ds_train)} train / {len(ds_val)} val events"
          + (", ln E appended to the inputs" if cfg.known_energy else ", energy not given")
          + (", cos(module, axis) appended to the tokens" if cfg.known_axis else ", axis not given")
          + (", random rotations about the vertex in training" if augment else ""))
    best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}")
    trainer = L.Trainer(accelerator=cfg.accelerator, max_epochs=cfg.epochs, log_every_n_steps=10,
                        logger=TensorBoardLogger(save_dir=str(LOG_DIR), name="neptune"), num_sanity_val_steps=0,
                        enable_progress_bar=False, callbacks=[best], gradient_clip_val=1.0)
    trainer.fit(model, dl_train, dl_val)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"last epoch: train NLL {metrics.get('train_loss', float('nan')):.3f}, val NLL {metrics.get('val_loss', float('nan')):.3f}; "
          f"best val NLL {float(best.best_model_score):.3f} ({Path(best.best_model_path).stem}, used below). "
          f"Per event in standardised units: ln 2 pi + 1 = {np.log(2 * np.pi) + 1:.3f} is predicting the prior.")
    model = ProfilePosterior.load_from_checkpoint(best.best_model_path)     # early stopping: best validation epoch
    mu, Sigma = predict(model, dl_val, meta)
    y = val[TARGETS].to_numpy(float)
    save_predictions(val, y, mu, Sigma)


def evaluate_checkpoint(cfg):
    """Validation predictions of a saved checkpoint (no training)."""
    df, meta = load_dataset()
    model = ProfilePosterior.load_from_checkpoint(cfg.evaluate)
    cfg = replace(cfg, known_energy=bool(model.hparams.get("known_energy", cfg.known_energy)),   # inputs as at training time
                  known_axis=bool(model.hparams.get("known_axis", cfg.known_axis)))
    _, ds_val, val, _ = datasets(df, meta, cfg)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batch_size, collate_fn=collate, num_workers=0)
    mu, Sigma = predict(model, dl_val, meta)
    y = val[TARGETS].to_numpy(float)
    save_predictions(val, y, mu, Sigma)


# ----------------------------------------------------------------------------------------------------------
def main(cfg=None):
    cfg = Settings() if cfg is None else cfg
    if not (cfg.build or cfg.train or cfg.evaluate):
        raise SystemExit("nothing to do: set build, train and/or evaluate in Settings")
    torch.manual_seed(cfg.seed)
    if cfg.build:
        build(cfg)
    if cfg.train:
        fit_posterior(cfg)
    if cfg.evaluate:
        evaluate_checkpoint(cfg)


if __name__ == "__main__":
    main()
