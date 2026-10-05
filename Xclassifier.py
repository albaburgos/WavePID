"""NC/CC cascade classifier on single-DOM photon arrival times, built with iwpc.

Classes (label convention from the project brief):
    1 = nu_e NC : purely hadronic cascade                      -> one pi- WavePID event
    0 = nu_e CC : hadronic + electromagnetic cascade           -> one pi- event + one e- event
                  superposed at the same vertex, direction and time zero

The classifier sees only the *shape* of the photon arrival-time distribution at
one DOM: quantiles, early/late fractions, a normalised 1 ns waveform and the
mixture-fit fraction below. It is deliberately blind to the photon count and to
any energy, so it cannot separate the classes by how bright they are. To close
the indirect leak through statistical noise, every event is reduced to the same
number of randomly chosen photons (N_SUBSAMPLE, --n-subsample) first.

By default events are composed from the fixed-energy 10 GeV WavePID samples
(wavepid/run_wavepid.sh, the data behind wavepid/out/arrival_times_E10_d5.png):
CC = e-(10 GeV) + pi-(10 GeV), NC = pi-(10 GeV). With --energy-matched the
events come instead from log-uniform pools (wavepid/run_benchmark_pools.sh) with
shared E_nu and y distributions for both classes.

Real GENIE events (--genie CC_ROOT NC_ROOT, e.g. the cluster samples
wavepid/out/nue_cc_E2-50_d5_hits.root and nue_nc_E2-50_d5_hits.root): every
event is one complete nu_e CC or NC final state simulated by OMSim, labelled by
its file, nothing is superposed. Real NC events are much dimmer and softer than
CC events (median 9 vs 231 hits, 2.4 vs 16 GeV visible), so each usable NC event
is paired with the unused CC event closest in visible energy (MATCH_TOL) and
both classes share the E_vis spectrum by construction; --no-match keeps every
usable event instead. The mixture templates still come from the single-particle
pools (--em-pool/--had-pool) so y_hat_mle keeps its meaning; --templates-from
genie measures them from the two GENIE classes. Truth (E_nu, y) is joined from
the extraction CSVs (<tag>_truth.csv, global_index == eventID) found in
--truth-dir. GENIE datasets are written under their own tag
(classifier_data_genie/ ...) so the benchmark dataset survives; --train reads
the tag the same way (--dataset-tag).

Mixture model. The CC arrival-time density is a two-component mixture of the
single-cascade templates measured from the pools (times relative to the first
detected photon):

    f_CC(t) = \hat{y}_{\mathrm{MLE}} \, f_{\pi^-}(t) + (1 - \hat{y}_{\mathrm{MLE}}) \, f_{e^-}(t)

and the NC density is f_{pi^-}(t) alone (\hat{y} = 1). For every event the hadronic
light fraction \hat{y}_{MLE} is fitted by maximising sum_i log f(t_i | y) over
y in [0, 1] and is stored as the feature `y_hat_mle`; it is both a physics-driven
one-number classifier and an input to the MLP. Note that it is a fraction of
*light*, not of energy: y_light = y L_had / (y L_had + (1-y) L_em).

Usage (from anywhere; datasets are written next to this folder, in IceCube/):
    python classifier.py --build            # compose events, write classifier_data/
    python classifier.py --build --wf-bin 0.5 --wf-nbins 40 --tpl-bin 0.5   # variable binning
    python classifier.py --train            # train the MLP, logs in classifier/
    python classifier.py --build --train
    python classifier.py --build --genie out/nue_cc_E2-50_d5_hits.root out/nue_nc_E2-50_d5_hits.root
    python classifier.py --train --dataset-tag genie
"""

import argparse
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import uproot
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score

from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.models.utils import basic_model_factory
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.visualisable import Visualisable

ROOT = Path(__file__).resolve().parent            # wavepid/
PROJECT = ROOT.parent                              # IceCube/ : datasets are written here
OUT_DIR = ROOT / "out"
EM_POOL = OUT_DIR / "em_E10_d5_hits.root"    # e-  10 GeV: electromagnetic cascades
HAD_POOL = OUT_DIR / "pim_E10_d5_hits.root"  # pi- 10 GeV: hadronic cascades (--hadronic)
POOL_DIR = OUT_DIR / "benchmark"             # optional log-uniform pools (--energy-matched)
DATA_DIR = PROJECT / "classifier_data"          # iwpc dataset (features + label + weight)
TRUTH_FILE = PROJECT / "classifier_data_truth.parquet"  # same rows + truth + hit times
META_FILE = PROJECT / "classifier_data_meta.json"       # binning etc. used at --build, read at --train
PRED_FILE = PROJECT / "classifier_val_preds.npz"        # validation labels + P(NC) after --train


def set_dataset_paths(tag: str | None):
    """Redirect every dataset artefact to *_<tag> so different builds can coexist."""
    global DATA_DIR, TRUTH_FILE, META_FILE, PRED_FILE, TEMPLATE_FILE
    suffix = f"_{tag}" if tag else ""
    DATA_DIR = PROJECT / f"classifier_data{suffix}"
    TRUTH_FILE = PROJECT / f"classifier_data{suffix}_truth.parquet"
    META_FILE = PROJECT / f"classifier_data{suffix}_meta.json"
    PRED_FILE = PROJECT / f"classifier_val_preds{suffix}.npz"
    TEMPLATE_FILE = PROJECT / f"classifier_data{suffix}_templates.npz"

# Benchmark event generation. Both classes share these distributions.
E_NU_MIN, E_NU_MAX = 3.0, 50.0     # GeV, log-uniform
Y_MIN, Y_MAX = 0.2, 0.8            # inelasticity, uniform
MATCH_TOL = 0.15                   # max |dE|/E when picking a pool event
MIN_HITS = 3                       # events with fewer detected photons are dropped
# Every event is reduced to a random subsample of exactly N_SUBSAMPLE photons
# before the features are computed. Without this the classes leak the photon
# count through the statistical noise of the shape features (a 290-photon CC
# event has smoother quantiles than a 90-photon NC event). Events with fewer
# photons are dropped; 0 disables the subsampling.
N_SUBSAMPLE = 30

# Waveform: normalised photon-arrival histogram relative to the first hit.
# Defaults; --wf-bin / --wf-nbins override them at --build, and --train reads the
# values actually used from META_FILE so the network input matches the dataset.
WF_BIN_NS = 3.0
WF_NBINS = 6
WF_FEATURES = [f"wf_{i:02d}" for i in range(WF_NBINS)]


def set_binning(wf_bin_ns: float, wf_nbins: int, tpl_bin_ns: float | None = None,
                tpl_tmax_ns: float | None = None):
    """Set the waveform (and optionally template) binning and rebuild the feature list."""
    global WF_BIN_NS, WF_NBINS, WF_FEATURES, FEATURES, TPL_BIN_NS, TPL_TMAX_NS
    WF_BIN_NS, WF_NBINS = float(wf_bin_ns), int(wf_nbins)
    WF_FEATURES = [f"wf_{i:02d}" for i in range(WF_NBINS)]
    FEATURES = SUMMARY_FEATURES + WF_FEATURES
    if tpl_bin_ns is not None:
        TPL_BIN_NS = float(tpl_bin_ns)
    if tpl_tmax_ns is not None:
        TPL_TMAX_NS = float(tpl_tmax_ns)

# Summary features come first, then the waveform bins. Every summary feature
# needs a matching Scalar in get_input_scalars(); the waveform bins are
# generated there programmatically. The photon count (log10_nhits) is computed
# and saved in the truth file but is NOT an input: shape only.
SUMMARY_FEATURES = [
    "t10", "t25", "t50", "t75", "t90",   # arrival-time quantiles [ns] after first hit
    "mean_dt", "std_dt",                 # mean / spread of arrival times after first hit
    "frac_lt2", "frac_lt5", "frac_gt10", # fraction of photons in early / late windows
    "y_hat_mle",     # MLE hadronic light fraction from the two-template mixture fit
]
TEMPLATE_FILE = PROJECT / "classifier_data_templates.npz"
TPL_BIN_NS, TPL_TMAX_NS = 0.25, 40.0   # template binning for the mixture fit (--tpl-bin, --tpl-tmax)
FEATURES = SUMMARY_FEATURES + WF_FEATURES
LABEL = "is_nc"
WEIGHT = "weight"


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_pool(path: Path) -> pd.DataFrame:
    """One row per simulated event: primary energy [GeV] and its hit times [ns].

    Zero-hit events are kept (empty array) via the Events tree, so the pool is
    an unbiased sample of the energy spectrum.
    """
    f = uproot.open(path)
    hits = f["PhotonHits"].arrays(["eventID", "hitTime"], library="np")
    if "Events" in f:
        ev = f["Events"].arrays(["eventID", "primaryEnergy", "primaryParticle"], library="np")
    else:
        # Fixed-energy run from run_wavepid.sh: OMSim wrote no Events tree, so take
        # the event count and energy from the args json written next to the file.
        a = json.load(open(str(path).replace("_hits.root", "_args.json")))
        n = int(a["numevents"])
        ev = {"eventID": np.arange(n), "primaryEnergy": np.full(n, float(a["primary_energy"])),
              "primaryParticle": np.array([a["primary_particle"]] * n)}
    order = np.argsort(hits["eventID"], kind="stable")
    ids, starts = np.unique(hits["eventID"][order], return_index=True)
    split = dict(zip(ids, np.split(hits["hitTime"][order], starts[1:])))
    df = pd.DataFrame({
        "eventID": ev["eventID"],
        "energy": ev["primaryEnergy"],
        "particle": ev["primaryParticle"].astype(str),
    })
    df["times"] = [np.sort(split.get(i, np.empty(0))) for i in df["eventID"]]
    df["nhits"] = df["times"].map(len)
    print(f"  {path.name}: {len(df):,} events ({set(df['particle'])}), "
          f"E in [{df['energy'].min():.2f}, {df['energy'].max():.2f}] GeV, "
          f"mean hits {df['nhits'].mean():.1f}, zero-hit events {int((df['nhits']==0).sum())}")
    return df


class PoolSampler:
    """Draw the unused pool event closest in energy to a target, within MATCH_TOL."""

    def __init__(self, pool: pd.DataFrame, rng: np.random.Generator):
        self.pool = pool.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        self.log_e = np.log(self.pool["energy"].to_numpy())
        self.order = np.argsort(self.log_e)
        self.used = np.zeros(len(self.pool), dtype=bool)
        self.residuals = []

    def take(self, energy: float | None):
        if energy is None:                       # fixed-energy pools: any unused event
            free = np.flatnonzero(~self.used)
            if len(free) == 0:
                return None
            self.used[free[0]] = True
            self.residuals.append(0.0)
            return self.pool.iloc[free[0]]
        target = np.log(energy)
        pos = np.searchsorted(self.log_e[self.order], target)
        # walk outwards from the insertion point until an unused event is found
        lo, hi = pos - 1, pos
        while lo >= 0 or hi < len(self.order):
            cand = []
            if lo >= 0:
                cand.append(self.order[lo])
            if hi < len(self.order):
                cand.append(self.order[hi])
            cand = [c for c in cand if not self.used[c]]
            if cand:
                best = min(cand, key=lambda c: abs(self.log_e[c] - target))
                if abs(self.log_e[best] - target) > np.log1p(MATCH_TOL):
                    return None
                self.used[best] = True
                self.residuals.append(np.exp(self.log_e[best] - target) - 1)
                return self.pool.iloc[best]
            lo -= 1
            hi += 1
        return None

    @property
    def n_left(self):
        return int((~self.used).sum())


class Templates:
    """Arrival-time PDFs f_{pi-}(t), f_{e-}(t) relative to the first hit, from the pools."""

    def __init__(self, had_times: list, em_times: list):
        self.edges = np.arange(0.0, TPL_TMAX_NS + TPL_BIN_NS / 2, TPL_BIN_NS)
        self.f_had = self._pdf(had_times)
        self.f_em = self._pdf(em_times)
        self.y_grid = np.linspace(0.0, 1.0, 401)

    def _pdf(self, events):
        dt = np.concatenate([np.sort(t) - t.min() for t in events if len(t) >= MIN_HITS])
        h, _ = np.histogram(np.clip(dt, 0, TPL_TMAX_NS - 1e-6), bins=self.edges)
        pdf = (h + 0.5) / ((h + 0.5).sum() * TPL_BIN_NS)   # +0.5 avoids log(0) in empty bins
        return pdf

    def y_mle(self, dt: np.ndarray) -> float:
        idx = np.clip((np.clip(dt, 0, TPL_TMAX_NS - 1e-6) / TPL_BIN_NS).astype(int), 0, len(self.f_had) - 1)
        fh, fe = self.f_had[idx], self.f_em[idx]
        ll = np.log(self.y_grid[:, None] * fh[None, :] + (1 - self.y_grid[:, None]) * fe[None, :]).sum(axis=1)
        return float(self.y_grid[np.argmax(ll)])

    def save(self, path: Path):
        np.savez(path, edges=self.edges, f_had=self.f_had, f_em=self.f_em)


def features_from_times(times: np.ndarray, templates: Templates) -> dict:
    """Fixed-size description of one event's photon arrival times.

    Everything is measured from the first detected photon, which is what a real
    DOM can do without knowing the interaction time.
    """
    t = np.sort(np.asarray(times, dtype=float))
    dt = t - t[0]
    q = np.quantile(dt, [0.10, 0.25, 0.50, 0.75, 0.90])
    wf, _ = np.histogram(dt, bins=np.arange(0.0, WF_NBINS * WF_BIN_NS + 1e-9, WF_BIN_NS))
    out = {
        "t10": q[0], "t25": q[1], "t50": q[2], "t75": q[3], "t90": q[4],
        "mean_dt": dt.mean(), "std_dt": dt.std(),
        "frac_lt2": np.mean(dt < 2.0),
        "frac_lt5": np.mean(dt < 5.0),
        "frac_gt10": np.mean(dt > 10.0),
        "y_hat_mle": templates.y_mle(dt),
    }
    out.update({k: v for k, v in zip(WF_FEATURES, wf / len(t))})
    return out


def build_dataframe(em_pool: Path, had_pool: Path, n_per_class: int | None, seed: int,
                    fixed_energy: bool = False) -> pd.DataFrame:
    """Compose an equal mixture of NC (label 1) and CC (label 0) benchmark events.

    fixed_energy=True pairs events regardless of energy (for the single-energy
    samples from run_wavepid.sh). The classes then differ in total energy and
    are separable largely by photon count, so treat it as a pipeline check.
    """
    rng = np.random.default_rng(seed)
    print("loading pools")
    em_df, had_df = load_pool(em_pool), load_pool(had_pool)
    # NC events are a single hadronic cascade, so hadronic events below the hit
    # threshold can never be used; drop them up front instead of wasting draws.
    min_hits = max(MIN_HITS, N_SUBSAMPLE)
    n_short_pool = int((had_df["nhits"] < min_hits).sum())
    had_df = had_df[had_df["nhits"] >= min_hits].reset_index(drop=True)
    if n_per_class is None:
        # every event of either class consumes one hadronic event; CC also one EM event
        n_per_class = min(len(had_df) // 2, len(em_df))
        print(f"  auto n_per_class = {n_per_class} ({len(had_df)} usable pi- events "
              f"after dropping {n_short_pool} below {min_hits} hits, {len(em_df)} e- events)")
    em = PoolSampler(em_df, rng)
    had = PoolSampler(had_df, rng)

    templates = Templates(had.pool["times"].tolist(), em.pool["times"].tolist())
    templates.save(TEMPLATE_FILE)
    print(f"  templates: {len(templates.edges)-1} bins of {TPL_BIN_NS} ns, written to {TEMPLATE_FILE.name}")

    rows, n_fail, n_short = [], 0, 0
    for label, name in ((1.0, "NC"), (0.0, "CC")):
        made = 0
        while made < n_per_class:
            if had.n_left == 0 or (label == 0.0 and em.n_left == 0) or n_fail > 50 * n_per_class:
                raise SystemExit(f"pool exhausted while building {name}: made {made}/{n_per_class}, "
                                 f"{n_fail} unmatched draws; simulate larger pools "
                                 "(wavepid/run_benchmark_pools.sh) or loosen --match-tol")
            e_nu = float(np.exp(rng.uniform(np.log(E_NU_MIN), np.log(E_NU_MAX))))
            y = float(rng.uniform(Y_MIN, Y_MAX))
            h = had.take(None if fixed_energy else y * e_nu)
            if h is None:
                n_fail += 1
                continue
            times, e_em_true = h["times"], 0.0
            if label == 0.0:
                e = em.take(None if fixed_energy else (1.0 - y) * e_nu)
                if e is None:
                    n_fail += 1
                    continue
                times = np.concatenate([times, e["times"]])
                e_em_true = float(e["energy"])
            if len(times) < max(MIN_HITS, N_SUBSAMPLE):
                n_short += 1
                continue
            used_times = times
            if N_SUBSAMPLE > 0:
                used_times = rng.choice(times, size=N_SUBSAMPLE, replace=False)
            row = features_from_times(used_times, templates)
            if fixed_energy:                      # bookkeeping: what was actually superposed
                e_nu = float(h["energy"]) + e_em_true
                y = float(h["energy"]) / e_nu
            row.update({
                LABEL: label, "e_nu": e_nu, "y": y,
                "e_had_true": float(h["energy"]), "e_em_true": e_em_true,
                "nhits": len(times), "log10_nhits": np.log10(len(times)),
                "n_used": len(used_times), "times": np.sort(times),
            })
            rows.append(row)
            made += 1
        print(f"  {name}: {made:,} events")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    res = np.abs(np.concatenate([had.residuals, em.residuals]))
    print(f"  energy matching |dE|/E: median {np.median(res):.3f}, max {res.max():.3f}; "
          f"{n_fail} draws without a match, {n_short} events below {max(MIN_HITS, N_SUBSAMPLE)} hits dropped")
    print(f"  NC fraction = {df[LABEL].mean():.3f}; mean hits NC {df.loc[df[LABEL]==1,'nhits'].mean():.1f}, "
          f"CC {df.loc[df[LABEL]==0,'nhits'].mean():.1f}")
    print(f"  one-feature baselines (AUC for NC=1): y_hat_mle {roc_auc_score(df[LABEL], df['y_hat_mle']):.3f}; "
          f"for reference only, NOT an input: -log10_nhits {roc_auc_score(df[LABEL], -df['log10_nhits']):.3f}")

    # Equal mixture by construction; the reweighting is kept for consistency with
    # projectD so that the network converges to the clean posterior p(NC|x).
    df[WEIGHT] = 1.0
    df[WEIGHT] = df[WEIGHT] / df.groupby(LABEL)[WEIGHT].transform("mean") / 2.0
    return df


# ---------------------------------------------------------------------------
# Real GENIE events (--genie): one complete CC or NC final state per event
# ---------------------------------------------------------------------------

GENIE_TAGS = {1.0: "nue_nc", 0.0: "nue_cc"}
TRUTH_DIRS = (OUT_DIR / "genie_nue", OUT_DIR / "vis")    # searched after --truth-dir


def load_genie_truth(truth_dir: Path | None, tag: str) -> pd.DataFrame | None:
    """extract_genie_events.py truth table for <tag>, indexed by global_index, which equals
    the ROOT eventID after cluster/merge_parts.py. None when no CSV is found."""
    dirs = ((Path(truth_dir),) if truth_dir is not None else ()) + TRUTH_DIRS
    for d in dirs:
        p = d / f"{tag}_truth.csv"
        if p.exists():
            df = pd.read_csv(p).set_index("global_index")
            print(f"  truth for {tag}: {p} ({len(df):,} events)")
            return df
    print(f"  no {tag}_truth.csv in {', '.join(str(d) for d in dirs)}: e_nu and y will be NaN")
    return None


def attach_genie_truth(df: pd.DataFrame, truth: pd.DataFrame | None, tag: str) -> pd.DataFrame:
    """Add e_nu / y to a load_pool frame and check the eventID <-> global_index join."""
    df = df.copy()
    df["e_nu"], df["y"] = np.nan, np.nan
    if truth is None:
        return df
    common = df["eventID"].isin(truth.index).to_numpy()
    idx = df.loc[common, "eventID"].to_numpy()
    df.loc[common, "e_nu"] = truth.loc[idx, "e_nu_gev"].to_numpy()
    df.loc[common, "y"] = truth.loc[idx, "y"].to_numpy()
    e_vis = truth.loc[idx, "e_visible_gev"].to_numpy()
    rel = np.abs(df.loc[common, "energy"].to_numpy() - e_vis) / np.clip(e_vis, 1e-3, None)
    print(f"  {tag}: truth joined for {int(common.sum()):,}/{len(df):,} events; "
          f"|E_root - e_visible|/e_visible median {np.median(rel):.4f}, max {rel.max():.4f}")
    if np.median(rel) > 0.01:
        raise SystemExit(f"{tag}: ROOT eventID does not line up with the truth global_index "
                         "(was the file merged with cluster/merge_parts.py?)")
    return df


def genie_row(h: pd.Series, label: float, rng: np.random.Generator, templates: Templates) -> dict:
    """Feature row for one real event; truth energies follow the benchmark's conventions."""
    times = h["times"]
    used = rng.choice(times, size=N_SUBSAMPLE, replace=False) if N_SUBSAMPLE > 0 else times
    row = features_from_times(used, templates)
    e_vis, e_nu, y = float(h["energy"]), float(h["e_nu"]), float(h["y"])
    if np.isfinite(e_nu) and np.isfinite(y):
        e_had, e_em = y * e_nu, (0.0 if label == 1.0 else (1.0 - y) * e_nu)
    else:                       # no truth: NC light is all hadronic, the CC split is unknown
        e_had, e_em = (e_vis, 0.0) if label == 1.0 else (np.nan, np.nan)
    row.update({
        LABEL: label, "e_nu": e_nu, "y": y, "e_had_true": e_had, "e_em_true": e_em, "e_vis": e_vis,
        "nhits": len(times), "log10_nhits": np.log10(len(times)),
        "n_used": len(used), "times": np.sort(times),
        "eventID": int(h["eventID"]), "source": GENIE_TAGS[label],
    })
    return row


def build_genie_dataframe(cc_path: Path, nc_path: Path, em_pool: Path, had_pool: Path,
                          n_per_class: int | None, seed: int, templates_from: str = "pools",
                          match: bool = True, truth_dir: Path | None = None) -> pd.DataFrame:
    """Labelled real events: NC (1) from nc_path, CC (0) from cc_path, nothing superposed.

    match=True pairs every usable NC event with the unused CC event closest in
    visible energy (within MATCH_TOL), so both classes share the E_vis spectrum;
    unmatched NC events are dropped. match=False keeps every usable event.
    """
    rng = np.random.default_rng(seed)
    min_hits = max(MIN_HITS, N_SUBSAMPLE)
    print("loading GENIE event files")
    frames = {}
    for label, path in ((0.0, cc_path), (1.0, nc_path)):
        tag = GENIE_TAGS[label]
        df = attach_genie_truth(load_pool(path), load_genie_truth(truth_dir, tag), tag)
        n_short = int((df["nhits"] < min_hits).sum())
        frames[label] = df[df["nhits"] >= min_hits].reset_index(drop=True)
        print(f"  {tag}: {len(frames[label]):,} usable events with >= {min_hits} hits "
              f"({n_short:,} dropped), E_vis median {frames[label]['energy'].median():.2f} GeV")
    cc_df, nc_df = frames[0.0], frames[1.0]
    if cc_df.empty or nc_df.empty:
        raise SystemExit("no usable events in one class; lower --n-subsample")

    if templates_from == "pools":
        print("loading template pools")
        em_t, had_t = load_pool(em_pool), load_pool(had_pool)
        templates = Templates(had_t["times"].tolist(), em_t["times"].tolist())
    else:
        templates = Templates(nc_df["times"].tolist(), cc_df["times"].tolist())
    templates.save(TEMPLATE_FILE)
    print(f"  templates from {templates_from}: {len(templates.edges)-1} bins of {TPL_BIN_NS} ns, "
          f"written to {TEMPLATE_FILE.name}")

    rows = []
    if match:
        cc = PoolSampler(cc_df, rng)
        made, n_fail = 0, 0
        for _, h in nc_df.sample(frac=1.0, random_state=int(rng.integers(2**31))).iterrows():
            if n_per_class is not None and made >= n_per_class:
                break
            c = cc.take(float(h["energy"]))
            if c is None:
                n_fail += 1
                continue
            rows.append(genie_row(h, 1.0, rng, templates))
            rows.append(genie_row(c, 0.0, rng, templates))
            made += 1
        if made == 0:
            raise SystemExit("no NC/CC pair within --match-tol; loosen it or use --no-match")
        res = np.abs(np.asarray(cc.residuals))
        print(f"  {made:,} energy-matched NC/CC pairs ({n_fail:,} NC events without a CC partner within "
              f"{MATCH_TOL:.0%}); |dE|/E median {np.median(res):.3f}, max {res.max():.3f}")
    else:
        for label, df in ((1.0, nc_df), (0.0, cc_df)):
            if n_per_class is not None and len(df) > n_per_class:
                df = df.sample(n=n_per_class, random_state=int(rng.integers(2**31)))
            rows += [genie_row(h, label, rng, templates) for _, h in df.iterrows()]
            print(f"  {GENIE_TAGS[label]}: {len(df):,} events (no energy matching)")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for label in (1.0, 0.0):
        sel = df[LABEL] == label
        q = np.percentile(df.loc[sel, "e_vis"], [10, 50, 90])
        print(f"  {GENIE_TAGS[label]}: {int(sel.sum()):,} events, E_vis 10/50/90% = "
              f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} GeV, mean hits {df.loc[sel, 'nhits'].mean():.1f}")
    print(f"  NC fraction = {df[LABEL].mean():.3f}")
    print(f"  one-feature baselines (AUC for NC=1): y_hat_mle {roc_auc_score(df[LABEL], df['y_hat_mle']):.3f}; "
          f"for reference only, NOT an input: -log10_nhits {roc_auc_score(df[LABEL], -df['log10_nhits']):.3f}")
    df[WEIGHT] = 1.0
    df[WEIGHT] = df[WEIGHT] / df.groupby(LABEL)[WEIGHT].transform("mean") / 2.0
    return df


def build(args):
    set_dataset_paths(args.dataset_tag or ("genie" if args.genie else None))
    if args.genie:
        cc_path, nc_path = (Path(p) for p in args.genie)
        print(f"building from real GENIE events: CC {cc_path}, NC {nc_path}")
        df = build_genie_dataframe(cc_path, nc_path, Path(args.em_pool), Path(args.had_pool),
                                   args.n_per_class, args.seed, templates_from=args.templates_from,
                                   match=not args.no_match,
                                   truth_dir=Path(args.truth_dir) if args.truth_dir else None)
    else:
        print(f"building benchmark from {args.em_pool} and {args.had_pool}")
        df = build_dataframe(Path(args.em_pool), Path(args.had_pool), args.n_per_class, args.seed,
                             fixed_energy=args.fixed_energy)

    truth_cols = [LABEL, "e_nu", "y", "e_had_true", "e_em_true", "nhits", "log10_nhits", "n_used", "times"] + FEATURES
    truth_cols = [c for c in ("e_vis", "eventID", "source") if c in df] + truth_cols
    df[truth_cols].to_parquet(TRUTH_FILE)
    print(f"wrote truth + hit times for {len(df):,} events to {TRUTH_FILE}")
    meta = dict(wf_bin_ns=WF_BIN_NS, wf_nbins=WF_NBINS, tpl_bin_ns=TPL_BIN_NS, tpl_tmax_ns=TPL_TMAX_NS,
                n_subsample=N_SUBSAMPLE, min_hits=MIN_HITS, fixed_energy=bool(args.fixed_energy),
                em_pool=str(args.em_pool), had_pool=str(args.had_pool), n_per_class=int(len(df) // 2),
                features=FEATURES, mode="genie" if args.genie else "benchmark", dataset_tag=args.dataset_tag,
                cc_file=args.genie[0] if args.genie else None, nc_file=args.genie[1] if args.genie else None,
                templates_from=args.templates_from if args.genie else "pools",
                energy_matched_pairs=bool(args.genie and not args.no_match))
    META_FILE.write_text(json.dumps(meta, indent=1))
    print(f"wrote binning/meta to {META_FILE.name}: waveform {WF_NBINS} x {WF_BIN_NS:g} ns, "
          f"templates {TPL_BIN_NS:g} ns up to {TPL_TMAX_NS:g} ns")

    # PandasDirDataModule splits train/val by FILE: the first ceil(N_files*split)
    # files train and the rest validate, so keep n_files >= 2.
    file_size = max(100, len(df) // args.n_files)
    print(f"writing {len(df):,} rows to {DATA_DIR} (file_size={file_size:,} -> ~{args.n_files} files)")
    with PandasDirDataModuleBuilder(DATA_DIR, force=True, file_size=int(file_size)) as builder:
        builder.write(df[FEATURES + [LABEL, WEIGHT]].astype(np.float32))
    print("done")


# ---------------------------------------------------------------------------
# Output scalars for the Bokeh visualiser
# ---------------------------------------------------------------------------

def phi(x):
    return x["phi"]


def p_nc(x):
    return 1 / (1 + np.exp(-x["phi"]))


def p_cc(x):
    return 1 - p_nc(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CascadeClassifier(L.LightningModule, Visualisable):
    """Binary NC/CC classifier over single-DOM arrival-time features."""

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.model = basic_model_factory(
            TrivialEncoding(len(FEATURES)),
            TrivialEncoding(1),
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y, w = batch
        out = self.forward(x)
        y = y.view_as(out)
        w = w.view_as(out)
        loss = (self.criterion(out, y) * w).mean()

        preds = (torch.sigmoid(out) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.03,
            anneal_strategy="cos",
        )
        return [optimizer], [scheduler]

    # -- Visualisable ------------------------------------------------------
    def get_input_scalars(self):
        scalars = [
            Scalar("t10", r"$$t_{10}\;[\mathrm{ns}]$$", bins=np.linspace(0, 5, 100)),
            Scalar("t25", r"$$t_{25}\;[\mathrm{ns}]$$", bins=np.linspace(0, 6, 100)),
            Scalar("t50", r"$$t_{50}\;[\mathrm{ns}]$$", bins=np.linspace(0, 8, 100)),
            Scalar("t75", r"$$t_{75}\;[\mathrm{ns}]$$", bins=np.linspace(0, 12, 100)),
            Scalar("t90", r"$$t_{90}\;[\mathrm{ns}]$$", bins=np.linspace(0, 20, 100)),
            Scalar("mean_dt", r"$$\langle \Delta t\rangle\;[\mathrm{ns}]$$", bins=np.linspace(0, 10, 100)),
            Scalar("std_dt", r"$$\sigma_{\Delta t}\;[\mathrm{ns}]$$", bins=np.linspace(0, 15, 100)),
            Scalar("frac_lt2", r"$$f(\Delta t<2\,\mathrm{ns})$$", bins=np.linspace(0, 1, 100)),
            Scalar("frac_lt5", r"$$f(\Delta t<5\,\mathrm{ns})$$", bins=np.linspace(0, 1, 100)),
            Scalar("frac_gt10", r"$$f(\Delta t>10\,\mathrm{ns})$$", bins=np.linspace(0, 1, 100)),
            Scalar("y_hat_mle", r"$$\hat{y}_{\mathrm{MLE}}$$", bins=np.linspace(0, 1, 100)),
        ]
        scalars += [
            Scalar(name, rf"$$w_{{{i}}}\;[{i}\text{{-}}{i+1}\,\mathrm{{ns}}]$$", bins=np.linspace(0, 1, 100))
            for i, name in enumerate(WF_FEATURES)
        ]
        return scalars

    def get_output_scalars(self):
        return [
            ScalarFunction(phi, "phi", r"$$\phi$$"),
            ScalarFunction(p_nc, "p_NC", r"$$P(\mathrm{NC}\,|\,x)$$", bins=np.linspace(0, 1, 100)),
            ScalarFunction(p_cc, "p_CC", r"$$P(\mathrm{CC}\,|\,x)$$", bins=np.linspace(0, 1, 100)),
        ]

    def evaluate_for_visualiser(self, z):
        self.eval()
        self.cpu()
        z = torch.tensor(np.asarray(z), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.forward(z).cpu().numpy()[:, 0]
        return {"phi": out}

    @property
    def center_point(self):
        # A typical event: ~100 photons, prompt peak with a short tail.
        summary = [0.5, 0.8, 1.3, 2.0, 3.0, 1.8, 2.0, 0.7, 0.95, 0.02, 0.5]
        waveform = list(np.array([0.45, 0.35, 0.10, 0.04, 0.02, 0.01, 0.01, 0.005] + [0.0] * (WF_NBINS - 8)))
        return summary + waveform


# ---------------------------------------------------------------------------

def train(args):
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        set_binning(meta["wf_bin_ns"], meta["wf_nbins"], meta.get("tpl_bin_ns"), meta.get("tpl_tmax_ns"))
        print(f"dataset binning from {META_FILE.name}: waveform {WF_NBINS} x {WF_BIN_NS:g} ns "
              f"-> {len(FEATURES)} input features")
    dm = PandasDirDataModule(
        dataset_dir=str(DATA_DIR),
        feature_spec=[FEATURES, LABEL],
        weight_col=[WEIGHT],
        dataloader_kwargs={"num_workers": 0, "batch_size": args.batch_size},
    )
    model = CascadeClassifier(lr=args.lr)
    trainer = L.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="classifier"),
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    evaluate(model, dm)


def evaluate(model, dm):
    """Validation ROC AUC and accuracy at the end of training."""
    model.eval()
    model.cpu()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in dm.val_dataloader():
            ps.append(torch.sigmoid(model(x)).numpy()[:, 0])
            ys.append(y.numpy().reshape(-1))
    if not ys:
        print("no validation data (need n_files >= 2 at --build)")
        return
    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"validation: {len(y):,} events, AUC = {roc_auc_score(y, p):.3f}, "
          f"accuracy = {np.mean((p > 0.5) == (y > 0.5)):.3f}  (NC = 1, CC = 0)")
    np.savez(PRED_FILE, y=y, p=p, wf_bin_ns=WF_BIN_NS, wf_nbins=WF_NBINS, features=np.array(FEATURES))
    print(f"wrote validation predictions to {PRED_FILE.name}")


def main():
    global MATCH_TOL, N_SUBSAMPLE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="compose the benchmark and write the dataset")
    ap.add_argument("--train", action="store_true", help="train the classifier")
    ap.add_argument("--em-pool", default=None, help="e- WavePID sample (ROOT); default: 10 GeV run")
    ap.add_argument("--had-pool", default=None, help="pi- WavePID sample (ROOT); default: 10 GeV run")
    ap.add_argument("--energy-matched", action="store_true",
                    help="use the log-uniform pools in wavepid/out/benchmark and match energies "
                         "so both classes share E_nu and y (needs run_benchmark_pools.sh)")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="events per class for --build; default: as many as the samples allow "
                         "(each NC or CC event consumes one pi- event, each CC also one e- event)")
    ap.add_argument("--genie", nargs=2, metavar=("CC_ROOT", "NC_ROOT"), default=None,
                    help="build from real GENIE events: OMSim hits files of nu_e CC and NC final states "
                         "(one complete event each, labelled by file, nothing superposed)")
    ap.add_argument("--truth-dir", default=None,
                    help="--genie: directory with <tag>_truth.csv from extract_genie_events.py "
                         "(default search: wavepid/out/genie_nue, then wavepid/out/vis)")
    ap.add_argument("--templates-from", choices=("pools", "genie"), default="pools",
                    help="--genie: mixture templates from the single-particle pools (--em-pool/--had-pool; "
                         "y_hat_mle stays an EM/hadronic light fraction) or from the two GENIE classes")
    ap.add_argument("--no-match", action="store_true",
                    help="--genie: keep every usable event instead of energy-matched NC/CC pairs")
    ap.add_argument("--dataset-tag", default=None,
                    help="suffix of the dataset artefacts (classifier_data_<tag>/ ...); "
                         "default: none for the benchmark, 'genie' with --genie")

    ap.add_argument("--n-files", type=int, default=5,
                    help="how many files --build writes; the datamodule splits "
                         "train/val by file, so this must be >= 2")
    ap.add_argument("--wf-bin", type=float, default=WF_BIN_NS,
                    help="waveform bin width [ns] for --build")
    ap.add_argument("--wf-nbins", type=int, default=WF_NBINS,
                    help="number of waveform bins for --build (covers wf_nbins * wf_bin ns after the first hit)")
    ap.add_argument("--tpl-bin", type=float, default=TPL_BIN_NS,
                    help="template bin width [ns] for the mixture fit (--build)")
    ap.add_argument("--tpl-tmax", type=float, default=TPL_TMAX_NS,
                    help="template upper edge [ns]; later photons go in the last bin (--build)")
    ap.add_argument("--n-subsample", type=int, default=N_SUBSAMPLE,
                    help="photons kept per event before computing shape features (0 = all)")
    ap.add_argument("--match-tol", type=float, default=MATCH_TOL,
                    help="max relative energy mismatch when picking a pool event")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accelerator", default="cpu",
                    help="cpu is fastest for a dataset this small; mps/gpu also work")
    args = ap.parse_args()

    if not (args.build or args.train):
        ap.error("pass --build and/or --train")
    MATCH_TOL = args.match_tol
    N_SUBSAMPLE = args.n_subsample
    if args.build:
        set_binning(args.wf_bin, args.wf_nbins, args.tpl_bin, args.tpl_tmax)
    args.fixed_energy = not args.energy_matched
    if args.em_pool is None:
        args.em_pool = str(POOL_DIR / "em_pool_hits.root" if args.energy_matched else EM_POOL)
    if args.had_pool is None:
        args.had_pool = str(POOL_DIR / "pim_pool_hits.root" if args.energy_matched else HAD_POOL)
    if args.build:
        build(args)
    if args.train:
        train(args)


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("CLASSIFIER_TORCH_SEED", 1234)))
    main()
