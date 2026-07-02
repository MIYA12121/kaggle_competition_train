from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET = "TVT"
SUB_TARGET = "tvt"
VERSION = "rogii_v4_pf_spatial_calibrated_optimized"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
COMMON_NUMERIC = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TAIL_WINDOWS = [15, 30, 50, 100, 250, 500, 1000]


try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except Exception:
    cKDTree = None
    HAS_SCIPY = False


def find_data_dir(user_path: str | None = None) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.extend(
        [
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
            Path("/kaggle/input/rogii-wellbore-geology-prediction"),
            Path("../input/rogii-wellbore-geology-prediction"),
            Path("rogii-wellbore-geology-prediction"),
            Path.cwd() / "rogii-wellbore-geology-prediction",
        ]
    )
    for path in candidates:
        if (path / "train").exists() and (path / "test").exists() and (path / "sample_submission.csv").exists():
            return path

    for root in [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]:
        if not root.exists():
            continue
        for sample_path in root.rglob("sample_submission.csv"):
            data_dir = sample_path.parent
            if (data_dir / "train").exists() and (data_dir / "test").exists():
                return data_dir
    raise FileNotFoundError("Could not locate ROGII competition data directory.")


def save_object(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(obj, path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("inf")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.nanstd(x[mask]) < 1e-8:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def robust_std(x: np.ndarray, default: float = 30.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return default
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return float(mad if mad > 1e-6 else np.std(x))


def interpolate_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.full(len(values), np.nan), index=values.index)
    x = np.arange(len(values), dtype=float)
    known = values.notna().to_numpy()
    filled = np.interp(x, x[known], values.to_numpy(dtype=float)[known])
    return pd.Series(filled, index=values.index)


class FormationPrior:
    def __init__(self, train_dir: Path, sample_points_per_well: int = 48):
        rows = []
        dense_xy = []
        dense_form = []
        dense_well = []
        for h_path in sorted(train_dir.glob("*__horizontal_well.csv")):
            wid = h_path.name.split("__")[0]
            try:
                df = pd.read_csv(h_path, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"well_id": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for col in FORMATIONS:
                row[col] = float(df[col].median())
            rows.append(row)

            take = np.linspace(0, len(df) - 1, min(sample_points_per_well, len(df)), dtype=int)
            sampled = df.iloc[take]
            dense_xy.append(sampled[["X", "Y"]].to_numpy(dtype=float))
            dense_form.append(sampled[FORMATIONS].to_numpy(dtype=float))
            dense_well.extend([wid] * len(sampled))

        self.well_df = pd.DataFrame(rows)
        if len(self.well_df):
            self.global_form = self.well_df[FORMATIONS].mean().to_numpy(dtype=float)
            xy = self.well_df[["x", "y"]].to_numpy(dtype=float)
        else:
            self.global_form = np.zeros(len(FORMATIONS), dtype=float)
            xy = np.empty((0, 2), dtype=float)
        self.well_ids = self.well_df["well_id"].astype(str).to_numpy() if len(self.well_df) else np.array([])
        self.scale = np.where(np.nanstd(xy, axis=0) < 1e-6, 1.0, np.nanstd(xy, axis=0)) if len(xy) else np.ones(2)
        self.tree = cKDTree(xy / self.scale) if HAS_SCIPY and len(xy) else None
        self.form = self.well_df[FORMATIONS].to_numpy(dtype=float) if len(self.well_df) else np.empty((0, len(FORMATIONS)))

        if HAS_SCIPY and dense_xy:
            self.dense_xy = np.vstack(dense_xy)
            self.dense_form = np.vstack(dense_form)
            self.dense_well = np.asarray(dense_well)
            self.dense_scale = np.where(np.nanstd(self.dense_xy, axis=0) < 1e-6, 1.0, np.nanstd(self.dense_xy, axis=0))
            self.dense_tree = cKDTree(self.dense_xy / self.dense_scale)
        else:
            self.dense_xy = np.empty((0, 2), dtype=float)
            self.dense_form = np.empty((0, len(FORMATIONS)), dtype=float)
            self.dense_well = np.array([])
            self.dense_scale = np.ones(2)
            self.dense_tree = None

    def _weighted(self, dist: np.ndarray, idx: np.ndarray, values: np.ndarray, ids: np.ndarray, self_wid: str | None) -> Tuple[np.ndarray, np.ndarray]:
        dist = np.atleast_2d(dist).astype(float)
        idx = np.atleast_2d(idx).astype(int)
        if self_wid is not None and len(ids):
            dist = np.where(ids[idx] == self_wid, np.inf, dist)
        valid = np.isfinite(dist)
        weights = np.where(valid, 1.0 / (dist + 1e-3), 0.0)
        denom = weights.sum(axis=1)
        denom_safe = np.where(denom <= 1e-12, 1.0, denom)
        pred = (values[idx] * weights[:, :, None]).sum(axis=1) / denom_safe[:, None]
        pred[denom <= 1e-12] = self.global_form
        min_dist = np.where(valid, dist, np.inf).min(axis=1)
        return pred, min_dist

    def predict(self, xy: np.ndarray, self_wid: str | None = None, k: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if self.tree is None or len(self.form) == 0:
            return np.tile(self.global_form, (len(xy), 1)), np.full(len(xy), np.inf)
        kk = min(max(k + 1, 1), len(self.form))
        dist, idx = self.tree.query(xy / self.scale, k=kk)
        return self._weighted(dist, idx, self.form, self.well_ids, self_wid)

    def predict_dense(self, xy: np.ndarray, self_wid: str | None = None, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if self.dense_tree is None or len(self.dense_form) == 0:
            return np.tile(self.global_form, (len(xy), 1)), np.full(len(xy), np.inf)
        kk = min(max(k + 1, 1), len(self.dense_form))
        dist, idx = self.dense_tree.query(xy / self.dense_scale, k=kk)
        return self._weighted(dist, idx, self.dense_form, self.dense_well, self_wid)


def typewell_features(tw: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tw = tw.copy()
    for col in ["TVT", "GR"]:
        if col in tw:
            tw[col] = pd.to_numeric(tw[col], errors="coerce")
    if "TVT" in tw:
        out["type_tvt_min"] = tw["TVT"].min()
        out["type_tvt_max"] = tw["TVT"].max()
        out["type_tvt_mean"] = tw["TVT"].mean()
        out["type_tvt_std"] = tw["TVT"].std()
    if "GR" in tw:
        out["type_gr_mean"] = tw["GR"].mean()
        out["type_gr_std"] = tw["GR"].std()
        for q in [0.05, 0.10, 0.50, 0.90, 0.95]:
            out[f"type_gr_p{int(q * 100):02d}"] = tw["GR"].quantile(q)
    if {"TVT", "GR"}.issubset(tw.columns):
        out["type_gr_slope"] = safe_slope(tw["TVT"].to_numpy(), tw["GR"].to_numpy())
    if "Geology" in tw:
        out["type_geology_nunique"] = float(tw["Geology"].nunique(dropna=True))
    return out


def calibrate_typewell_gr(kn_tvt: np.ndarray, kn_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> Tuple[float, float, float]:
    tw_at_known = np.interp(kn_tvt, tw_tvt, tw_gr)
    mask = np.isfinite(kn_gr) & np.isfinite(tw_at_known)
    if mask.sum() < 20 or np.std(tw_at_known[mask]) < 1e-6:
        bias = float(np.nanmedian(kn_gr[mask] - tw_at_known[mask])) if mask.any() else 0.0
        sigma = robust_std(kn_gr[mask] - tw_at_known[mask], default=30.0) if mask.any() else 30.0
        return 1.0, bias, float(np.clip(sigma, 8.0, 60.0))
    a, b = np.polyfit(tw_at_known[mask], kn_gr[mask], 1)
    resid = kn_gr[mask] - (a * tw_at_known[mask] + b)
    sigma = float(np.clip(robust_std(resid, default=30.0), 8.0, 60.0))
    return float(a), float(b), sigma


def gr_particle_tracker(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 128,
    n_seeds: int = 3,
    seed: int = 42,
    sigma_scale: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(hw)
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce")
    known_mask = tvt_input.notna().to_numpy()
    eval_idx = np.flatnonzero(~known_mask)
    full = interpolate_series(tvt_input).to_numpy(dtype=float).copy()
    std_full = np.zeros(n, dtype=float)
    if len(eval_idx) == 0 or known_mask.sum() < 20:
        return full, std_full

    tw_s = tw.sort_values("TVT").copy()
    tw_tvt = pd.to_numeric(tw_s["TVT"], errors="coerce").to_numpy(dtype=float)
    tw_gr = pd.to_numeric(tw_s["GR"], errors="coerce").interpolate(limit_direction="both").fillna(tw_s["GR"].mean()).to_numpy(dtype=float)
    valid_tw = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt, tw_gr = tw_tvt[valid_tw], tw_gr[valid_tw]
    if len(tw_tvt) < 5:
        return full, std_full

    order = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]
    gr_full = pd.to_numeric(hw["GR"], errors="coerce").interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=float)
    kn_idx = np.flatnonzero(known_mask)
    kn_tvt = tvt_input.to_numpy(dtype=float)[kn_idx]
    kn_gr = gr_full[kn_idx]
    a_gr, b_gr, sigma = calibrate_typewell_gr(kn_tvt, kn_gr, tw_tvt, tw_gr)
    sigma *= sigma_scale

    last_idx = int(kn_idx[-1])
    last_tvt = float(tvt_input.iloc[last_idx])
    last_z = float(hw["Z"].iloc[last_idx])
    last_md = float(hw["MD"].iloc[last_idx])
    tail = kn_idx[-min(60, len(kn_idx)) :]
    tail_md = hw["MD"].iloc[tail].to_numpy(dtype=float)
    tail_z = hw["Z"].iloc[tail].to_numpy(dtype=float)
    tail_tvt = tvt_input.iloc[tail].to_numpy(dtype=float)
    u_tail = tail_tvt + tail_z
    init_rate = safe_slope(tail_md, u_tail)
    init_u = last_tvt + last_z

    md_eval = hw["MD"].iloc[eval_idx].to_numpy(dtype=float)
    z_eval = hw["Z"].iloc[eval_idx].to_numpy(dtype=float)
    gr_eval = gr_full[eval_idx]
    all_preds = []
    for s in range(max(1, n_seeds)):
        rng = np.random.default_rng(seed + 1009 * s)
        pos_u = init_u + 2.5 * rng.standard_normal(n_particles)
        rate = init_rate + 0.01 * rng.standard_normal(n_particles)
        weights = np.ones(n_particles, dtype=float) / n_particles
        prev_md = last_md
        pred = np.empty(len(eval_idx), dtype=float)
        for i in range(len(eval_idx)):
            dm = max(md_eval[i] - prev_md, 1.0)
            prev_md = md_eval[i]
            rate = 0.995 * rate + 0.003 * rng.standard_normal(n_particles)
            pos_u = pos_u + rate * dm + 0.012 * np.sqrt(dm) * rng.standard_normal(n_particles)
            tvt_particles = pos_u - z_eval[i]
            tvt_particles = np.clip(tvt_particles, tw_tvt[0] - 150.0, tw_tvt[-1] + 150.0)
            pos_u = tvt_particles + z_eval[i]

            expected = a_gr * np.interp(tvt_particles, tw_tvt, tw_gr) + b_gr
            resid = (gr_eval[i] - expected) / sigma
            like = np.exp(-0.5 * np.minimum(resid * resid, 400.0))
            like = np.maximum(like, 1e-300)
            weights *= like
            wsum = weights.sum()
            if not np.isfinite(wsum) or wsum <= 0:
                weights[:] = 1.0 / n_particles
            else:
                weights /= wsum
            pred[i] = float(np.sum(weights * tvt_particles))
            neff = 1.0 / np.sum(weights * weights)
            if neff < 0.5 * n_particles:
                cdf = np.cumsum(weights)
                u0 = rng.uniform(0, 1.0 / n_particles)
                take = np.searchsorted(cdf, u0 + np.arange(n_particles) / n_particles)
                take = np.clip(take, 0, n_particles - 1)
                pos_u = pos_u[take] + 0.05 * rng.standard_normal(n_particles)
                rate = rate[take] + 0.001 * rng.standard_normal(n_particles)
                weights[:] = 1.0 / n_particles
        all_preds.append(pred)

    pred_mat = np.vstack(all_preds)
    full[eval_idx] = np.nanmean(pred_mat, axis=0)
    std_full[eval_idx] = np.nanstd(pred_mat, axis=0)
    return full, std_full


def tail_stats(hw: pd.DataFrame, kn_idx: np.ndarray, last_idx: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    for window in TAIL_WINDOWS:
        idx = kn_idx[-min(window, len(kn_idx)) :]
        out[f"tail_slope_md_{window}"] = safe_slope(hw["MD"].iloc[idx].to_numpy(dtype=float), tvt_input[idx])
        out[f"tail_slope_z_{window}"] = safe_slope(hw["Z"].iloc[idx].to_numpy(dtype=float), tvt_input[idx])
        out[f"tail_slope_row_{window}"] = safe_slope(idx.astype(float), tvt_input[idx])
    out["known_tvt_mean"] = float(np.nanmean(tvt_input[kn_idx]))
    out["known_tvt_std"] = float(np.nanstd(tvt_input[kn_idx]))
    out["last_known_tvt"] = float(tvt_input[last_idx])
    out["last_known_md"] = float(hw["MD"].iloc[last_idx])
    out["last_known_z"] = float(hw["Z"].iloc[last_idx])
    out["last_known_gr"] = float(hw["GR"].iloc[last_idx]) if pd.notna(hw["GR"].iloc[last_idx]) else np.nan
    out["last_known_row"] = float(last_idx)
    return out


def build_well_features(
    hw_path: Path,
    tw_path: Path,
    split: str,
    formation_prior: FormationPrior,
    pf_particles: int,
    pf_seeds: int,
) -> pd.DataFrame | None:
    wid = hw_path.name.split("__")[0]
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
    except Exception:
        return None
    for col in COMMON_NUMERIC:
        if col not in hw:
            hw[col] = np.nan
        hw[col] = pd.to_numeric(hw[col], errors="coerce")
    if split == "train" and TARGET not in hw:
        return None

    known_mask = hw["TVT_input"].notna().to_numpy()
    eval_idx = np.flatnonzero(~known_mask)
    kn_idx = np.flatnonzero(known_mask)
    if len(eval_idx) == 0 or len(kn_idx) < 10:
        return None
    last_idx = int(kn_idx[-1])
    stats = tail_stats(hw, kn_idx, last_idx)
    last_tvt = stats["last_known_tvt"]

    gr_full = hw["GR"].interpolate(limit_direction="both")
    hw["GR_filled"] = gr_full.fillna(gr_full.mean())
    pf_tvt, pf_std = gr_particle_tracker(hw, tw, n_particles=pf_particles, n_seeds=pf_seeds)

    xy_known = hw.loc[kn_idx, ["X", "Y"]].to_numpy(dtype=float)
    xy_eval = hw.loc[eval_idx, ["X", "Y"]].to_numpy(dtype=float)
    self_wid = wid if split == "train" else None
    form_known, form_known_dist = formation_prior.predict(xy_known, self_wid=self_wid)
    form_eval, form_dist = formation_prior.predict(xy_eval, self_wid=self_wid)
    dense_known, dense_known_dist = formation_prior.predict_dense(xy_known, self_wid=self_wid)
    dense_eval, dense_dist = formation_prior.predict_dense(xy_eval, self_wid=self_wid)

    z_known = hw["Z"].iloc[kn_idx].to_numpy(dtype=float)
    z_eval = hw["Z"].iloc[eval_idx].to_numpy(dtype=float)
    tvt_known = hw["TVT_input"].iloc[kn_idx].to_numpy(dtype=float)
    form_candidates = []
    form_features: Dict[str, np.ndarray] = {}
    for i, col in enumerate(FORMATIONS):
        b = float(np.nanmedian(tvt_known + z_known - form_known[:, i]))
        b_late = float(np.nanmedian(tvt_known[-min(80, len(tvt_known)) :] + z_known[-min(80, len(z_known)) :] - form_known[-min(80, len(form_known)) :, i]))
        cand = -z_eval + form_eval[:, i] + b
        cand_late = -z_eval + form_eval[:, i] + b_late
        form_candidates.append(cand)
        form_features[f"form_{col}_d"] = (cand - last_tvt).astype("float32")
        form_features[f"form_{col}_late_d"] = (cand_late - last_tvt).astype("float32")
    form_stack = np.vstack(form_candidates).T if form_candidates else np.full((len(eval_idx), 1), last_tvt)
    dense_b = float(np.nanmedian(tvt_known + z_known - dense_known[:, 0]))
    dense_late_b = float(np.nanmedian(tvt_known[-min(80, len(tvt_known)) :] + z_known[-min(80, len(z_known)) :] - dense_known[-min(80, len(dense_known)) :, 0]))
    dense_tvt = -z_eval + dense_eval[:, 0] + dense_b
    dense_late_tvt = -z_eval + dense_eval[:, 0] + dense_late_b

    md = hw["MD"].to_numpy(dtype=float)
    x = hw["X"].to_numpy(dtype=float)
    y = hw["Y"].to_numpy(dtype=float)
    z = hw["Z"].to_numpy(dtype=float)
    gr = hw["GR_filled"].to_numpy(dtype=float)
    md_since = md[eval_idx] - stats["last_known_md"]
    row_since = eval_idx.astype(float) - stats["last_known_row"]
    denom_md = np.maximum(md[-1] - stats["last_known_md"], 1.0)

    gr_series = pd.Series(gr)
    rolls: Dict[str, np.ndarray] = {}
    for window in [5, 15, 31, 51, 101]:
        r = gr_series.rolling(window, center=True, min_periods=1)
        rolls[f"gr_mean_{window}"] = r.mean().iloc[eval_idx].to_numpy(dtype="float32")
        rolls[f"gr_std_{window}"] = r.std().fillna(0.0).iloc[eval_idx].to_numpy(dtype="float32")
    for lag in [1, 5, 15, 30]:
        rolls[f"gr_lag_{lag}"] = gr_series.shift(lag).bfill().iloc[eval_idx].to_numpy(dtype="float32")
        rolls[f"gr_lead_{lag}"] = gr_series.shift(-lag).ffill().iloc[eval_idx].to_numpy(dtype="float32")

    mdd = pd.Series(md).diff().replace(0, np.nan)
    out = pd.DataFrame(
        {
            "well_id": wid,
            "id": [f"{wid}_{i}" for i in eval_idx],
            "row_index": eval_idx.astype("int32"),
            "rel_row": (eval_idx / max(len(hw) - 1, 1)).astype("float32"),
            "eval_frac": (np.arange(len(eval_idx)) / max(len(eval_idx) - 1, 1)).astype("float32"),
            "md_since": md_since.astype("float32"),
            "row_since": row_since.astype("float32"),
            "warmup_85": (1.0 - np.exp(-np.maximum(md_since, 0.0) / 85.0)).astype("float32"),
            "md_to_end": (md[-1] - md[eval_idx]).astype("float32"),
            "eval_md_frac": (md_since / denom_md).astype("float32"),
            "MD": md[eval_idx].astype("float32"),
            "X": x[eval_idx].astype("float32"),
            "Y": y[eval_idx].astype("float32"),
            "Z": z[eval_idx].astype("float32"),
            "GR": gr[eval_idx].astype("float32"),
            "last_known_tvt": np.full(len(eval_idx), last_tvt, dtype="float32"),
            "pf_tvt": pf_tvt[eval_idx].astype("float32"),
            "pf_d": (pf_tvt[eval_idx] - last_tvt).astype("float32"),
            "pf_std": pf_std[eval_idx].astype("float32"),
            "form_median_tvt": np.nanmedian(form_stack, axis=1).astype("float32"),
            "form_median_d": (np.nanmedian(form_stack, axis=1) - last_tvt).astype("float32"),
            "form_mean_d": (np.nanmean(form_stack, axis=1) - last_tvt).astype("float32"),
            "form_std": np.nanstd(form_stack, axis=1).astype("float32"),
            "form_range": (np.nanmax(form_stack, axis=1) - np.nanmin(form_stack, axis=1)).astype("float32"),
            "form_knn_dist": form_dist.astype("float32"),
            "dense_tvt": dense_tvt.astype("float32"),
            "dense_d": (dense_tvt - last_tvt).astype("float32"),
            "dense_late_d": (dense_late_tvt - last_tvt).astype("float32"),
            "dense_dist": dense_dist.astype("float32"),
            "dZ_dMD": (pd.Series(z).diff() / mdd).iloc[eval_idx].fillna(0.0).to_numpy(dtype="float32"),
            "dX_dMD": (pd.Series(x).diff() / mdd).iloc[eval_idx].fillna(0.0).to_numpy(dtype="float32"),
            "dY_dMD": (pd.Series(y).diff() / mdd).iloc[eval_idx].fillna(0.0).to_numpy(dtype="float32"),
            "gr_d1": gr_series.diff().fillna(0.0).iloc[eval_idx].to_numpy(dtype="float32"),
            "gr_d2": gr_series.diff().diff().fillna(0.0).iloc[eval_idx].to_numpy(dtype="float32"),
        }
    )
    for key, value in stats.items():
        out[key] = np.float32(value)
    for window in TAIL_WINDOWS:
        out[f"slope_md_{window}_d"] = (out[f"tail_slope_md_{window}"].astype(float) * md_since).astype("float32")
        out[f"slope_z_{window}_d"] = (out[f"tail_slope_z_{window}"].astype(float) * (z[eval_idx] - stats["last_known_z"])).astype("float32")
    for key, arr in rolls.items():
        out[key] = arr
    for key, arr in form_features.items():
        out[key] = arr
    for key, value in typewell_features(tw).items():
        out[key] = np.float32(value) if pd.notna(value) else np.float32(np.nan)
    out["gr_minus_type_p50"] = (out["GR"] - out.get("type_gr_p50", np.nan)).astype("float32")
    out["last_minus_type_mean"] = (out["last_known_tvt"] - out.get("type_tvt_mean", np.nan)).astype("float32")

    if split == "train":
        true_tvt = pd.to_numeric(hw[TARGET], errors="coerce").iloc[eval_idx].to_numpy(dtype=float)
        out["target_tvt"] = true_tvt.astype("float32")
        out["target_residual"] = (true_tvt - last_tvt).astype("float32")
    return out


def build_dataset(
    data_dir: Path,
    split: str,
    formation_prior: FormationPrior,
    pf_particles: int,
    pf_seeds: int,
    max_wells: int | None = None,
) -> pd.DataFrame:
    frames = []
    files = sorted((data_dir / split).glob("*__horizontal_well.csv"))
    if max_wells is not None:
        files = files[:max_wells]
    t0 = time.time()
    for i, h_path in enumerate(files, 1):
        wid = h_path.name.split("__")[0]
        tw_path = data_dir / split / f"{wid}__typewell.csv"
        frame = build_well_features(h_path, tw_path, split, formation_prior, pf_particles, pf_seeds)
        if frame is not None and len(frame):
            frames.append(frame)
        if i % 50 == 0 or i == len(files):
            print(f"  built {split} features {i}/{len(files)} wells, frames={len(frames)}, elapsed={time.time()-t0:.0f}s")
    if not frames:
        raise RuntimeError(f"No feature frames built for {split}.")
    data = pd.concat(frames, ignore_index=True)
    data["well_id"] = data["well_id"].astype("category")
    return data


def feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    blocked = {"well_id", "id", "target_tvt", "target_residual"}
    features = []
    for col in train_df.columns:
        if col in blocked or col not in test_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].notna().any():
            features.append(col)
    return features


def deterministic_well_split(wells: pd.Series, validation_fraction: float) -> np.ndarray:
    threshold = int(validation_fraction * 10_000)
    values = []
    for well in wells.astype(str):
        values.append(sum((i + 1) * ord(ch) for i, ch in enumerate(well)) % 10_000 < threshold)
    return np.asarray(values, dtype=bool)


def sample_idx(idx: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(idx) <= max_rows:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    out = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(out)


def make_model_candidates(seed: int = 42) -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []
    try:
        from lightgbm import LGBMRegressor

        models.append(
            (
                "lgbm_l1",
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=1800,
                    learning_rate=0.03,
                    num_leaves=143,
                    min_child_samples=60,
                    subsample=0.86,
                    subsample_freq=1,
                    colsample_bytree=0.86,
                    reg_alpha=0.05,
                    reg_lambda=0.40,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )
        models.append(
            (
                "lgbm_l2",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=1500,
                    learning_rate=0.03,
                    num_leaves=111,
                    min_child_samples=70,
                    subsample=0.90,
                    subsample_freq=1,
                    colsample_bytree=0.80,
                    reg_alpha=0.03,
                    reg_lambda=0.60,
                    random_state=seed + 1,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "xgb_abs",
                XGBRegressor(
                    objective="reg:absoluteerror",
                    n_estimators=1100,
                    learning_rate=0.03,
                    max_depth=8,
                    subsample=0.86,
                    colsample_bytree=0.86,
                    reg_alpha=0.05,
                    reg_lambda=1.5,
                    tree_method="hist",
                    random_state=seed + 2,
                    n_jobs=-1,
                ),
            )
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        models.append(
            (
                "catboost",
                CatBoostRegressor(
                    loss_function="RMSE",
                    iterations=1100,
                    learning_rate=0.035,
                    depth=7,
                    l2_leaf_reg=5.0,
                    random_seed=seed + 3,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
        )
    except Exception:
        pass
    if models:
        return models
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        models.append(
            (
                "hist_gradient_boosting",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingRegressor(
                        max_iter=800,
                        learning_rate=0.035,
                        max_leaf_nodes=45,
                        l2_regularization=0.03,
                        random_state=seed,
                    ),
                ),
            )
        )
        models.append(
            (
                "extra_trees",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_features=0.80,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=seed + 4,
                    ),
                ),
            )
        )
    except Exception:
        pass
    return models


def fit_models(models: List[Tuple[str, Any]], x: pd.DataFrame, y: pd.Series) -> List[Tuple[str, Any]]:
    fitted = []
    for name, model in models:
        print(f"fitting {name}: rows={len(x):,}, features={x.shape[1]}")
        model.fit(x, y)
        fitted.append((name, model))
    return fitted


def predict_each(models: Sequence[Tuple[str, Any]], x: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {name: np.asarray(model.predict(x), dtype=float) for name, model in models}


def model_weighted_pred(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    if not preds:
        n = 0
        return np.zeros(n, dtype=float)
    names = list(preds)
    w = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
    if w.sum() <= 0:
        w[:] = 1.0
    w /= w.sum()
    out = np.zeros(len(next(iter(preds.values()))), dtype=float)
    for name, wi in zip(names, w):
        out += wi * preds[name]
    return out


def learn_model_weights(y: np.ndarray, preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    if not preds:
        return {}
    scores = {name: rmse(y, pred) for name, pred in preds.items()}
    inv = {name: 1.0 / max(score, 1e-6) for name, score in scores.items()}
    total = sum(inv.values())
    weights = {name: value / total for name, value in inv.items()}
    print("model residual validation RMSE:", scores)
    print("model weights:", weights)
    return weights


def calibrate_blend(val_df: pd.DataFrame, model_delta: np.ndarray) -> Dict[str, Any]:
    true = val_df["target_tvt"].to_numpy(dtype=float)
    last = val_df["last_known_tvt"].to_numpy(dtype=float)
    warm = val_df["warmup_85"].to_numpy(dtype=float)
    model_d = np.asarray(model_delta, dtype=float) * warm
    pf_d = val_df["pf_d"].to_numpy(dtype=float)
    form_d = val_df["form_median_d"].to_numpy(dtype=float)
    dense_d = val_df["dense_d"].to_numpy(dtype=float)
    slope_d = val_df["slope_md_50_d"].to_numpy(dtype=float)

    candidates = {
        "baseline": last,
        "pf": last + pf_d,
        "form": last + form_d,
        "dense": last + dense_d,
        "slope50": last + slope_d,
        "model": last + model_d,
    }
    report = {name: {"rmse": rmse(true, pred), "mae": mae(true, pred)} for name, pred in candidates.items()}
    print("candidate validation report:", report)

    best = {"w_model": 0.0, "w_pf": 0.0, "w_form": 0.0, "w_dense": 0.0, "bias": 0.0}
    best_rmse = report["baseline"]["rmse"]
    grid_main = np.linspace(0.0, 1.0, 26)
    grid_aux = np.linspace(0.0, 0.6, 11)
    grid_wd = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for wm in grid_main:
        for wpf in grid_main:
            for wf in grid_aux:
                for wd in grid_wd:
                    raw = wm * model_d + wpf * pf_d + wf * form_d + wd * dense_d
                    bias = float(np.nanmedian(true - (last + raw)))
                    pred = last + raw + bias
                    score = rmse(true, pred)
                    if score < best_rmse:
                        best_rmse = score
                        best = {"w_model": float(wm), "w_pf": float(wpf), "w_form": float(wf), "w_dense": float(wd), "bias": bias}
    residual_clip = np.nanpercentile(val_df["target_residual"].to_numpy(dtype=float), [0.25, 99.75]).astype(float)
    best["validation_rmse"] = float(best_rmse)
    best["baseline_rmse"] = float(report["baseline"]["rmse"])
    best["candidate_report"] = report
    best["residual_clip"] = [float(residual_clip[0]), float(residual_clip[1])]
    print("blend:", best)
    return best


@dataclass
class ModelBundle:
    features: List[str]
    models: List[Tuple[str, Any]]
    model_weights: Dict[str, float]
    blend: Dict[str, Any]
    metadata: Dict[str, Any]


def train_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_train_rows: int,
    calibration_rows: int,
    validation_fraction: float,
) -> ModelBundle:
    features = feature_columns(train_df, test_df)
    print(f"features={len(features)} train_rows={len(train_df):,} test_rows={len(test_df):,}")
    idx_all = np.arange(len(train_df))
    val_mask = deterministic_well_split(train_df["well_id"], validation_fraction)
    val_idx_all = idx_all[val_mask]
    fit_idx_all = idx_all[~val_mask]
    if len(val_idx_all) == 0 or len(fit_idx_all) == 0:
        val_idx_all = idx_all[::5]
        fit_idx_all = np.setdiff1d(idx_all, val_idx_all)
    val_idx = sample_idx(val_idx_all, calibration_rows, seed=7)
    fit_idx = sample_idx(fit_idx_all, max(30_000, min(max_train_rows // 2, max_train_rows - len(val_idx))), seed=42)

    models = make_model_candidates(seed=42)
    if models:
        x_fit = train_df.loc[fit_idx, features].replace([np.inf, -np.inf], np.nan)
        y_fit = train_df.loc[fit_idx, "target_residual"]
        fitted_cal = fit_models(models, x_fit, y_fit)
        x_val = train_df.loc[val_idx, features].replace([np.inf, -np.inf], np.nan)
        pred_val_each = predict_each(fitted_cal, x_val)
        model_weights = learn_model_weights(train_df.loc[val_idx, "target_residual"].to_numpy(dtype=float), pred_val_each)
        pred_val = model_weighted_pred(pred_val_each, model_weights)
    else:
        print("No model libraries available. Calibrating PF/spatial blend only.")
        fitted_cal = []
        model_weights = {}
        pred_val = np.zeros(len(val_idx), dtype=float)

    blend = calibrate_blend(train_df.loc[val_idx].reset_index(drop=True), pred_val)

    final_idx = sample_idx(idx_all, max_train_rows, seed=123)
    final_models = make_model_candidates(seed=123)
    if final_models:
        x_final = train_df.loc[final_idx, features].replace([np.inf, -np.inf], np.nan)
        y_final = train_df.loc[final_idx, "target_residual"]
        final_fitted = fit_models(final_models, x_final, y_final)
    else:
        final_fitted = []

    metadata = {
        "version": VERSION,
        "features": len(features),
        "model_names": [name for name, _ in final_fitted],
        "model_weights": model_weights,
        "blend": blend,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fit_rows_calibration": int(len(fit_idx)),
        "calibration_rows": int(len(val_idx)),
        "final_train_rows": int(len(final_idx)),
        "has_scipy": HAS_SCIPY,
    }
    return ModelBundle(features, final_fitted, model_weights, blend, metadata)


def predict_bundle(bundle: ModelBundle, test_df: pd.DataFrame) -> np.ndarray:
    last = test_df["last_known_tvt"].to_numpy(dtype=float)
    if bundle.models:
        x_test = test_df[bundle.features].replace([np.inf, -np.inf], np.nan)
        pred_each = predict_each(bundle.models, x_test)
        model_d = model_weighted_pred(pred_each, bundle.model_weights)
        lo, hi = bundle.blend.get("residual_clip", [-500.0, 500.0])
        model_d = np.clip(model_d, lo, hi)
    else:
        model_d = np.zeros(len(test_df), dtype=float)
    warm = test_df["warmup_85"].to_numpy(dtype=float)
    blend = bundle.blend
    delta = (
        blend.get("w_model", 0.0) * model_d * warm
        + blend.get("w_pf", 0.0) * test_df["pf_d"].to_numpy(dtype=float)
        + blend.get("w_form", 0.0) * test_df["form_median_d"].to_numpy(dtype=float)
        + blend.get("w_dense", 0.0) * test_df["dense_d"].to_numpy(dtype=float)
    )
    pred = last + delta + blend.get("bias", 0.0)
    return pred.astype(float)


def robust_polyfit(s: np.ndarray, y: np.ndarray, degree: int = 4) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    if mask.sum() < degree + 2:
        return y.copy()
    sm, ym = s[mask], y[mask]
    coef = np.polyfit(sm, ym, degree)
    for _ in range(4):
        resid = ym - np.polyval(coef, sm)
        scale = np.median(np.abs(resid)) * 1.4826 + 1e-6
        weights = 1.0 / (1.0 + (resid / (2.0 * scale)) ** 2)
        coef = np.polyfit(sm, ym, degree, w=weights)
    fitted = y.copy()
    fitted[mask] = np.polyval(coef, sm)
    return fitted


def projection_smooth(sub: pd.DataFrame, data_dir: Path, blend: float = 0.68, degree: int = 4) -> Tuple[pd.DataFrame, int]:
    out = sub.copy()
    parts = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well_id"] = parts[0]
    out["row_index"] = parts[1].astype(int)
    pred = dict(zip(out["id"].astype(str), out["tvt"].astype(float)))
    applied = 0
    for wid, group in out.groupby("well_id"):
        try:
            hw = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
            known = hw[hw["TVT_input"].notna()]
            if len(known) < 20:
                continue
            last = known.iloc[-1]
            anchor = float(last["TVT_input"]) + float(last["Z"])
            md0 = float(last["MD"])
            md1 = float(hw["MD"].iloc[-1])
            g = group.sort_values("row_index")
            ri = g["row_index"].to_numpy(dtype=int)
            md = hw["MD"].iloc[ri].to_numpy(dtype=float)
            z = hw["Z"].iloc[ri].to_numpy(dtype=float)
            tvt = g["tvt"].to_numpy(dtype=float)
            s = (md - md0) / max(md1 - md0, 1e-6)
            u = (tvt + z) - anchor
            u_fit = robust_polyfit(s, u, degree=degree)
            tvt_fit = (anchor + u_fit) - z
            tvt_new = (1.0 - blend) * tvt + blend * tvt_fit
            if np.isfinite(tvt_new).all():
                for rid, val in zip(g["id"].astype(str), tvt_new):
                    pred[rid] = float(val)
                applied += 1
        except Exception:
            continue
    final = sub[["id"]].copy()
    final["tvt"] = final["id"].astype(str).map(pred).astype(float)
    return final, applied


def contact_tvt_from_train(hw_tr: pd.DataFrame, tw_tr: pd.DataFrame, formation: str) -> np.ndarray | None:
    if formation not in hw_tr.columns or TARGET not in hw_tr.columns:
        return None
    tw_g = tw_tr.dropna(subset=["Geology"]) if "Geology" in tw_tr.columns else pd.DataFrame()
    if len(tw_g) == 0:
        return None
    if formation in set(tw_g["Geology"].astype(str)):
        ref_tvt = tw_g.loc[tw_g["Geology"].astype(str) == formation, "TVT"].min()
    else:
        ref_tvt = tw_g["TVT"].min()
    if not np.isfinite(ref_tvt):
        return None
    phys = ref_tvt - (pd.to_numeric(hw_tr["Z"], errors="coerce") - pd.to_numeric(hw_tr[formation], errors="coerce"))
    offset = pd.to_numeric(hw_tr[TARGET], errors="coerce") - phys
    return (phys + offset.mean()).to_numpy(dtype=float)


def guarded_overlap_override(sub: pd.DataFrame, data_dir: Path, threshold_rmse: float = 1.15) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = sub.copy()
    parts = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well_id"] = parts[0]
    out["row_index"] = parts[1].astype(int)
    pred = dict(zip(out["id"].astype(str), out["tvt"].astype(float)))
    train_wells = {p.name.split("__")[0] for p in (data_dir / "train").glob("*__horizontal_well.csv")}
    counts = {"overridden_wells": 0, "skipped_wells": 0, "overridden_rows": 0}
    for wid, group in out.groupby("well_id"):
        if wid not in train_wells:
            continue
        try:
            hw_te = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
            hw_tr = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
            tw_tr = pd.read_csv(data_dir / "train" / f"{wid}__typewell.csv")
            best = None
            known = hw_te[hw_te["TVT_input"].notna()]
            if len(known) < 50:
                counts["skipped_wells"] += 1
                continue
            md_tr = pd.to_numeric(hw_tr["MD"], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(md_tr)
            md_sorted = md_tr[order]
            for formation in FORMATIONS:
                phys = contact_tvt_from_train(hw_tr, tw_tr, formation)
                if phys is None:
                    continue
                phys_sorted = phys[order]
                m = np.isfinite(md_sorted) & np.isfinite(phys_sorted)
                if m.sum() < 100:
                    continue
                kn = known[(known["MD"] >= md_sorted[m][0]) & (known["MD"] <= md_sorted[m][-1])]
                if len(kn) < 50:
                    continue
                interp = np.interp(kn["MD"].to_numpy(dtype=float), md_sorted[m], phys_sorted[m])
                score = rmse(kn["TVT_input"].to_numpy(dtype=float), interp)
                if best is None or score < best[0]:
                    best = (score, md_sorted[m], phys_sorted[m], formation)
            if best is None or best[0] > threshold_rmse:
                counts["skipped_wells"] += 1
                continue
            score, md_good, phys_good, formation = best
            md_te = hw_te["MD"].to_numpy(dtype=float)
            changed = 0
            for rid, row_idx in zip(group["id"].astype(str), group["row_index"].astype(int)):
                if 0 <= row_idx < len(md_te) and md_good[0] <= md_te[row_idx] <= md_good[-1]:
                    pred[rid] = float(np.interp(md_te[row_idx], md_good, phys_good))
                    changed += 1
            counts["overridden_wells"] += 1
            counts["overridden_rows"] += changed
            print(f"overlap override OK {wid}: formation={formation} prefix_rmse={score:.4f} rows={changed}")
        except Exception as exc:
            print(f"overlap override skipped {wid}: {exc}")
            counts["skipped_wells"] += 1
    final = sub[["id"]].copy()
    final["tvt"] = final["id"].astype(str).map(pred).astype(float)
    return final, counts


def make_submission(
    data_dir: Path,
    output_path: Path,
    model_dir: Path,
    max_train_rows: int = 1_000_000,
    calibration_rows: int = 280_000,
    validation_fraction: float = 0.16,
    pf_train_particles: int = 128,
    pf_train_seeds: int = 3,
    pf_test_particles: int = 256,
    pf_test_seeds: int = 5,
    max_train_wells: int | None = None,
) -> pd.DataFrame:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("building formation prior...")
    formation_prior = FormationPrior(data_dir / "train")
    print(f"formation prior wells={len(formation_prior.well_df)} scipy={HAS_SCIPY}")

    print("building train features...")
    train_df = build_dataset(
        data_dir,
        "train",
        formation_prior,
        pf_particles=pf_train_particles,
        pf_seeds=pf_train_seeds,
        max_wells=max_train_wells,
    )
    print("building test features...")
    test_df = build_dataset(data_dir, "test", formation_prior, pf_particles=pf_test_particles, pf_seeds=pf_test_seeds)

    bundle = train_bundle(train_df, test_df, max_train_rows, calibration_rows, validation_fraction)
    test_df = test_df.copy()
    test_df["prediction"] = predict_bundle(bundle, test_df)

    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    pred_map = dict(zip(test_df["id"].astype(str), test_df["prediction"].astype(float)))
    fallback_map = dict(zip(test_df["id"].astype(str), test_df["last_known_tvt"].astype(float)))
    raw = sample_sub[["id"]].copy()
    raw["tvt"] = raw["id"].astype(str).map(pred_map)
    raw["tvt"] = raw["tvt"].fillna(raw["id"].astype(str).map(fallback_map))
    raw["tvt"] = raw["tvt"].fillna(train_df["target_tvt"].median()).astype(float)
    raw_path = output_path.with_name(output_path.stem + "_raw.csv")
    raw.to_csv(raw_path, index=False)

    projected, n_projected = projection_smooth(raw, data_dir)
    overridden, override_counts = guarded_overlap_override(projected, data_dir)
    overridden.to_csv(output_path, index=False)

    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "raw_submission": str(raw_path),
            "final_submission": str(output_path),
            "projection_wells": int(n_projected),
            "overlap_override": override_counts,
            "elapsed_seconds": float(time.time() - t0),
            "pf_train_particles": pf_train_particles,
            "pf_train_seeds": pf_train_seeds,
            "pf_test_particles": pf_test_particles,
            "pf_test_seeds": pf_test_seeds,
            "max_train_wells": max_train_wells,
        }
    )
    bundle.metadata = metadata
    save_object(bundle, model_dir / "model_artifacts.joblib")
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        test_df.to_parquet(model_dir / "test_features.parquet", index=False)
        metadata["test_features_path"] = str(model_dir / "test_features.parquet")
    except Exception as exc:
        test_df.to_csv(model_dir / "test_features.csv", index=False)
        metadata["test_features_path"] = str(model_dir / "test_features.csv")
        metadata["parquet_fallback"] = str(exc)

    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote raw submission: {raw_path}")
    print(f"wrote final submission: {output_path}, shape={overridden.shape}")
    print(f"saved model: {model_dir / 'model_artifacts.joblib'}")
    print(f"saved metadata: {model_dir / 'model_metadata.json'}")
    return overridden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROGII v4 PF + spatial + calibrated residual solution")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--calibration-rows", type=int, default=280_000)
    parser.add_argument("--validation-fraction", type=float, default=0.16)
    parser.add_argument("--pf-train-particles", type=int, default=128)
    parser.add_argument("--pf-train-seeds", type=int, default=3)
    parser.add_argument("--pf-test-particles", type=int, default=256)
    parser.add_argument("--pf-test-seeds", type=int, default=5)
    parser.add_argument("--max-train-wells", type=int, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring notebook/kernel arguments: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    make_submission(
        data_dir=data_dir,
        output_path=Path(args.output),
        model_dir=Path(args.model_dir),
        max_train_rows=args.max_train_rows,
        calibration_rows=args.calibration_rows,
        validation_fraction=args.validation_fraction,
        pf_train_particles=args.pf_train_particles,
        pf_train_seeds=args.pf_train_seeds,
        pf_test_particles=args.pf_test_particles,
        pf_test_seeds=args.pf_test_seeds,
        max_train_wells=args.max_train_wells,
    )


if __name__ == "__main__":
    main()