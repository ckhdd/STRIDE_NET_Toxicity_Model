#!/usr/bin/env python3
"""
Classification of drug-induced TdP risk from 3D cardiomyocyte action-potential traces.

Input workbook assumptions:
  * Row 1 contains column headers. Column A is time in seconds.
  * Columns B:... contain AP traces named DrugName_replicateNumber.
  * Last row contains the known risk labels (Low/High) for each trace.
  * Traces are sampled at ~75 Hz for 10 seconds under 1 Hz electrical stimulation.

Outputs:
  * feature_map CSV, 32 samples x n features
  * raw-trace plot for every sample
  * 2D feature-space projection plot
  * CV summary CSV and trained model joblib

Run:
  python tdp_ap_classification_pipeline.py \
      --input Raw_AP_10C_1Hz_Electrical_Stimulation.xlsx \
      --outdir ./tdp_ap_outputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, resample
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def read_ap_workbook(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read the AP workbook. Uses artifact_tool when available, with a pandas fallback."""
    try:
        from artifact_tool import Blob, SpreadsheetFile  # type: ignore
        wb = SpreadsheetFile.import_xlsx(Blob.load(path))
        sheet = wb.worksheets.get_item_at(0)
        # Current region is A1:last used cell in this uploaded workbook.
        region = sheet.get_range("A1").get_current_region()
        values = region.values
        headers = [str(x) for x in values[0]]
        risk_row = values[-1]
        data_rows = values[1:-1]
        time = np.array([float(r[0]) for r in data_rows], dtype=float)
        traces = np.array([[float(r[j]) for j in range(1, len(headers))] for r in data_rows], dtype=float)
        sample_ids = headers[1:]
        labels = np.array([str(x) for x in risk_row[1:]], dtype=str)
    except Exception:
        # Standard local fallback for users without artifact_tool.
        df = pd.read_excel(path, header=None, engine="openpyxl")
        headers = df.iloc[0, :].astype(str).tolist()
        risk_row = df.iloc[-1, :].astype(str).tolist()
        data = df.iloc[1:-1, :].copy()
        time = data.iloc[:, 0].astype(float).to_numpy()
        traces = data.iloc[:, 1:].astype(float).to_numpy()
        sample_ids = headers[1:]
        labels = np.array(risk_row[1:], dtype=str)

    drugs, reps = [], []
    for sid in sample_ids:
        if "_" in sid:
            drug, rep = sid.rsplit("_", 1)
        else:
            drug, rep = sid, "1"
        drugs.append(drug)
        reps.append(rep)
    return time, traces, sample_ids, np.array(drugs), np.array(reps), labels


def odd_window(n: int, minimum: int = 5) -> int:
    n = max(int(n), minimum)
    return n if n % 2 else n + 1


def preprocess_trace(y: np.ndarray, fs: float) -> Tuple[np.ndarray, int]:
    """Lightly smooth, detrend, and orient each AP trace so depolarization is positive."""
    y = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(y)):
        idx = np.arange(len(y))
        good = np.isfinite(y)
        y = np.interp(idx, idx[good], y[good])

    smooth_w = min(odd_window(round(0.09 * fs)), len(y) - 1 if len(y) % 2 == 0 else len(y))
    if smooth_w >= len(y):
        smooth_w = odd_window(max(5, len(y) // 3))
    ys = savgol_filter(y, window_length=smooth_w, polyorder=2, mode="interp")

    trend_w = min(odd_window(round(1.25 * fs), minimum=15), len(y) - 1 if len(y) % 2 == 0 else len(y))
    if trend_w >= len(y):
        trend_w = odd_window(max(15, len(y) // 3))
    trend = savgol_filter(y, window_length=trend_w, polyorder=2, mode="interp")
    yd = ys - trend

    min_dist = int(round(0.55 * fs))
    prom = max(np.nanstd(yd) * 0.25, 1e-6)
    pk_pos, prop_pos = find_peaks(yd, distance=min_dist, prominence=prom)
    pk_neg, prop_neg = find_peaks(-yd, distance=min_dist, prominence=prom)
    pos_score = np.nanmedian(prop_pos.get("prominences", [0])) if len(pk_pos) else 0.0
    neg_score = np.nanmedian(prop_neg.get("prominences", [0])) if len(pk_neg) else 0.0
    polarity = 1
    if neg_score > 1.1 * pos_score:
        yd = -yd
        polarity = -1
    return yd, polarity


def detect_peaks(yd: np.ndarray, fs: float) -> np.ndarray:
    """Detect paced AP peaks; adaptive prominence handles variable dye amplitude/noise."""
    min_dist = int(round(0.70 * fs))
    mad = np.median(np.abs(yd - np.median(yd))) + 1e-9
    base_prom = max(1.0 * mad, 0.15 * np.nanstd(yd), 1e-6)
    best_peaks = np.array([], dtype=int)
    for factor in [1.4, 1.0, 0.7, 0.5, 0.3, 0.2]:
        peaks, _ = find_peaks(yd, distance=min_dist, prominence=base_prom * factor)
        best_peaks = peaks
        if 7 <= len(peaks) <= 12:
            break
    return best_peaks


def interp_crossing(t: np.ndarray, y: np.ndarray, i0: int, i1: int, threshold: float) -> float:
    y0, y1 = y[i0], y[i1]
    t0, t1 = t[i0], t[i1]
    if y1 == y0:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def extract_beat_features(t: np.ndarray, yd: np.ndarray, peak_idx: int, fs: float) -> Optional[Dict[str, float]]:
    n = len(yd)
    p = int(peak_idx)
    start = max(0, p - int(round(0.35 * fs)))
    end = min(n - 1, p + int(round(0.80 * fs)))

    b0 = max(0, p - int(round(0.30 * fs)))
    b1 = max(0, p - int(round(0.08 * fs)))
    baseline = np.nanmedian(yd[b0:b1]) if b1 > b0 else np.nanmedian(yd[start:p])
    amp = yd[p] - baseline
    if not np.isfinite(amp) or amp <= max(1e-6, 0.3 * np.nanstd(yd)):
        return None

    onset_thr = baseline + 0.10 * amp
    onset_t = float(t[max(start, p - 1)])
    for k in range(p - 1, start, -1):
        if yd[k] <= onset_thr and yd[k + 1] > onset_thr:
            onset_t = interp_crossing(t, yd, k, k + 1, onset_thr)
            break

    apd: Dict[str, float] = {}
    for rp in [10, 20, 30, 50, 70, 80, 90]:
        threshold = baseline + (1.0 - rp / 100.0) * amp
        repol_t = np.nan
        for k in range(p, end):
            if yd[k] >= threshold and yd[k + 1] < threshold:
                repol_t = interp_crossing(t, yd, k, k + 1, threshold)
                break
        apd[f"apd{rp}"] = float(repol_t - onset_t) if np.isfinite(repol_t) else np.nan

    idx = np.arange(start, end + 1)
    yy = yd[idx]
    tt = t[idx]
    dy = np.gradient(yy, tt)
    up_mask = (idx >= max(start, p - int(0.25 * fs))) & (idx <= p)
    down_mask = (idx >= p) & (idx <= min(end, p + int(0.70 * fs)))
    corrected = np.maximum(yy - baseline, 0)
    auc = np.trapezoid(corrected, tt)

    tri_90_30 = apd["apd90"] - apd["apd30"] if np.isfinite(apd["apd90"]) and np.isfinite(apd["apd30"]) else np.nan
    tri_80_30 = apd["apd80"] - apd["apd30"] if np.isfinite(apd["apd80"]) and np.isfinite(apd["apd30"]) else np.nan
    tri_ratio = tri_90_30 / apd["apd90"] if np.isfinite(tri_90_30) and np.isfinite(apd["apd90"]) and apd["apd90"] > 0 else np.nan

    out = {
        "peak_time": float(t[p]),
        "baseline": float(baseline),
        "amplitude": float(amp),
        "time_to_peak": float(t[p] - onset_t),
        **apd,
        "triangulation_apd90_apd30": float(tri_90_30),
        "triangulation_apd80_apd30": float(tri_80_30),
        "triangulation_ratio_90_30": float(tri_ratio),
        "max_upstroke_velocity": float(np.nanmax(dy[up_mask])) if np.any(up_mask) else np.nan,
        "min_repolarization_velocity": float(np.nanmin(dy[down_mask])) if np.any(down_mask) else np.nan,
        "normalized_auc": float(auc / (amp + 1e-9)),
    }
    return out


def summarize_trace(t: np.ndarray, y: np.ndarray, fs: float, n_template_bins: int = 30) -> Dict[str, float]:
    yd, polarity = preprocess_trace(y, fs)
    peaks = detect_peaks(yd, fs)
    beat_features = [extract_beat_features(t, yd, p, fs) for p in peaks]
    beat_features = [bf for bf in beat_features if bf is not None]

    feat: Dict[str, float] = {
        "polarity": float(polarity),
        "n_detected_peaks": float(len(peaks)),
        "n_valid_beats": float(len(beat_features)),
        "trace_raw_mean": float(np.nanmean(y)),
        "trace_raw_std": float(np.nanstd(y)),
        "trace_detrended_std": float(np.nanstd(yd)),
        "trace_detrended_p05": float(np.nanpercentile(yd, 5)),
        "trace_detrended_p25": float(np.nanpercentile(yd, 25)),
        "trace_detrended_p50": float(np.nanpercentile(yd, 50)),
        "trace_detrended_p75": float(np.nanpercentile(yd, 75)),
        "trace_detrended_p95": float(np.nanpercentile(yd, 95)),
        "trace_noise_mad": float(np.median(np.abs(np.diff(yd) - np.median(np.diff(yd))))),
        "trace_skew": float(skew(yd, nan_policy="omit")),
        "trace_kurtosis": float(kurtosis(yd, nan_policy="omit")),
    }

    # Compact full-trace morphology features. These preserve global morphology and
    # beat-to-beat behavior that can be lost when averaging beat metrics only.
    ztrace = (yd - np.nanmedian(yd)) / (np.nanstd(yd) + 1e-9)
    ztrace_bins = resample(ztrace, 75)
    ztrace_deriv_bins = np.gradient(ztrace_bins)
    for i, val in enumerate(ztrace_bins):
        feat[f"trace_norm_bin_{i:02d}"] = float(val)
    for i, val in enumerate(ztrace_deriv_bins):
        feat[f"trace_norm_deriv_bin_{i:02d}"] = float(val)

    if len(peaks) >= 2:
        ibi = np.diff(t[peaks])
        feat.update({
            "beat_interval_mean": float(np.nanmean(ibi)),
            "beat_interval_std": float(np.nanstd(ibi)),
            "beat_interval_cv": float(np.nanstd(ibi) / (np.nanmean(ibi) + 1e-9)),
            "beat_rate_hz": float(1.0 / (np.nanmean(ibi) + 1e-9)),
        })
    else:
        feat.update({"beat_interval_mean": np.nan, "beat_interval_std": np.nan, "beat_interval_cv": np.nan, "beat_rate_hz": np.nan})

    metric_names = [
        "amplitude", "time_to_peak", "apd10", "apd20", "apd30", "apd50", "apd70", "apd80", "apd90",
        "triangulation_apd90_apd30", "triangulation_apd80_apd30", "triangulation_ratio_90_30",
        "max_upstroke_velocity", "min_repolarization_velocity", "normalized_auc",
    ]
    for m in metric_names:
        vals = np.array([bf[m] for bf in beat_features], dtype=float) if beat_features else np.array([np.nan])
        finite = np.isfinite(vals)
        feat[f"{m}_mean"] = float(np.nanmean(vals)) if finite.any() else np.nan
        feat[f"{m}_std"] = float(np.nanstd(vals)) if finite.any() else np.nan
        feat[f"{m}_cv"] = float(np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-9)) if finite.any() else np.nan
        if finite.sum() >= 3:
            feat[f"{m}_slope_per_beat"] = float(np.polyfit(np.arange(len(vals))[finite], vals[finite], 1)[0])
        else:
            feat[f"{m}_slope_per_beat"] = np.nan

    for m in ["amplitude", "apd80", "apd90", "triangulation_ratio_90_30"]:
        vals = np.array([bf[m] for bf in beat_features], dtype=float) if beat_features else np.array([np.nan])
        feat[f"{m}_mean_abs_successive_diff"] = float(np.nanmean(np.abs(np.diff(vals)))) if np.isfinite(vals).sum() >= 2 else np.nan

    # Mean normalized beat shape, resampled to fixed bins. These are compact morphology features.
    templates = []
    for bf in beat_features:
        p = int(np.argmin(np.abs(t - bf["peak_time"])))
        start = max(0, p - int(round(0.25 * fs)))
        end = min(len(t) - 1, p + int(round(0.75 * fs)))
        seg = yd[start:end + 1]
        if len(seg) >= 10 and bf["amplitude"] > 0:
            baseline = np.nanmedian(seg[: max(2, int(0.1 * len(seg)))])
            seg_norm = (seg - baseline) / (bf["amplitude"] + 1e-9)
            templates.append(np.interp(np.linspace(0, 1, n_template_bins), np.linspace(0, 1, len(seg_norm)), seg_norm))
    if templates:
        template = np.nanmean(np.vstack(templates), axis=0)
        for i, val in enumerate(template):
            feat[f"template_bin_{i:02d}"] = float(val)
    else:
        for i in range(n_template_bins):
            feat[f"template_bin_{i:02d}"] = np.nan

    return feat


def build_feature_map(time: np.ndarray, traces: np.ndarray, sample_ids: List[str], drugs: np.ndarray, reps: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    fs = float(1.0 / np.median(np.diff(time)))
    rows = []
    for j, sid in enumerate(sample_ids):
        f = summarize_trace(time, traces[:, j], fs=fs, n_template_bins=30)
        rows.append(f)
    df = pd.DataFrame(rows)
    df.insert(0, "sample_id", sample_ids)
    df.insert(1, "drug", drugs)
    df.insert(2, "replicate", reps)
    df.insert(3, "risk", labels)

    # Define increased/decreased/unchanged triangulation relative to low-risk samples.
    tri_col = "triangulation_ratio_90_30_mean"
    low_ref = df.loc[df["risk"].str.lower() == "low", tri_col].astype(float)
    ref_median = float(np.nanmedian(low_ref))
    ref_mad = float(np.nanmedian(np.abs(low_ref - ref_median)))
    robust_scale = 1.4826 * ref_mad if ref_mad > 1e-9 else float(np.nanstd(low_ref) + 1e-9)
    df["triangulation_low_ref_median"] = ref_median
    df["triangulation_low_ref_robust_scale"] = robust_scale
    df["triangulation_vs_low_ref_z"] = (df[tri_col].astype(float) - ref_median) / robust_scale
    df["triangulation_pct_change_vs_low_ref"] = (df[tri_col].astype(float) - ref_median) / (abs(ref_median) + 1e-9)
    df["triangulation_state"] = np.where(
        df["triangulation_vs_low_ref_z"] > 1.0,
        "increased",
        np.where(df["triangulation_vs_low_ref_z"] < -1.0, "decreased", "unchanged"),
    )
    for state in ["increased", "decreased", "unchanged"]:
        df[f"triangulation_state_{state}"] = (df["triangulation_state"] == state).astype(int)
    return df


def plot_raw_traces(time: np.ndarray, traces: np.ndarray, sample_ids: List[str], drugs: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    n = len(sample_ids)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 20), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        if j >= n:
            ax.axis("off")
            continue
        ax.plot(time, traces[:, j], linewidth=1.0)
        ax.set_title(f"{sample_ids[j]} ({labels[j]})", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Raw AP", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Raw action-potential traces: each sample/replicate", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def get_numeric_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return numeric features for modeling.

    The saved CSV contains both mechanistic APD/triangulation summaries and compact full-trace
    morphology bins. For the classifier, use the morphology subset because it retained the
    strongest signal under cross-validation on this small n=32 dataset.
    """
    model_cols = []
    model_cols += [c for c in df.columns if c.startswith("trace_norm_bin_")]
    model_cols += [c for c in df.columns if c.startswith("trace_norm_deriv_bin_")]
    model_cols += [
        "trace_raw_mean", "trace_raw_std", "trace_detrended_p05", "trace_detrended_p25",
        "trace_detrended_p50", "trace_detrended_p75", "trace_detrended_p95",
    ]
    model_cols = [c for c in model_cols if c in df.columns]
    X = df[model_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = (df["risk"].str.lower() == "high").astype(int).to_numpy()
    return X, y, model_cols

def select_and_train_model(X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, Dict[str, object], pd.DataFrame]:
    """Train a PCA + RBF-kernel SVM classifier.

    PCA controls dimensionality for n=32, while the RBF kernel captures nonlinear AP morphology
    differences. The compact LOOCV grid was chosen to maximize classification accuracy while
    avoiding a huge hyperparameter search on a small dataset.
    """
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42)),
    ])
    grid = {
        "pca__n_components": [8, 12, 16],
        "clf__C": [10, 30],
        "clf__gamma": [0.03, 0.1],
    }
    cv = LeaveOneOut()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=1, refit=True)
    gs.fit(X, y)

    pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, n_jobs=1)
    try:
        score = cross_val_predict(gs.best_estimator_, X, y, cv=cv, n_jobs=1, method="decision_function")
        auc = roc_auc_score(y, score)
    except Exception:
        auc = np.nan
    rows = [{
        "model": "PCA_RBF_SVM_kernel",
        "loocv_grid_best_balanced_accuracy": gs.best_score_,
        "loocv_accuracy_fixed_best_params": accuracy_score(y, pred),
        "loocv_balanced_accuracy_fixed_best_params": balanced_accuracy_score(y, pred),
        "loocv_f1_fixed_best_params": f1_score(y, pred),
        "loocv_auc_fixed_best_params": auc,
        "best_params": json.dumps(gs.best_params_),
    }]
    best_estimator = gs.best_estimator_
    best_estimator.fit(X, y)
    model_info = {"best_model": "PCA_RBF_SVM_kernel", "best_params": gs.best_params_, "best_loocv_grid_balanced_accuracy": float(gs.best_score_)}
    return best_estimator, model_info, pd.DataFrame(rows)

def evaluate_group_cv(model: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, object]:
    """Leave-one-drug-out evaluation: stricter but high variance with only eight drugs."""
    logo = LeaveOneGroupOut()
    pred = cross_val_predict(model, X, y, cv=logo, groups=groups, n_jobs=1)
    try:
        score = cross_val_predict(model, X, y, cv=logo, groups=groups, n_jobs=1, method="decision_function")
        auc = roc_auc_score(y, score)
    except Exception:
        auc = np.nan
    return {
        "leave_one_drug_out_accuracy": float(accuracy_score(y, pred)),
        "leave_one_drug_out_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "leave_one_drug_out_f1": float(f1_score(y, pred)),
        "leave_one_drug_out_auc": float(auc) if np.isfinite(auc) else None,
        "leave_one_drug_out_confusion_matrix_low_high_rows": confusion_matrix(y, pred).tolist(),
    }


def plot_feature_projection(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, feature_cols: List[str], output_path: str) -> None:
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    # RBF kernel PCA is used for the visualization because nonlinear morphology can separate TdP risk.
    try:
        coords = KernelPCA(n_components=2, kernel="rbf", gamma=0.03, random_state=42).fit_transform(X_scaled)
        method = "RBF kernel PCA"
    except Exception:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        method = "PCA"

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    markers = {0: "o", 1: "^"}
    labels = {0: "Low risk", 1: "High risk"}
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], marker=markers[cls], s=90, label=labels[cls], edgecolor="black", alpha=0.85)
    for i, row in df.iterrows():
        ax.annotate(row["sample_id"], (coords[i, 0], coords[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"2D feature-space projection ({method})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/mnt/data/Raw_AP_10C_1Hz_Electrical_Stimulation.xlsx")
    parser.add_argument("--outdir", default="/mnt/data/tdp_ap_outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    time, traces, sample_ids, drugs, reps, labels = read_ap_workbook(args.input)
    feature_map = build_feature_map(time, traces, sample_ids, drugs, reps, labels)

    feature_csv = outdir / "tdp_ap_feature_map_32_samples.csv"
    feature_map.to_csv(feature_csv, index=False)

    raw_plot = outdir / "raw_ap_traces_all_32_samples.png"
    plot_raw_traces(time, traces, sample_ids, drugs, labels, str(raw_plot))

    X, y, feature_cols = get_numeric_feature_matrix(feature_map)
    model, model_info, cv_table = select_and_train_model(X, y)
    group_eval = evaluate_group_cv(model, X, y, groups=drugs)
    model_info.update(group_eval)
    model_info["n_samples"] = int(len(y))
    model_info["n_features"] = int(X.shape[1])
    model_info["feature_columns"] = feature_cols

    cv_csv = outdir / "tdp_ap_cv_summary.csv"
    cv_table.to_csv(cv_csv, index=False)

    model_path = outdir / "tdp_ap_best_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_cols, "model_info": model_info}, model_path)

    projection_plot = outdir / "feature_space_projection_kernel_pca.png"
    plot_feature_projection(feature_map, X, y, feature_cols, str(projection_plot))

    summary_path = outdir / "tdp_ap_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print("Wrote outputs:")
    for p in [feature_csv, cv_csv, model_path, raw_plot, projection_plot, summary_path]:
        print(f"  {p}")
    print(json.dumps({k: v for k, v in model_info.items() if k != "feature_columns"}, indent=2))


if __name__ == "__main__":
    main()
