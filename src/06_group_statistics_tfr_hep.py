#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Between-subject group statistics for HEP-TFR (Good vs Non-Good).

Adaptation of src/06_group_statistics_tfr.py for Heartbeat-Evoked TFR data.
Includes cropping logic to avoid edge artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _default_processed_dir() -> Path:
    return _repo_root() / "data" / "processed"

def _default_out_dir() -> Path:
    return _repo_root() / "data" / "group_analysis_between_subjects_tfr_hep"

def _tfr_hep_path(processed_dir: Path, subject_id: str, condition: str) -> Path:
    pdir = processed_dir / subject_id
    if condition == "target":
        return pdir / f"{subject_id}_tfr_hep_target.h5"
    if condition == "control":
        return pdir / f"{subject_id}_tfr_hep_control.h5"
    raise ValueError("condition must be 'target' or 'control'")

def _load_good_responders(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"good_responders_median.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("good_responders_median.csv must contain a 'session_id' column")
    ids = [str(x).strip() for x in df["session_id"].tolist() if x]
    seen = set()
    uniq = []
    for sid in ids:
        if sid not in seen:
            uniq.append(sid)
            seen.add(sid)
    return uniq

def _list_processed_subjects(processed_dir: Path) -> List[str]:
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    subjects = []
    for p in processed_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            subjects.append(p.name)
    subjects.sort()
    return subjects

# ---------------------------------------------------------------------------
# TFR Loading & Processing
# ---------------------------------------------------------------------------

def _read_average_tfr(path: Path) -> mne.time_frequency.AverageTFR:
    obj = mne.time_frequency.read_tfrs(str(path))
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError(f"read_tfrs empty: {path}")
        tfr = obj[0]
    else:
        tfr = obj
    return tfr

def _roi_mean_and_mask_from_tfr(
    tfr: mne.time_frequency.AverageTFR, roi_candidates: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    existing = set(tfr.info["ch_names"])
    picked = [ch for ch in roi_candidates if ch in existing]
    if not picked:
        raise ValueError(f"None of ROI channels found. ROI={roi_candidates}")
    
    mask = np.asarray([ch in set(picked) for ch in roi_candidates], dtype=bool)
    t = tfr.copy().pick(picked)
    data = t.data
    return data.mean(axis=0), picked, mask

def _align_tfr_roi_to_reference(
    data_ft: np.ndarray, freqs: np.ndarray, times: np.ndarray,
    ref_freqs: np.ndarray, ref_times: np.ndarray, subject_id: str
) -> Optional[np.ndarray]:
    if freqs[0] > ref_freqs[0] or freqs[-1] < ref_freqs[-1]: return None
    if times[0] > ref_times[0] or times[-1] < ref_times[-1]: return None

    # Freq interp
    tmp = np.empty((ref_freqs.size, times.size), dtype=float)
    for ti in range(times.size):
        tmp[:, ti] = np.interp(ref_freqs, freqs, data_ft[:, ti])
    
    # Time interp
    out = np.empty((ref_freqs.size, ref_times.size), dtype=float)
    for fi in range(ref_freqs.size):
        out[fi, :] = np.interp(ref_times, times, tmp[fi, :])
    return out

def _collect_tfr_group_matrix(
    processed_dir: Path, subject_ids: Sequence[str], condition: str,
    roi_candidates: List[str], ref_freqs: np.ndarray, ref_times: np.ndarray,
    crop_tmin: float, crop_tmax: float
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    
    data_list, used, pick_masks = [], [], []

    for sid in subject_ids:
        path = _tfr_hep_path(processed_dir, sid, condition)
        
        if not path.exists():
            _log_warn(f"Missing HEP-TFR {condition} for {sid}; skip")
            continue
        try:
            tfr = _read_average_tfr(path)
            
            # --- Crop Logic ---
            # Crop immediately after loading to ensure stats are only on valid window
            tfr.crop(tmin=crop_tmin, tmax=crop_tmax)
            # ------------------

            data_ft, _, pick_mask = _roi_mean_and_mask_from_tfr(tfr, roi_candidates)
        except Exception as e:
            _log_warn(f"Failed to load HEP-TFR {sid}: {e}")
            continue

        aligned = _align_tfr_roi_to_reference(
            data_ft, tfr.freqs, tfr.times, ref_freqs, ref_times, sid
        )
        if aligned is None:
            continue
            
        data_list.append(aligned)
        used.append(sid)
        pick_masks.append(pick_mask)

    if not data_list:
        raise RuntimeError(f"No usable subjects for HEP-TFR condition={condition}")
        
    return np.stack(data_list), used, np.stack(pick_masks) if pick_masks else np.array([])

# ---------------------------------------------------------------------------
# Statistics & Plotting
# ---------------------------------------------------------------------------

def _run_cluster_2samp(X1, X2, n_perm, tail, seed):
    # Minimal wrapper for mne permutation test
    T_obs, clusters, cluster_p, _ = mne.stats.permutation_cluster_test(
        [X1, X2], n_permutations=n_perm, tail=tail, seed=seed,
        out_type="mask", verbose=False
    )
    return T_obs, cluster_p, clusters

def _plot_tfr_group_triplet(
    roi_name: str, condition: str, times: np.ndarray, freqs: np.ndarray,
    good_mean: np.ndarray, nongood_mean: np.ndarray, diff_mean: np.ndarray,
    sig_mask: np.ndarray, out_png: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    vmax = np.nanmax(np.abs(diff_mean)) or 1.0
    
    kw = dict(origin="lower", aspect="auto", extent=[times[0], times[-1], freqs[0], freqs[-1]])
    
    axes[0].imshow(good_mean, cmap="viridis", **kw)
    axes[0].set_title(f"Good ({condition})")
    axes[1].imshow(nongood_mean, cmap="viridis", **kw)
    axes[1].set_title(f"Non-Good ({condition})")
    axes[2].imshow(diff_mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax, **kw)
    axes[2].set_title(f"Diff (G-NG)")

    if sig_mask.any():
        axes[2].contour(times, freqs, sig_mask, levels=[0.5], colors="k", linewidths=1.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"HEP-TFR {roi_name}")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def run_analysis(
    good_csv: Path, processed_dir: Path, out_dir: Path,
    roi_dict: Dict[str, List[str]], roi_names: Sequence[str],
    exclude: Sequence[str], n_permutations: int, alpha: float, tail: int, seed: int,
    stat_tmin: float, stat_tmax: float
) -> None:
    
    good_all = _load_good_responders(good_csv)
    proc_subjs = _list_processed_subjects(processed_dir)
    
    good_subjs = [s for s in good_all if s in set(proc_subjs) and s not in exclude]
    nongood_subjs = [s for s in proc_subjs if s not in set(good_all) and s not in exclude]

    _log_info(f"Analysis: Good={len(good_subjs)}, NonGood={len(nongood_subjs)}")
    _log_info(f"Analysis Window: {stat_tmin}s to {stat_tmax}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("target", "control"):
        # Determine Reference Grid from first available subject (with cropping)
        ref_freqs, ref_times = None, None
        for sid in good_subjs + nongood_subjs:
            p = _tfr_hep_path(processed_dir, sid, condition)
            if p.exists():
                t = _read_average_tfr(p)
                t.crop(tmin=stat_tmin, tmax=stat_tmax) # Crop reference
                ref_freqs, ref_times = t.freqs, t.times
                break
        
        if ref_freqs is None:
            _log_warn(f"No HEP-TFR data found for {condition}. Skipping.")
            continue

        for roi in roi_names:
            cands = roi_dict[roi]
            try:
                X_g, _, _ = _collect_tfr_group_matrix(
                    processed_dir, good_subjs, condition, cands, 
                    ref_freqs, ref_times, stat_tmin, stat_tmax
                )
                X_ng, _, _ = _collect_tfr_group_matrix(
                    processed_dir, nongood_subjs, condition, cands, 
                    ref_freqs, ref_times, stat_tmin, stat_tmax
                )
            except RuntimeError as e:
                _log_warn(str(e))
                continue

            # Stats
            _, p_vals, clusters = _run_cluster_2samp(X_g, X_ng, n_permutations, tail, seed)
            
            sig_mask = np.zeros_like(X_g[0], dtype=bool)
            for cl, p in zip(clusters, p_vals):
                if p < alpha:
                    if isinstance(cl, tuple): # indices
                        sig_mask[cl] = True
                    else:
                        sig_mask |= cl

            # Plot
            out_png = out_dir / f"hep_tfr_{roi}_{condition}_Good_vs_NonGood.png"
            _plot_tfr_group_triplet(
                roi, condition, ref_times, ref_freqs,
                X_g.mean(0), X_ng.mean(0), X_g.mean(0)-X_ng.mean(0),
                sig_mask, out_png
            )
            _log_info(f"Saved: {out_png}")

def main() -> None:
    p = argparse.ArgumentParser(description="HEP-TFR Group Statistics")
    p.add_argument("--good_csv", type=str, default=str(_repo_root() / "data" / "classification" / "good_responders_median.csv"))
    p.add_argument("--processed_dir", type=str, default=str(_default_processed_dir()))
    p.add_argument("--out_dir", type=str, default=str(_default_out_dir()))
    p.add_argument("--roi", nargs="+", default=["Frontal"])
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    
    # New args for cropping statistics window
    p.add_argument("--stat_tmin", type=float, default=-0.3, help="Analysis start time (s)")
    p.add_argument("--stat_tmax", type=float, default=0.8, help="Analysis end time (s)")
    
    args = p.parse_args()

    roi_dict = {
        "Frontal": ["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"],
        "Visual": ["Oz", "O1", "O2"],
        "Parietal": ["Pz", "P3", "P4", "CPz", "CP1", "CP2"],
    }

    run_analysis(
        good_csv=Path(args.good_csv),
        processed_dir=Path(args.processed_dir),
        out_dir=Path(args.out_dir),
        roi_dict=roi_dict,
        roi_names=args.roi,
        exclude=args.exclude,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        tail=0,
        seed=42,
        stat_tmin=float(args.stat_tmin),
        stat_tmax=float(args.stat_tmax),
    )

if __name__ == "__main__":
    main()