#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Between-subject group statistics for TFR (Good vs Non-Good).

This script performs independent 2-sample permutation cluster tests on TFR data,
comparing Good Responders vs Non-Good Responders for each condition (Target/Control).

No HEP processing is included.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import mne


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_processed_dir() -> Path:
    return _repo_root() / "data" / "processed"


def _default_out_dir() -> Path:
    return _repo_root() / "data" / "group_analysis_between_subjects_tfr"


# ---------------------------------------------------------------------------
# Good/Non-Good subject handling (from src/05_group_statistics_hep.py)
# ---------------------------------------------------------------------------


def _load_good_responders(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"good_responders_median.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("good_responders_median.csv must contain a 'session_id' column")

    ids = [str(x).strip() for x in df["session_id"].tolist()]
    ids = [x for x in ids if x]

    seen: Set[str] = set()
    uniq: List[str] = []
    for sid in ids:
        if sid in seen:
            continue
        uniq.append(sid)
        seen.add(sid)
    return uniq


def _list_processed_subjects(processed_dir: Path) -> List[str]:
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    if not processed_dir.is_dir():
        raise NotADirectoryError(f"processed_dir is not a directory: {processed_dir}")

    subjects: List[str] = []
    for p in processed_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            subjects.append(p.name)
    subjects.sort()
    return subjects


def _apply_exclude(
    good_subjects: List[str],
    nongood_subjects: List[str],
    exclude: Sequence[str],
) -> Tuple[List[str], List[str]]:
    if not exclude:
        return good_subjects, nongood_subjects

    exclude_set = {str(x).strip() for x in exclude if str(x).strip()}

    good_set = set(good_subjects)
    nongood_set = set(nongood_subjects)

    for sid in sorted(exclude_set):
        if sid not in good_set and sid not in nongood_set:
            _log_warn(f"Exclude ID not found in either group; ignore: {sid}")

    good_out = [sid for sid in good_subjects if sid not in exclude_set]
    nongood_out = [sid for sid in nongood_subjects if sid not in exclude_set]
    return good_out, nongood_out


# ---------------------------------------------------------------------------
# TFR data loading and ROI processing (from src/04_condition_statistics.py)
# ---------------------------------------------------------------------------


def _read_average_tfr(path: Path) -> mne.time_frequency.AverageTFR:
    obj = mne.time_frequency.read_tfrs(str(path))
    # MNE API differences: sometimes returns list[AverageTFR], sometimes AverageTFR.
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError(f"read_tfrs returned an empty list: {path}")
        tfr = obj[0]
    else:
        tfr = obj
    if not isinstance(tfr, mne.time_frequency.AverageTFR):
        raise TypeError(f"Expected AverageTFR from {path} but got {type(tfr)}")
    return tfr


def _pick_roi_channels(info: mne.Info, roi_chs: List[str]) -> List[str]:
    existing = set(info["ch_names"])
    picked = [ch for ch in roi_chs if ch in existing]
    if not picked:
        raise ValueError(
            "None of the ROI channels exist in the data. "
            f"ROI={roi_chs} example existing={list(existing)[:10]}"
        )
    return picked


def _roi_pick_mask(info: mne.Info, roi_candidates: List[str]) -> Tuple[List[str], np.ndarray]:
    """Pick existing ROI channels and return both names and a mask over candidates."""
    picked = _pick_roi_channels(info, roi_candidates)
    picked_set = set(picked)
    mask = np.asarray([ch in picked_set for ch in roi_candidates], dtype=bool)
    return picked, mask


def _roi_mean_and_mask_from_tfr(
    tfr: mne.time_frequency.AverageTFR, roi_candidates: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return ROI-mean TFR data (freq x time), picked channel names, and pick mask."""
    picked, mask = _roi_pick_mask(tfr.info, roi_candidates)
    t = tfr.copy().pick(picked)
    data = cast(np.ndarray, t.data)
    return data.mean(axis=0), picked, mask


def _align_tfr_roi_to_reference(
    data_ft: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    ref_freqs: np.ndarray,
    ref_times: np.ndarray,
    *,
    subject_id: str,
) -> Optional[np.ndarray]:
    """Align ROI-mean TFR data (freq x time) to a reference freq/time grid.

    Returns aligned (ref_freq x ref_time) or None if the subject grid cannot cover the reference.
    """
    data_ft = np.asarray(data_ft, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    times = np.asarray(times, dtype=float)
    ref_freqs = np.asarray(ref_freqs, dtype=float)
    ref_times = np.asarray(ref_times, dtype=float)

    if data_ft.ndim != 2:
        raise ValueError("TFR ROI data must be 2D (freq x time)")
    if data_ft.shape != (freqs.size, times.size):
        raise ValueError("TFR ROI data shape mismatch with freqs/times")

    # Ensure the subject grid covers the reference grid (avoid endpoint clamping extrapolation).
    if freqs[0] > ref_freqs[0] or freqs[-1] < ref_freqs[-1]:
        _log_warn(
            "TFR: freqs range does not cover reference; skip subject "
            f"{subject_id}. subj=[{freqs[0]:.1f},{freqs[-1]:.1f}] ref=[{ref_freqs[0]:.1f},{ref_freqs[-1]:.1f}]"
        )
        return None
    if times[0] > ref_times[0] or times[-1] < ref_times[-1]:
        _log_warn(
            "TFR: times range does not cover reference; skip subject "
            f"{subject_id}. subj=[{times[0]:.3f},{times[-1]:.3f}] ref=[{ref_times[0]:.3f},{ref_times[-1]:.3f}]"
        )
        return None

    # Sort axes if needed
    if np.any(np.diff(freqs) <= 0):
        order_f = np.argsort(freqs)
        freqs = freqs[order_f]
        data_ft = data_ft[order_f, :]
    if np.any(np.diff(times) <= 0):
        order_t = np.argsort(times)
        times = times[order_t]
        data_ft = data_ft[:, order_t]

    # 1) Align along frequency for each time point
    freqs_match = freqs.shape == ref_freqs.shape and np.allclose(freqs, ref_freqs)
    if not freqs_match:
        _log_warn(f"TFR grid mismatch (freq); interpolating subject {subject_id} onto reference")
        tmp = np.empty((ref_freqs.size, times.size), dtype=float)
        for ti in range(times.size):
            tmp[:, ti] = np.interp(ref_freqs, freqs, data_ft[:, ti])
    else:
        tmp = data_ft

    # 2) Align along time for each frequency
    times_match = times.shape == ref_times.shape and np.allclose(times, ref_times)
    if not times_match:
        _log_warn(f"TFR grid mismatch (time); interpolating subject {subject_id} onto reference")
        out = np.empty((ref_freqs.size, ref_times.size), dtype=float)
        for fi in range(ref_freqs.size):
            out[fi, :] = np.interp(ref_times, times, tmp[fi, :])
    else:
        out = tmp

    return out


def _effective_channels_from_masks(roi_candidates: List[str], masks: object) -> List[str]:
    m = np.asarray(masks)
    if m.size == 0:
        return []
    if m.ndim == 1:
        any_mask = m.astype(bool)
    elif m.ndim == 2:
        any_mask = np.any(m.astype(bool), axis=0)
    else:
        raise ValueError("Masks must be 1D (candidates) or 2D (subjects x candidates)")
    if any_mask.size != len(roi_candidates):
        raise ValueError("ROI candidate/mask length mismatch")
    return [ch for ch, ok in zip(roi_candidates, any_mask.tolist()) if ok]


# ---------------------------------------------------------------------------
# TFR file path resolution
# ---------------------------------------------------------------------------


def _tfr_path(processed_dir: Path, subject_id: str, condition: str) -> Path:
    pdir = processed_dir / subject_id
    if condition == "target":
        return pdir / f"{subject_id}_tfr_target.h5"
    if condition == "control":
        return pdir / f"{subject_id}_tfr_control.h5"
    raise ValueError("condition must be 'target' or 'control'")


# ---------------------------------------------------------------------------
# Reference grid determination
# ---------------------------------------------------------------------------


def _find_reference_tfr_grid(
    processed_dir: Path,
    subject_ids: Sequence[str],
    condition: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find reference freqs and times from the first readable subject."""
    for sid in subject_ids:
        path = _tfr_path(processed_dir, sid, condition)
        if not path.exists():
            continue
        try:
            tfr = _read_average_tfr(path)
        except Exception as e:
            _log_warn(f"Failed to read TFR for reference; skip {sid}: {e}")
            continue
        return np.asarray(tfr.freqs, dtype=float), np.asarray(tfr.times, dtype=float)
    return None


# ---------------------------------------------------------------------------
# Group matrix collection for TFR
# ---------------------------------------------------------------------------


def _collect_tfr_group_matrix(
    *,
    processed_dir: Path,
    subject_ids: Sequence[str],
    condition: str,
    roi_candidates: List[str],
    ref_freqs: np.ndarray,
    ref_times: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Collect TFR ROI-mean data for a group of subjects.

    Returns:
        X: (n_subjects, n_freqs, n_times)
        used: list of subject IDs actually used
        pick_masks: (n_subjects, n_candidates) bool array
    """
    data_list: List[np.ndarray] = []
    used: List[str] = []
    pick_masks: List[np.ndarray] = []

    for sid in subject_ids:
        path = _tfr_path(processed_dir, sid, condition)
        if not path.exists():
            _log_warn(f"Missing TFR {condition} for {sid}; skip")
            continue

        try:
            tfr = _read_average_tfr(path)
            data_ft, _, pick_mask = _roi_mean_and_mask_from_tfr(tfr, roi_candidates)
            subj_freqs = np.asarray(tfr.freqs, dtype=float)
            subj_times = np.asarray(tfr.times, dtype=float)
        except Exception as e:
            _log_warn(f"Failed to load/compute TFR ROI mean for {sid} ({condition}); skip: {e}")
            continue

        # Check if alignment is needed
        freqs_match = subj_freqs.shape == ref_freqs.shape and np.allclose(subj_freqs, ref_freqs)
        times_match = subj_times.shape == ref_times.shape and np.allclose(subj_times, ref_times)

        if not freqs_match or not times_match:
            aligned = _align_tfr_roi_to_reference(
                data_ft=data_ft,
                freqs=subj_freqs,
                times=subj_times,
                ref_freqs=ref_freqs,
                ref_times=ref_times,
                subject_id=sid,
            )
            if aligned is None:
                _log_warn(f"Skipping subject for TFR due to grid mismatch: {sid}")
                continue
            data_ft = aligned

        data_list.append(np.asarray(data_ft, float))
        used.append(sid)
        pick_masks.append(np.asarray(pick_mask, dtype=bool))

    if not data_list:
        raise RuntimeError(f"No usable subjects after alignment for TFR condition={condition}")

    X = np.stack(data_list, axis=0)
    picked_mask = (
        np.stack(pick_masks, axis=0)
        if pick_masks
        else np.zeros((0, len(roi_candidates)), dtype=bool)
    )
    return X, used, picked_mask


# ---------------------------------------------------------------------------
# Statistical testing (2-sample cluster permutation for TFR)
# ---------------------------------------------------------------------------


def _cluster_threshold_t(df: int, cluster_alpha: float, tail: int) -> float:
    if tail == 0:
        return float(stats.t.ppf(1.0 - cluster_alpha / 2.0, df))
    if tail == 1:
        return float(stats.t.ppf(1.0 - cluster_alpha, df))
    if tail == -1:
        return float(stats.t.ppf(cluster_alpha, df))
    raise ValueError("tail must be -1, 0, or 1")


def _run_cluster_2samp_tfr(
    X_good: np.ndarray,
    X_nongood: np.ndarray,
    n_permutations: int,
    cluster_alpha: float,
    alpha: float,
    tail: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Run 2-sample permutation cluster test for TFR data (n_subjects x n_freqs x n_times)."""
    if X_good.ndim != 3 or X_nongood.ndim != 3:
        raise ValueError("X must have shape (n_subjects, n_freqs, n_times)")
    if X_good.shape[1:] != X_nongood.shape[1:]:
        raise ValueError("Good/Non-Good feature shapes do not match")

    n1 = int(X_good.shape[0])
    n2 = int(X_nongood.shape[0])
    if n1 < 2 or n2 < 2:
        raise RuntimeError(f"Not enough subjects per group for two-sample test (n1={n1}, n2={n2})")

    df = n1 + n2 - 2
    threshold = _cluster_threshold_t(df=df, cluster_alpha=cluster_alpha, tail=tail)

    stat_fun = getattr(mne.stats, "ttest_ind_no_p", None)
    if stat_fun is None:
        raise RuntimeError("mne.stats.ttest_ind_no_p is required for two-sample cluster test")

    T_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_test(
        [X_good, X_nongood],
        n_permutations=int(n_permutations),
        threshold=threshold,
        tail=int(tail),
        adjacency=None,
        out_type="mask",
        seed=int(seed),
        stat_fun=stat_fun,
        verbose=False,
    )

    n_freqs = int(X_good.shape[1])
    n_times = int(X_good.shape[2])
    sig_mask = np.zeros((n_freqs, n_times), dtype=bool)
    cluster_masks: List[np.ndarray] = []

    for cl, p in zip(clusters, cluster_pv):
        m = _cluster_to_2d_mask(cl, n_freqs, n_times)
        cluster_masks.append(m)
        if float(p) < float(alpha):
            sig_mask |= m

    return T_obs, np.asarray(cluster_pv, float), cluster_masks, sig_mask


def _cluster_to_2d_mask(cluster: object, n_freqs: int, n_times: int) -> np.ndarray:
    """Normalize MNE cluster representation to a 2D boolean mask (freq x time)."""
    if isinstance(cluster, np.ndarray):
        arr = np.asarray(cluster)
        if arr.dtype == bool:
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                if arr.shape != (n_freqs, n_times):
                    raise ValueError("Cluster mask shape mismatch")
                return arr.astype(bool)
            elif arr.ndim == 1:
                # Flattened mask
                if arr.size != n_freqs * n_times:
                    raise ValueError("Cluster mask size mismatch")
                return arr.reshape(n_freqs, n_times).astype(bool)

        # Index array
        idx = np.asarray(arr, dtype=int).ravel()
        mask = np.zeros(n_freqs * n_times, dtype=bool)
        idx = idx[(idx >= 0) & (idx < n_freqs * n_times)]
        mask[idx] = True
        return mask.reshape(n_freqs, n_times)

    if isinstance(cluster, (list, tuple)):
        if len(cluster) == 2:
            # (freq_indices, time_indices) format
            freq_idx = np.asarray(cluster[0], dtype=int).ravel()
            time_idx = np.asarray(cluster[1], dtype=int).ravel()
            mask = np.zeros((n_freqs, n_times), dtype=bool)
            valid = (
                (freq_idx >= 0) & (freq_idx < n_freqs) &
                (time_idx >= 0) & (time_idx < n_times)
            )
            mask[freq_idx[valid], time_idx[valid]] = True
            return mask
        elif len(cluster) == 1:
            return _cluster_to_2d_mask(cluster[0], n_freqs, n_times)

    arr = np.asarray(cluster)
    if arr.dtype == object and arr.size == 1:
        return _cluster_to_2d_mask(arr.item(), n_freqs, n_times)

    raise TypeError(f"Unsupported cluster type: {type(cluster)}")


# ---------------------------------------------------------------------------
# Cluster summary utilities
# ---------------------------------------------------------------------------


def _format_range(start: float, end: float, unit: str, decimals: int = 1) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(start)}-{fmt.format(end)} {unit}"


def _cluster_time_freq_ranges(
    mask_ft: np.ndarray,
    times_s: np.ndarray,
    freqs_hz: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (tmin_ms, tmax_ms, fmin_hz, fmax_hz) for a TFR cluster mask."""
    mask_ft = np.asarray(mask_ft, dtype=bool)
    if mask_ft.ndim != 2:
        raise ValueError("TFR cluster mask must be 2D (freq x time)")
    if not mask_ft.any():
        return None

    times_s = np.asarray(times_s, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if mask_ft.shape != (freqs_hz.size, times_s.size):
        raise ValueError("TFR cluster mask shape mismatch")

    time_idx = np.where(mask_ft.any(axis=0))[0]
    freq_idx = np.where(mask_ft.any(axis=1))[0]
    tmin_ms = float(times_s[time_idx[0]] * 1e3)
    tmax_ms = float(times_s[time_idx[-1]] * 1e3)
    fmin_hz = float(freqs_hz[freq_idx[0]])
    fmax_hz = float(freqs_hz[freq_idx[-1]])
    return tmin_ms, tmax_ms, fmin_hz, fmax_hz


# ---------------------------------------------------------------------------
# Visualization (3-panel TFR heatmap: Good, Non-Good, Diff)
# ---------------------------------------------------------------------------


def _plot_tfr_group_triplet(
    *,
    roi_name: str,
    roi_chs: List[str],
    condition: str,
    times: np.ndarray,
    freqs: np.ndarray,
    good_mean: np.ndarray,
    nongood_mean: np.ndarray,
    diff_mean: np.ndarray,
    sig_mask: np.ndarray,
    out_png: Path,
) -> None:
    """Plot 3-panel TFR: Good mean, Non-Good mean, Difference (Good - NonGood)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    vmax_diff = float(np.nanpercentile(np.abs(diff_mean), 98))
    if vmax_diff <= 0:
        vmax_diff = float(np.nanmax(np.abs(diff_mean)) or 1.0)

    def _imshow(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        return im

    condition_label = "Target" if condition == "target" else "Control"

    im0 = _imshow(axes[0], good_mean, f"Good ({condition_label})", cmap="viridis")
    im1 = _imshow(axes[1], nongood_mean, f"Non-Good ({condition_label})", cmap="viridis")
    im2 = _imshow(
        axes[2], diff_mean, f"Good - NonGood ({condition_label})",
        cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff
    )

    # Overlay significance on diff panel
    if sig_mask.any():
        axes[2].contour(
            times,
            freqs,
            sig_mask.astype(float),
            levels=[0.5],
            colors=["k"],
            linewidths=1.5,
        )

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046)
    cbar0.set_label("Power")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046)
    cbar1.set_label("Power")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046)
    cbar2.set_label("\u0394 Power")

    fig.suptitle(f"TFR ROI={roi_name} ({', '.join(roi_chs)})", y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    good_csv: Path,
    processed_dir: Path,
    out_dir: Path,
    roi_dict: Dict[str, List[str]],
    roi_names: Sequence[str],
    exclude: Sequence[str],
    n_permutations: int,
    cluster_alpha: float,
    alpha: float,
    tail: int,
    seed: int,
) -> None:
    good_listed = _load_good_responders(good_csv)
    processed_subjects = _list_processed_subjects(processed_dir)

    processed_set = set(processed_subjects)

    good_subjects = [sid for sid in good_listed if sid in processed_set]
    missing_good = [sid for sid in good_listed if sid not in processed_set]
    for sid in missing_good:
        _log_warn(f"Good listed but not found in processed_dir; skip: {sid}")

    good_set = set(good_subjects)
    nongood_subjects = [sid for sid in processed_subjects if sid not in good_set]

    good_subjects, nongood_subjects = _apply_exclude(good_subjects, nongood_subjects, exclude)

    _log_info(f"Good responders (usable): {len(good_subjects)}")
    _log_info(f"Non-Good responders (usable): {len(nongood_subjects)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []

    roi_names = [str(x) for x in roi_names]
    for roi_name in roi_names:
        if roi_name not in roi_dict:
            raise KeyError(f"ROI '{roi_name}' not found. Available={list(roi_dict.keys())}")

    for condition in ("target", "control"):
        condition_label = "Target" if condition == "target" else "Control"

        for roi_name in roi_names:
            roi_candidates = roi_dict[roi_name]

            # Determine reference grid
            ref_grid = _find_reference_tfr_grid(
                processed_dir,
                list(good_subjects) + list(nongood_subjects),
                condition,
            )
            if ref_grid is None:
                raise RuntimeError(f"Could not determine reference TFR grid for condition={condition}")
            ref_freqs, ref_times = ref_grid

            # Collect group matrices
            X_good, good_used, good_pick_mask = _collect_tfr_group_matrix(
                processed_dir=processed_dir,
                subject_ids=good_subjects,
                condition=condition,
                roi_candidates=roi_candidates,
                ref_freqs=ref_freqs,
                ref_times=ref_times,
            )
            X_nongood, nongood_used, nongood_pick_mask = _collect_tfr_group_matrix(
                processed_dir=processed_dir,
                subject_ids=nongood_subjects,
                condition=condition,
                roi_candidates=roi_candidates,
                ref_freqs=ref_freqs,
                ref_times=ref_times,
            )

            # Effective channels
            used_union_mask = np.any(
                np.vstack([good_pick_mask, nongood_pick_mask]).astype(bool),
                axis=0,
            )
            roi_effective = _effective_channels_from_masks(roi_candidates, used_union_mask)
            if not roi_effective:
                raise RuntimeError(
                    f"No effective ROI channels after filtering: ROI={roi_name} condition={condition_label}"
                )

            _log_info(
                f"TFR {condition_label} ROI={roi_name}: effective_channels={roi_effective}"
            )
            _log_info(
                f"TFR {condition_label} ROI={roi_name}: X_good={X_good.shape} X_nongood={X_nongood.shape}"
            )

            # Run cluster test
            T_obs, cluster_pv, cluster_masks, sig_mask = _run_cluster_2samp_tfr(
                X_good=X_good,
                X_nongood=X_nongood,
                n_permutations=n_permutations,
                cluster_alpha=cluster_alpha,
                alpha=alpha,
                tail=tail,
                seed=seed,
            )

            # Compute means and SEM
            good_mean = np.mean(X_good, axis=0)
            nongood_mean = np.mean(X_nongood, axis=0)
            diff_mean = good_mean - nongood_mean
            good_sem = stats.sem(X_good, axis=0, nan_policy="omit")
            nongood_sem = stats.sem(X_nongood, axis=0, nan_policy="omit")

            # Plot
            out_png = out_dir / f"tfr_{roi_name}_{condition}_Good_vs_NonGood.png"
            _plot_tfr_group_triplet(
                roi_name=roi_name,
                roi_chs=roi_effective,
                condition=condition,
                times=ref_times,
                freqs=ref_freqs,
                good_mean=good_mean,
                nongood_mean=nongood_mean,
                diff_mean=diff_mean,
                sig_mask=sig_mask,
                out_png=out_png,
            )
            _log_info(f"Saved: {out_png}")

            # Summarize significant clusters
            cluster_id = 0
            for mask, p in zip(cluster_masks, cluster_pv):
                if float(p) >= float(alpha):
                    continue
                r = _cluster_time_freq_ranges(mask_ft=mask, times_s=ref_times, freqs_hz=ref_freqs)
                if r is None:
                    continue
                tmin_ms, tmax_ms, fmin_hz, fmax_hz = r
                cluster_id += 1
                summary_rows.append(
                    {
                        "ROI": str(roi_name),
                        "Condition": str(condition_label),
                        "n_good": str(len(good_used)),
                        "n_nongood": str(len(nongood_used)),
                        "Cluster_ID": str(cluster_id),
                        "p_value": f"{float(p):.6g}",
                        "Time_Range": _format_range(tmin_ms, tmax_ms, "ms", decimals=1),
                        "Freq_Range": _format_range(fmin_hz, fmax_hz, "Hz", decimals=1),
                    }
                )

            # Save NPZ
            np.savez_compressed(
                out_dir / f"stats_{roi_name}_{condition}.npz",
                condition=str(condition_label),
                roi=str(roi_name),
                roi_channels=np.asarray(roi_effective, dtype=str),
                roi_candidates=np.asarray(roi_candidates, dtype=str),
                good_roi_pick_mask=np.asarray(good_pick_mask, dtype=bool),
                nongood_roi_pick_mask=np.asarray(nongood_pick_mask, dtype=bool),
                freqs_hz=np.asarray(ref_freqs, float),
                times_s=np.asarray(ref_times, float),
                good_subjects=np.asarray(good_used, dtype=str),
                nongood_subjects=np.asarray(nongood_used, dtype=str),
                X_good=np.asarray(X_good, float),
                X_nongood=np.asarray(X_nongood, float),
                good_mean=np.asarray(good_mean, float),
                nongood_mean=np.asarray(nongood_mean, float),
                good_sem=np.asarray(good_sem, float),
                nongood_sem=np.asarray(nongood_sem, float),
                T_obs=np.asarray(T_obs, float),
                cluster_p_values=np.asarray(cluster_pv, float),
                clusters=np.stack(cluster_masks, axis=0)
                if len(cluster_masks)
                else np.zeros((0,) + X_good.shape[1:], dtype=bool),
                sig_mask=np.asarray(sig_mask, bool),
                n_permutations=int(n_permutations),
                cluster_alpha=float(cluster_alpha),
                alpha=float(alpha),
                tail=int(tail),
                seed=int(seed),
            )

    # Write summary CSV
    summary_path = out_dir / "statistics_summary.csv"
    df = pd.DataFrame(
        summary_rows,
        columns=["ROI", "Condition", "n_good", "n_nongood", "Cluster_ID", "p_value", "Time_Range", "Freq_Range"],
    )
    df.to_csv(summary_path, index=False)
    _log_info(f"Saved: {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Between-subject group statistics for TFR (Good vs Non-Good). "
            "Loads precomputed TFR files (no recomputation). "
            "Performs independent 2-sample permutation cluster tests for each condition."
        )
    )

    p.add_argument(
        "--good_csv",
        type=str,
        default=str(_repo_root() / "data" / "classification" / "good_responders_median.csv"),
        help="Path to good_responders_median.csv",
    )
    p.add_argument(
        "--processed_dir",
        type=str,
        default=str(_default_processed_dir()),
        help="Path to processed directory containing subject subfolders",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_default_out_dir()),
        help="Output directory for between-subject TFR group stats",
    )

    p.add_argument(
        "--roi",
        type=str,
        nargs="+",
        default=["Frontal"],
        help="ROI name(s) from the built-in roi_dict (default: Frontal)",
    )

    p.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Subject IDs to exclude (space-separated list)",
    )

    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument(
        "--cluster_alpha",
        type=float,
        default=0.05,
        help="Cluster-forming threshold alpha (t-threshold derived from this)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Cluster-level significance alpha")
    p.add_argument("--tail", type=int, default=0, help="-1, 0, or 1")
    p.add_argument("--seed", type=int, default=0)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    # ROI definitions are written as *candidates*.
    # Actual channels are filtered per-subject to avoid failures due to cap/layout differences.
    roi_dict: Dict[str, List[str]] = {
        "Frontal": ["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"],
        "Visual": ["Oz", "O1", "O2"],
        "Parietal": ["Pz", "P3", "P4", "CPz", "CP1", "CP2"],
    }

    run_analysis(
        good_csv=Path(args.good_csv),
        processed_dir=Path(args.processed_dir),
        out_dir=Path(args.out_dir),
        roi_dict=roi_dict,
        roi_names=[str(x) for x in args.roi],
        exclude=[str(x) for x in args.exclude],
        n_permutations=int(args.n_permutations),
        cluster_alpha=float(args.cluster_alpha),
        alpha=float(args.alpha),
        tail=int(args.tail),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
