#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install requirements:
# pip install mne numpy pandas scipy matplotlib h5py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import mne


@dataclass(frozen=True)
class SubjectArtifacts:
    subject_id: str
    tfr_target_path: Path
    tfr_control_path: Path
    hep_target_path: Path
    hep_control_path: Path


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _processed_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "processed" / subject_id


def _default_out_dir() -> Path:
    return _repo_root() / "data" / "group_analysis_within_subject"


def _load_good_responders(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"good_responders_median.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("good_responders_median.csv must contain a 'session_id' column")

    ids = [str(x).strip() for x in df["session_id"].tolist()]
    ids = [x for x in ids if x]
    # Keep order; de-duplicate
    seen = set()
    uniq: List[str] = []
    for sid in ids:
        if sid in seen:
            continue
        uniq.append(sid)
        seen.add(sid)
    return uniq


def _resolve_artifacts(subject_id: str) -> Optional[SubjectArtifacts]:
    pdir = _processed_dir(subject_id)
    if not pdir.exists():
        _log_warn(f"Processed dir missing; skip: {pdir}")
        return None

    tfr_target = pdir / f"{subject_id}_tfr_target.h5"
    tfr_control = pdir / f"{subject_id}_tfr_control.h5"
    hep_target = pdir / f"{subject_id}_hep_target-ave.fif"
    hep_control = pdir / f"{subject_id}_hep_control-ave.fif"

    missing = [p for p in (tfr_target, tfr_control, hep_target, hep_control) if not p.exists()]
    if missing:
        _log_warn(
            f"Missing artifacts for {subject_id}; skip. Missing: "
            + ", ".join(m.name for m in missing)
        )
        return None

    return SubjectArtifacts(
        subject_id=subject_id,
        tfr_target_path=tfr_target,
        tfr_control_path=tfr_control,
        hep_target_path=hep_target,
        hep_control_path=hep_control,
    )


def _pick_roi_channels(info: mne.Info, roi_chs: List[str]) -> List[str]:
    existing = set(info["ch_names"])
    picked = [ch for ch in roi_chs if ch in existing]
    if not picked:
        raise ValueError(
            "None of the ROI channels exist in the data. "
            f"ROI={roi_chs} example existing={list(existing)[:10]}"
        )
    return picked


def _pick_common_roi_channels(info_a: mne.Info, info_b: mne.Info, roi_candidates: List[str]) -> List[str]:
    """Pick ROI channels that exist in *both* datasets.

    This keeps within-subject target/control comparisons consistent.
    """
    existing_a = set(info_a["ch_names"])
    existing_b = set(info_b["ch_names"])
    picked = [ch for ch in roi_candidates if ch in existing_a and ch in existing_b]
    if not picked:
        raise ValueError(
            "None of the ROI channels exist in both datasets. "
            f"ROI={roi_candidates} example existing_a={list(existing_a)[:10]} existing_b={list(existing_b)[:10]}"
        )
    return picked


def _roi_pick_mask(info: mne.Info, roi_candidates: List[str]) -> Tuple[List[str], np.ndarray]:
    picked = _pick_roi_channels(info, roi_candidates)
    picked_set = set(picked)
    mask = np.asarray([ch in picked_set for ch in roi_candidates], dtype=bool)
    return picked, mask


def _roi_common_pick_mask(
    info_a: mne.Info, info_b: mne.Info, roi_candidates: List[str]
) -> Tuple[List[str], np.ndarray]:
    picked = _pick_common_roi_channels(info_a, info_b, roi_candidates)
    picked_set = set(picked)
    mask = np.asarray([ch in picked_set for ch in roi_candidates], dtype=bool)
    return picked, mask


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


def _roi_mean_from_tfr(tfr: mne.time_frequency.AverageTFR, roi_chs: List[str]) -> np.ndarray:
    # Returns (n_freqs, n_times)
    picked = _pick_roi_channels(tfr.info, roi_chs)
    t = tfr.copy().pick(picked)
    data = cast(np.ndarray, t.data)
    return data.mean(axis=0)


def _roi_mean_and_mask_from_tfr(
    tfr: mne.time_frequency.AverageTFR, roi_candidates: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    picked, mask = _roi_pick_mask(tfr.info, roi_candidates)
    t = tfr.copy().pick(picked)
    data = cast(np.ndarray, t.data)
    return data.mean(axis=0), picked, mask


def _roi_mean_and_mask_common_from_tfrs(
    tfr_a: mne.time_frequency.AverageTFR,
    tfr_b: mne.time_frequency.AverageTFR,
    roi_candidates: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    picked, mask = _roi_common_pick_mask(tfr_a.info, tfr_b.info, roi_candidates)
    ta = tfr_a.copy().pick(picked)
    tb = tfr_b.copy().pick(picked)
    da = cast(np.ndarray, ta.data).mean(axis=0)
    db = cast(np.ndarray, tb.data).mean(axis=0)
    return da, db, picked, mask


def _align_1d_to_reference(
    x: np.ndarray,
    x_axis: np.ndarray,
    ref_axis: np.ndarray,
    *,
    label: str,
    subject_id: str,
) -> Optional[np.ndarray]:
    """Align a 1D array onto a reference axis using linear interpolation.

    If the subject axis does not fully cover the reference axis, returns None.
    """
    x = np.asarray(x, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    ref_axis = np.asarray(ref_axis, dtype=float)

    if x.ndim != 1 or x_axis.ndim != 1 or ref_axis.ndim != 1:
        raise ValueError("_align_1d_to_reference expects 1D inputs")
    if x.shape[0] != x_axis.shape[0]:
        raise ValueError(f"{label}: data/axis length mismatch")
    if x_axis.size < 2 or ref_axis.size < 2:
        raise ValueError(f"{label}: axis too small")

    if x_axis[0] > ref_axis[0] or x_axis[-1] < ref_axis[-1]:
        _log_warn(
            f"{label}: axis range does not cover reference; skip subject {subject_id}. "
            f"subj=[{x_axis[0]:.3f},{x_axis[-1]:.3f}] ref=[{ref_axis[0]:.3f},{ref_axis[-1]:.3f}]"
        )
        return None

    # np.interp requires increasing x coordinates.
    if np.any(np.diff(x_axis) <= 0):
        order = np.argsort(x_axis)
        x_axis = x_axis[order]
        x = x[order]

    return np.interp(ref_axis, x_axis, x).astype(float)


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


def _roi_mean_from_evoked(ev: mne.Evoked, roi_chs: List[str]) -> np.ndarray:
    # Returns (n_times,)
    picked = _pick_roi_channels(ev.info, roi_chs)
    e = ev.copy().pick(picked)
    return e.data.mean(axis=0)


def _roi_mean_and_mask_from_evoked(
    ev: mne.Evoked, roi_candidates: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    picked, mask = _roi_pick_mask(ev.info, roi_candidates)
    e = ev.copy().pick(picked)
    return e.data.mean(axis=0), picked, mask


def _roi_mean_and_mask_common_from_evokeds(
    ev_a: mne.Evoked,
    ev_b: mne.Evoked,
    roi_candidates: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    picked, mask = _roi_common_pick_mask(ev_a.info, ev_b.info, roi_candidates)
    ea = ev_a.copy().pick(picked)
    eb = ev_b.copy().pick(picked)
    return ea.data.mean(axis=0), eb.data.mean(axis=0), picked, mask


def _read_one_evoked(path: Path) -> mne.Evoked:
    obj = mne.read_evokeds(str(path), condition=0, verbose=False)
    # API differences: can return Evoked or list[Evoked]
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError(f"read_evokeds returned an empty list: {path}")
        ev = obj[0]
    else:
        ev = obj
    return cast(mne.Evoked, ev)


def _cluster_threshold_t(df: int, cluster_alpha: float, tail: int) -> float:
    if tail == 0:
        return float(stats.t.ppf(1.0 - cluster_alpha / 2.0, df))
    if tail == 1:
        return float(stats.t.ppf(1.0 - cluster_alpha, df))
    if tail == -1:
        return float(stats.t.ppf(cluster_alpha, df))
    raise ValueError("tail must be -1, 0, or 1")


def _run_cluster_1samp(
    X: np.ndarray,
    n_permutations: int,
    cluster_alpha: float,
    alpha: float,
    tail: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    # X shape: (n_subjects, ...)
    if X.ndim < 2:
        raise ValueError("X must have shape (n_subjects, n_features...) with at least 2 dims")

    n_subj = int(X.shape[0])
    df = n_subj - 1
    threshold = _cluster_threshold_t(df=df, cluster_alpha=cluster_alpha, tail=tail)

    T_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
        X,
        n_permutations=int(n_permutations),
        threshold=threshold,
        tail=int(tail),
        adjacency=None,
        out_type="mask",
        seed=int(seed),
        verbose=False,
    )

    # clusters is list of boolean masks
    sig_mask = np.zeros(X.shape[1:], dtype=bool)
    cluster_masks: List[np.ndarray] = []
    for mask, p in zip(clusters, cluster_pv):
        m = np.asarray(mask, dtype=bool)
        cluster_masks.append(m)
        if float(p) < float(alpha):
            sig_mask |= m

    return T_obs, np.asarray(cluster_pv, float), cluster_masks, sig_mask


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


def _cluster_time_range_1d(mask_t: np.ndarray, times_s: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (tmin_ms, tmax_ms) for a 1D cluster mask."""
    mask_t = np.asarray(mask_t, dtype=bool)
    if mask_t.ndim != 1:
        raise ValueError("1D cluster mask must be 1D (time)")
    if not mask_t.any():
        return None
    times_s = np.asarray(times_s, dtype=float)
    if mask_t.shape[0] != times_s.size:
        raise ValueError("1D cluster mask shape mismatch")
    idx = np.where(mask_t)[0]
    return float(times_s[idx[0]] * 1e3), float(times_s[idx[-1]] * 1e3)


def _plot_tfr_triplet(
    roi_name: str,
    roi_chs: List[str],
    times: np.ndarray,
    freqs: np.ndarray,
    target_mean: np.ndarray,
    control_mean: np.ndarray,
    diff_mean: np.ndarray,
    sig_mask: np.ndarray,
    out_png: Path,
) -> None:
    # Inputs are freq x time
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    vmax = float(np.nanpercentile(np.abs(diff_mean), 98))
    if vmax <= 0:
        vmax = float(np.nanmax(np.abs(diff_mean)) or 1.0)

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

    im0 = _imshow(axes[0], target_mean, "Target (mean)", cmap="viridis")
    im1 = _imshow(axes[1], control_mean, "Control (mean)", cmap="viridis")
    im2 = _imshow(axes[2], diff_mean, "Target - Control (mean)", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Overlay significance on diff
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
    cbar2.set_label("Δ Power")

    fig.suptitle(f"TFR ROI={roi_name} ({', '.join(roi_chs)})", y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hep_overlay(
    roi_name: str,
    roi_chs: List[str],
    times: np.ndarray,
    target_mean: np.ndarray,
    control_mean: np.ndarray,
    target_sem: np.ndarray,
    control_sem: np.ndarray,
    sig_mask: np.ndarray,
    out_png: Path,
) -> None:
    times_ms = times * 1e3

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    ax.plot(times_ms, target_mean * 1e6, color="red", lw=2, label="Target")
    ax.fill_between(
        times_ms,
        (target_mean - target_sem) * 1e6,
        (target_mean + target_sem) * 1e6,
        color="red",
        alpha=0.2,
        linewidth=0,
    )

    ax.plot(times_ms, control_mean * 1e6, color="blue", lw=2, label="Control")
    ax.fill_between(
        times_ms,
        (control_mean - control_sem) * 1e6,
        (control_mean + control_sem) * 1e6,
        color="blue",
        alpha=0.2,
        linewidth=0,
    )

    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"HEP ROI={roi_name} ({', '.join(roi_chs)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Significance bar at bottom
    if sig_mask.any():
        ymin, ymax = ax.get_ylim()
        bar_y = ymin + 0.05 * (ymax - ymin)
        # mark contiguous segments
        sig = sig_mask.astype(bool)
        idx = np.where(sig)[0]
        if idx.size > 0:
            # group contiguous
            starts = [idx[0]]
            ends: List[int] = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    ends.append(idx[i - 1])
                    starts.append(idx[i])
            ends.append(idx[-1])

            for s, e in zip(starts, ends):
                ax.hlines(
                    y=bar_y,
                    xmin=times_ms[s],
                    xmax=times_ms[e],
                    colors="k",
                    linewidth=6,
                )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    good_csv: Path,
    out_dir: Path,
    roi_dict: Dict[str, List[str]],
    roi_names: List[str],
    n_permutations: int,
    cluster_alpha: float,
    alpha: float,
    tail: int,
    seed: int,
) -> None:
    subjects = _load_good_responders(good_csv)
    _log_info(f"Good responders listed: {len(subjects)}")

    artifacts: List[SubjectArtifacts] = []
    for sid in subjects:
        a = _resolve_artifacts(sid)
        if a is not None:
            artifacts.append(a)

    if len(artifacts) < 2:
        raise RuntimeError("Not enough subjects with complete artifacts to run statistics")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []

    for roi_name in roi_names:
        if roi_name not in roi_dict:
            raise KeyError(f"ROI '{roi_name}' not found in roi_dict. Available={list(roi_dict.keys())}")
        roi_candidates = roi_dict[roi_name]

        # --- TFR ---
        tfr_diffs: List[np.ndarray] = []
        tfr_targets: List[np.ndarray] = []
        tfr_controls: List[np.ndarray] = []
        tfr_pick_masks: List[np.ndarray] = []
        tfr_used_subjects: List[str] = []
        times: Optional[np.ndarray] = None
        freqs: Optional[np.ndarray] = None

        for a in artifacts:
            tfr_t = _read_average_tfr(a.tfr_target_path)
            tfr_c = _read_average_tfr(a.tfr_control_path)

            try:
                data_t, data_c, _picked, pick_mask = _roi_mean_and_mask_common_from_tfrs(
                    tfr_t, tfr_c, roi_candidates
                )
            except Exception as e:
                _log_warn(f"TFR ROI pick failed for {a.subject_id}; skip: {e}")
                del tfr_t, tfr_c
                continue

            subj_times = np.asarray(tfr_t.times)
            subj_freqs = np.asarray(tfr_t.freqs)

            if times is None:
                times = subj_times
            if freqs is None:
                freqs = subj_freqs

            assert times is not None and freqs is not None
            subj_times_match = subj_times.shape == times.shape and np.allclose(subj_times, times)
            subj_freqs_match = subj_freqs.shape == freqs.shape and np.allclose(subj_freqs, freqs)
            if data_t.shape != (freqs.size, times.size) or not subj_times_match or not subj_freqs_match:
                aligned_t = _align_tfr_roi_to_reference(
                    data_ft=data_t,
                    freqs=subj_freqs,
                    times=subj_times,
                    ref_freqs=freqs,
                    ref_times=times,
                    subject_id=a.subject_id,
                )
                aligned_c = _align_tfr_roi_to_reference(
                    data_ft=data_c,
                    freqs=subj_freqs,
                    times=subj_times,
                    ref_freqs=freqs,
                    ref_times=times,
                    subject_id=a.subject_id,
                )
                if aligned_t is None or aligned_c is None:
                    _log_warn(f"Skipping subject for TFR due to grid mismatch: {a.subject_id}")
                    del tfr_t, tfr_c
                    continue
                data_t = aligned_t
                data_c = aligned_c

            tfr_targets.append(data_t)
            tfr_controls.append(data_c)
            tfr_diffs.append(data_t - data_c)
            tfr_pick_masks.append(np.asarray(pick_mask, dtype=bool))
            tfr_used_subjects.append(a.subject_id)

            # reduce memory
            del tfr_t, tfr_c

        if len(tfr_diffs) < 2:
            raise RuntimeError(f"Not enough subjects for TFR statistics after alignment for ROI={roi_name}")

        tfr_effective = _effective_channels_from_masks(
            roi_candidates,
            np.stack(tfr_pick_masks, axis=0) if tfr_pick_masks else np.zeros((0, len(roi_candidates)), bool),
        )
        _log_info(f"TFR ROI={roi_name}: effective_channels={tfr_effective}")

        X_tfr = np.stack(tfr_diffs, axis=0)  # subj x freq x time
        _log_info(f"TFR ROI={roi_name}: X shape={X_tfr.shape}")

        _, tfr_cluster_pv, tfr_cluster_masks, tfr_sig_mask = _run_cluster_1samp(
            X=X_tfr,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            alpha=alpha,
            tail=tail,
            seed=seed,
        )

        tfr_target_mean = np.mean(np.stack(tfr_targets, axis=0), axis=0)
        tfr_control_mean = np.mean(np.stack(tfr_controls, axis=0), axis=0)
        tfr_diff_mean = np.mean(X_tfr, axis=0)

        assert times is not None and freqs is not None
        out_png = out_dir / f"tfr_{roi_name}_target_control_diff.png"
        _plot_tfr_triplet(
            roi_name=roi_name,
            roi_chs=tfr_effective,
            times=times,
            freqs=freqs,
            target_mean=tfr_target_mean,
            control_mean=tfr_control_mean,
            diff_mean=tfr_diff_mean,
            sig_mask=tfr_sig_mask,
            out_png=out_png,
        )
        _log_info(f"Saved: {out_png}")

        # Summarize significant clusters (TFR)
        cluster_id = 0
        for i, (mask, p) in enumerate(zip(tfr_cluster_masks, tfr_cluster_pv), start=1):
            if float(p) >= float(alpha):
                continue
            r = _cluster_time_freq_ranges(mask_ft=mask, times_s=times, freqs_hz=freqs)
            if r is None:
                continue
            tmin_ms, tmax_ms, fmin_hz, fmax_hz = r
            cluster_id += 1
            summary_rows.append(
                {
                    "ROI": str(roi_name),
                    "Modality": "TFR",
                    "Cluster_ID": str(cluster_id),
                    "p_value": f"{float(p):.6g}",
                    "Time_Range": _format_range(tmin_ms, tmax_ms, "ms", decimals=1),
                    "Freq_Range": _format_range(fmin_hz, fmax_hz, "Hz", decimals=1),
                }
            )

        # --- HEP ---
        hep_targets: List[np.ndarray] = []
        hep_controls: List[np.ndarray] = []
        hep_diffs: List[np.ndarray] = []
        hep_pick_masks: List[np.ndarray] = []
        hep_used_subjects: List[str] = []
        hep_times: Optional[np.ndarray] = None

        for a in artifacts:
            ev_t = _read_one_evoked(a.hep_target_path)
            ev_c = _read_one_evoked(a.hep_control_path)

            try:
                y_t, y_c, _picked, pick_mask = _roi_mean_and_mask_common_from_evokeds(
                    ev_t, ev_c, roi_candidates
                )
            except Exception as e:
                _log_warn(f"HEP ROI pick failed for {a.subject_id}; skip: {e}")
                del ev_t, ev_c
                continue

            subj_hep_times = np.asarray(ev_t.times)
            if hep_times is None:
                hep_times = subj_hep_times

            assert hep_times is not None
            if y_t.shape[0] != hep_times.size or not np.allclose(subj_hep_times, hep_times):
                y_t_al = _align_1d_to_reference(
                    x=y_t,
                    x_axis=subj_hep_times,
                    ref_axis=hep_times,
                    label="HEP",
                    subject_id=a.subject_id,
                )
                y_c_al = _align_1d_to_reference(
                    x=y_c,
                    x_axis=subj_hep_times,
                    ref_axis=hep_times,
                    label="HEP",
                    subject_id=a.subject_id,
                )
                if y_t_al is None or y_c_al is None:
                    _log_warn(f"Skipping subject for HEP due to time grid mismatch: {a.subject_id}")
                    del ev_t, ev_c
                    continue
                y_t = y_t_al
                y_c = y_c_al

            hep_targets.append(y_t)
            hep_controls.append(y_c)
            hep_diffs.append(y_t - y_c)
            hep_pick_masks.append(np.asarray(pick_mask, dtype=bool))
            hep_used_subjects.append(a.subject_id)

            del ev_t, ev_c

        if len(hep_diffs) < 2:
            raise RuntimeError(f"Not enough subjects for HEP statistics after alignment for ROI={roi_name}")

        hep_effective = _effective_channels_from_masks(
            roi_candidates,
            np.stack(hep_pick_masks, axis=0) if hep_pick_masks else np.zeros((0, len(roi_candidates)), bool),
        )
        _log_info(f"HEP ROI={roi_name}: effective_channels={hep_effective}")

        X_hep = np.stack(hep_diffs, axis=0)  # subj x time
        _log_info(f"HEP ROI={roi_name}: X shape={X_hep.shape}")

        _, hep_cluster_pv, hep_cluster_masks, hep_sig_mask = _run_cluster_1samp(
            X=X_hep,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            alpha=alpha,
            tail=tail,
            seed=seed,
        )

        hep_target_mean = np.mean(np.stack(hep_targets, axis=0), axis=0)
        hep_control_mean = np.mean(np.stack(hep_controls, axis=0), axis=0)
        hep_target_sem = stats.sem(np.stack(hep_targets, axis=0), axis=0, nan_policy="omit")
        hep_control_sem = stats.sem(np.stack(hep_controls, axis=0), axis=0, nan_policy="omit")

        assert hep_times is not None
        out_png2 = out_dir / f"hep_{roi_name}_target_control.png"
        _plot_hep_overlay(
            roi_name=roi_name,
            roi_chs=hep_effective,
            times=hep_times,
            target_mean=hep_target_mean,
            control_mean=hep_control_mean,
            target_sem=hep_target_sem,
            control_sem=hep_control_sem,
            sig_mask=hep_sig_mask,
            out_png=out_png2,
        )
        _log_info(f"Saved: {out_png2}")

        # Summarize significant clusters (HEP)
        cluster_id = 0
        for i, (mask, p) in enumerate(zip(hep_cluster_masks, hep_cluster_pv), start=1):
            if float(p) >= float(alpha):
                continue
            r = _cluster_time_range_1d(mask_t=mask, times_s=hep_times)
            if r is None:
                continue
            tmin_ms, tmax_ms = r
            cluster_id += 1
            summary_rows.append(
                {
                    "ROI": str(roi_name),
                    "Modality": "HEP",
                    "Cluster_ID": str(cluster_id),
                    "p_value": f"{float(p):.6g}",
                    "Time_Range": _format_range(tmin_ms, tmax_ms, "ms", decimals=1),
                    "Freq_Range": "",
                }
            )

        # Save minimal stats summary
        np.savez_compressed(
            out_dir / f"stats_{roi_name}.npz",
            roi=str(roi_name),
            roi_candidates=np.asarray(roi_candidates, dtype=str),
            tfr_roi_channels=np.asarray(tfr_effective, dtype=str),
            hep_roi_channels=np.asarray(hep_effective, dtype=str),
            tfr_subjects=np.asarray(tfr_used_subjects, dtype=str),
            hep_subjects=np.asarray(hep_used_subjects, dtype=str),
            tfr_roi_pick_mask=np.stack(tfr_pick_masks, axis=0)
            if len(tfr_pick_masks)
            else np.zeros((0, len(roi_candidates)), dtype=bool),
            hep_roi_pick_mask=np.stack(hep_pick_masks, axis=0)
            if len(hep_pick_masks)
            else np.zeros((0, len(roi_candidates)), dtype=bool),
            tfr_cluster_p_values=np.asarray(tfr_cluster_pv, float),
            tfr_clusters=np.stack(tfr_cluster_masks, axis=0) if len(tfr_cluster_masks) else np.zeros((0,) + X_tfr.shape[1:], dtype=bool),
            tfr_sig_mask=tfr_sig_mask,
            hep_cluster_p_values=np.asarray(hep_cluster_pv, float),
            hep_clusters=np.stack(hep_cluster_masks, axis=0) if len(hep_cluster_masks) else np.zeros((0,) + X_hep.shape[1:], dtype=bool),
            hep_sig_mask=hep_sig_mask,
        )

    # Write summary CSV (significant clusters only)
    summary_path = out_dir / "statistics_summary.csv"
    df = pd.DataFrame(summary_rows, columns=["ROI", "Modality", "Cluster_ID", "p_value", "Time_Range", "Freq_Range"])
    df.to_csv(summary_path, index=False)
    _log_info(f"Saved: {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Within-subject condition statistics for Good Responders. "
            "Loads precomputed TFR/HEP intermediate files (no recomputation)."
        )
    )

    p.add_argument(
        "--good_csv",
        type=str,
        default=str(_repo_root() / "data" / "classification" / "good_responders_median.csv"),
        help="Path to good_responders_median.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_default_out_dir()),
        help="Output directory for group stats figures",
    )

    p.add_argument(
        "--roi",
        type=str,
        nargs="+",
        default=["Frontal"],
        help="ROI name(s) from the built-in roi_dict (default: Frontal)",
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

    # ROI definition lives here (per spec: after loading, exploratory).
    # ROI definitions are written as *candidates*.
    # Actual channels are filtered per-subject to avoid failures due to cap/layout differences.
    roi_dict: Dict[str, List[str]] = {
        "Frontal": ["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"],
        "Visual": ["Oz", "O1", "O2"],
        "Parietal": ["Pz", "P3", "P4", "CPz", "CP1", "CP2"],
    }

    run_analysis(
        good_csv=Path(args.good_csv),
        out_dir=Path(args.out_dir),
        roi_dict=roi_dict,
        roi_names=[str(x) for x in args.roi],
        n_permutations=int(args.n_permutations),
        cluster_alpha=float(args.cluster_alpha),
        alpha=float(args.alpha),
        tail=int(args.tail),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
