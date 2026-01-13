#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    return _repo_root() / "data" / "group_analysis_between_subjects"


def _load_good_responders(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"good_responders.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("good_responders.csv must contain a 'session_id' column")

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
    """Pick existing ROI channels and return both names and a mask over candidates.

    This provides a pickle-free way to persist per-subject channel usage.
    """
    picked = _pick_roi_channels(info, roi_candidates)
    picked_set = set(picked)
    mask = np.asarray([ch in picked_set for ch in roi_candidates], dtype=bool)
    return picked, mask


def _read_one_evoked(path: Path) -> mne.Evoked:
    obj = mne.read_evokeds(str(path), condition=0, verbose=False)
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError(f"read_evokeds returned an empty list: {path}")
        ev = obj[0]
    else:
        ev = obj
    return cast(mne.Evoked, ev)


def _roi_mean_from_evoked(ev: mne.Evoked, roi_chs: List[str]) -> np.ndarray:
    picked = _pick_roi_channels(ev.info, roi_chs)
    e = ev.copy().pick(picked)
    return e.data.mean(axis=0)


def _roi_mean_and_mask_from_evoked(
    ev: mne.Evoked, roi_candidates: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    picked, mask = _roi_pick_mask(ev.info, roi_candidates)
    e = ev.copy().pick(picked)
    y = e.data.mean(axis=0)
    return y, picked, mask


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

    if np.any(np.diff(x_axis) <= 0):
        order = np.argsort(x_axis)
        x_axis = x_axis[order]
        x = x[order]

    return np.interp(ref_axis, x_axis, x).astype(float)


def _cluster_threshold_t(df: int, cluster_alpha: float, tail: int) -> float:
    if tail == 0:
        return float(stats.t.ppf(1.0 - cluster_alpha / 2.0, df))
    if tail == 1:
        return float(stats.t.ppf(1.0 - cluster_alpha, df))
    if tail == -1:
        return float(stats.t.ppf(cluster_alpha, df))
    raise ValueError("tail must be -1, 0, or 1")


def _run_cluster_2samp(
    X_good: np.ndarray,
    X_nongood: np.ndarray,
    n_permutations: int,
    cluster_alpha: float,
    alpha: float,
    tail: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    if X_good.ndim < 2 or X_nongood.ndim < 2:
        raise ValueError("X must have shape (n_subjects, n_features...) with at least 2 dims")
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

    n_times = int(X_good.shape[1])
    sig_mask = np.zeros((n_times,), dtype=bool)
    cluster_masks: List[np.ndarray] = []
    for cl, p in zip(clusters, cluster_pv):
        m = _cluster_to_1d_mask(cl, n_times)
        cluster_masks.append(m)
        if float(p) < float(alpha):
            sig_mask |= m

    return T_obs, np.asarray(cluster_pv, float), cluster_masks, sig_mask


def _format_range(start: float, end: float, unit: str, decimals: int = 1) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(start)}-{fmt.format(end)} {unit}"


def _cluster_time_range_1d(mask_t: np.ndarray, times_s: np.ndarray) -> Optional[Tuple[float, float]]:
    mask_t = np.asarray(mask_t, dtype=bool)
    mask_t = np.squeeze(mask_t)
    if mask_t.ndim != 1:
        raise ValueError("1D cluster mask must be 1D (time)")
    if not mask_t.any():
        return None

    times_s = np.asarray(times_s, dtype=float)
    if mask_t.shape[0] != times_s.size:
        raise ValueError("1D cluster mask shape mismatch")

    idx = np.where(mask_t)[0]
    return float(times_s[idx[0]] * 1e3), float(times_s[idx[-1]] * 1e3)


def _cluster_to_1d_mask(cluster: object, n_times: int) -> np.ndarray:
    """Normalize MNE cluster representation to a 1D boolean mask.

    Depending on MNE version/out_type, clusters can be returned as boolean masks,
    slices, or index arrays/tuples. This normalizes them for downstream summary
    and serialization.
    """
    if isinstance(cluster, np.ndarray):
        arr = np.asarray(cluster)
        if arr.dtype == bool:
            arr = np.squeeze(arr)
            if arr.ndim != 1:
                raise ValueError("Cluster mask must be 1D after squeeze")
            if int(arr.shape[0]) != int(n_times):
                raise ValueError("Cluster mask length mismatch")
            return arr.astype(bool)

        idx = np.asarray(arr, dtype=int).ravel()
        mask = np.zeros(int(n_times), dtype=bool)
        idx = idx[(idx >= 0) & (idx < int(n_times))]
        mask[idx] = True
        return mask

    if isinstance(cluster, slice):
        mask = np.zeros(int(n_times), dtype=bool)
        start = 0 if cluster.start is None else int(cluster.start)
        stop = int(n_times) if cluster.stop is None else int(cluster.stop)
        step = 1 if cluster.step is None else int(cluster.step)
        mask[start:stop:step] = True
        return mask

    if isinstance(cluster, (list, tuple)):
        if len(cluster) != 1:
            raise ValueError("Unexpected cluster tuple/list for 1D data")
        return _cluster_to_1d_mask(cluster[0], n_times)

    arr = np.asarray(cluster)
    if arr.dtype == object and arr.size == 1:
        return _cluster_to_1d_mask(arr.item(), n_times)

    raise TypeError(f"Unsupported cluster type: {type(cluster)}")


def _plot_hep_group_overlay_ax(
    ax: plt.Axes,
    *,
    roi_name: str,
    roi_chs: List[str],
    times: np.ndarray,
    good_mean: np.ndarray,
    nongood_mean: np.ndarray,
    good_sem: np.ndarray,
    nongood_sem: np.ndarray,
    sig_mask: np.ndarray,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    times_ms = times * 1e3

    ax.plot(times_ms, good_mean * 1e6, color="red", lw=2, label="Good")
    ax.fill_between(
        times_ms,
        (good_mean - good_sem) * 1e6,
        (good_mean + good_sem) * 1e6,
        color="red",
        alpha=0.2,
        linewidth=0,
    )

    ax.plot(times_ms, nongood_mean * 1e6, color="blue", lw=2, label="Non-Good")
    ax.fill_between(
        times_ms,
        (nongood_mean - nongood_sem) * 1e6,
        (nongood_mean + nongood_sem) * 1e6,
        color="blue",
        alpha=0.2,
        linewidth=0,
    )

    ax.axvline(0, color="k", lw=1, alpha=0.6)
    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{roi_name} ({', '.join(roi_chs)})")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right")

    if sig_mask.any():
        ymin, ymax = ax.get_ylim()
        bar_y = ymin + 0.05 * (ymax - ymin)
        sig = sig_mask.astype(bool)
        idx = np.where(sig)[0]
        if idx.size > 0:
            starts = [idx[0]]
            ends: List[int] = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    ends.append(idx[i - 1])
                    starts.append(idx[i])
            ends.append(idx[-1])

            for s, e in zip(starts, ends):
                ax.hlines(y=bar_y, xmin=times_ms[s], xmax=times_ms[e], colors="k", linewidth=6)


def _plot_hep_group_overlay(
    *,
    roi_name: str,
    roi_chs: List[str],
    condition: str,
    times: np.ndarray,
    good_mean: np.ndarray,
    nongood_mean: np.ndarray,
    good_sem: np.ndarray,
    nongood_sem: np.ndarray,
    sig_mask: np.ndarray,
    out_png: Path,
) -> None:
    times_ms = times * 1e3

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    ax.plot(times_ms, good_mean * 1e6, color="red", lw=2, label="Good")
    ax.fill_between(
        times_ms,
        (good_mean - good_sem) * 1e6,
        (good_mean + good_sem) * 1e6,
        color="red",
        alpha=0.2,
        linewidth=0,
    )

    ax.plot(times_ms, nongood_mean * 1e6, color="blue", lw=2, label="Non-Good")
    ax.fill_between(
        times_ms,
        (nongood_mean - nongood_sem) * 1e6,
        (nongood_mean + nongood_sem) * 1e6,
        color="blue",
        alpha=0.2,
        linewidth=0,
    )

    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{roi_name} ({', '.join(roi_chs)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if sig_mask.any():
        ymin, ymax = ax.get_ylim()
        bar_y = ymin + 0.05 * (ymax - ymin)
        sig = sig_mask.astype(bool)
        idx = np.where(sig)[0]
        if idx.size > 0:
            starts = [idx[0]]
            ends: List[int] = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    ends.append(idx[i - 1])
                    starts.append(idx[i])
            ends.append(idx[-1])

            for s, e in zip(starts, ends):
                ax.hlines(y=bar_y, xmin=times_ms[s], xmax=times_ms[e], colors="k", linewidth=6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _hep_path(processed_dir: Path, subject_id: str, condition: str) -> Path:
    pdir = processed_dir / subject_id
    if condition == "target":
        return pdir / f"{subject_id}_hep_target-ave.fif"
    if condition == "control":
        return pdir / f"{subject_id}_hep_control-ave.fif"
    raise ValueError("condition must be 'target' or 'control'")


def _find_reference_times(
    processed_dir: Path,
    subject_ids: Sequence[str],
    condition: str,
) -> Optional[np.ndarray]:
    for sid in subject_ids:
        path = _hep_path(processed_dir, sid, condition)
        if not path.exists():
            continue
        try:
            ev = _read_one_evoked(path)
        except Exception as e:
            _log_warn(f"Failed to read evoked for reference; skip {sid}: {e}")
            continue
        return np.asarray(ev.times, dtype=float)
    return None


def _collect_group_matrix(
    *,
    processed_dir: Path,
    subject_ids: Sequence[str],
    condition: str,
    roi_candidates: List[str],
    ref_times: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    ys: List[np.ndarray] = []
    used: List[str] = []
    pick_masks: List[np.ndarray] = []

    for sid in subject_ids:
        path = _hep_path(processed_dir, sid, condition)
        if not path.exists():
            _log_warn(f"Missing HEP {condition} for {sid}; skip")
            continue

        try:
            ev = _read_one_evoked(path)
            y, _picked, pick_mask = _roi_mean_and_mask_from_evoked(ev, roi_candidates)
            subj_times = np.asarray(ev.times, dtype=float)
        except Exception as e:
            _log_warn(f"Failed to load/compute ROI mean for {sid} ({condition}); skip: {e}")
            continue

        if y.shape[0] != ref_times.size or not np.allclose(subj_times, ref_times):
            y_al = _align_1d_to_reference(
                x=y,
                x_axis=subj_times,
                ref_axis=ref_times,
                label=f"HEP-{condition}",
                subject_id=sid,
            )
            if y_al is None:
                _log_warn(f"Skipping subject for HEP due to time grid mismatch: {sid}")
                continue
            y = y_al

        ys.append(np.asarray(y, float))
        used.append(sid)
        pick_masks.append(np.asarray(pick_mask, dtype=bool))

    if not ys:
        raise RuntimeError(f"No usable subjects after alignment for condition={condition}")

    X = np.stack(ys, axis=0)
    picked_mask = (
        np.stack(pick_masks, axis=0)
        if pick_masks
        else np.zeros((0, len(roi_candidates)), dtype=bool)
    )
    return X, used, picked_mask


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

        fig_height = 3.6 * max(1, len(roi_names))
        fig, axes = plt.subplots(
            nrows=len(roi_names),
            ncols=1,
            figsize=(8, fig_height),
            constrained_layout=True,
            sharex=True,
        )
        axes_list = [cast(plt.Axes, axes)] if len(roi_names) == 1 else list(np.ravel(axes))

        for i, roi_name in enumerate(roi_names):
            roi_candidates = roi_dict[roi_name]
            ax = cast(plt.Axes, axes_list[i])

            ref_times = _find_reference_times(
                processed_dir,
                list(good_subjects) + list(nongood_subjects),
                condition,
            )
            if ref_times is None:
                raise RuntimeError(f"Could not determine reference times for condition={condition}")

            X_good, good_used, good_pick_mask = _collect_group_matrix(
                processed_dir=processed_dir,
                subject_ids=good_subjects,
                condition=condition,
                roi_candidates=roi_candidates,
                ref_times=ref_times,
            )
            X_nongood, nongood_used, nongood_pick_mask = _collect_group_matrix(
                processed_dir=processed_dir,
                subject_ids=nongood_subjects,
                condition=condition,
                roi_candidates=roi_candidates,
                ref_times=ref_times,
            )

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
                f"HEP {condition_label} ROI={roi_name}: effective_channels={roi_effective}"
            )
            _log_info(
                f"HEP {condition_label} ROI={roi_name}: X_good={X_good.shape} X_nongood={X_nongood.shape}"
            )

            T_obs, cluster_pv, cluster_masks, sig_mask = _run_cluster_2samp(
                X_good=X_good,
                X_nongood=X_nongood,
                n_permutations=n_permutations,
                cluster_alpha=cluster_alpha,
                alpha=alpha,
                tail=tail,
                seed=seed,
            )

            good_mean = np.mean(X_good, axis=0)
            nongood_mean = np.mean(X_nongood, axis=0)
            good_sem = stats.sem(X_good, axis=0, nan_policy="omit")
            nongood_sem = stats.sem(X_nongood, axis=0, nan_policy="omit")

            _plot_hep_group_overlay_ax(
                ax,
                roi_name=roi_name,
                roi_chs=roi_effective,
                times=ref_times,
                good_mean=good_mean,
                nongood_mean=nongood_mean,
                good_sem=good_sem,
                nongood_sem=nongood_sem,
                sig_mask=sig_mask,
                show_xlabel=(i == len(roi_names) - 1),
                show_legend=(i == 0),
            )

            # Summarize significant clusters
            cluster_id = 0
            for mask, p in zip(cluster_masks, cluster_pv):
                if float(p) >= float(alpha):
                    continue
                r = _cluster_time_range_1d(mask_t=mask, times_s=ref_times)
                if r is None:
                    continue
                tmin_ms, tmax_ms = r
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
                    }
                )

            np.savez_compressed(
                out_dir / f"stats_{roi_name}_{condition}.npz",
                condition=str(condition_label),
                roi=str(roi_name),
                # Backward compatible key: now stores effective channels actually used.
                roi_channels=np.asarray(roi_effective, dtype=str),
                # New keys: candidate list and per-subject usage masks.
                roi_candidates=np.asarray(roi_candidates, dtype=str),
                good_roi_pick_mask=np.asarray(good_pick_mask, dtype=bool),
                nongood_roi_pick_mask=np.asarray(nongood_pick_mask, dtype=bool),
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

        out_svg = out_dir / f"hep_group_{condition}_Good_vs_NonGood_all_rois.svg"
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, format="svg", bbox_inches="tight")
        plt.close(fig)
        _log_info(f"Saved: {out_svg}")

    summary_path = out_dir / "statistics_summary.csv"
    df = pd.DataFrame(
        summary_rows,
        columns=["ROI", "Condition", "n_good", "n_nongood", "Cluster_ID", "p_value", "Time_Range"],
    )
    df.to_csv(summary_path, index=False)
    _log_info(f"Saved: {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Between-subject group statistics for HEP (Good vs Non-Good). "
            "Loads precomputed HEP Evoked files (no recomputation)."
        )
    )

    p.add_argument(
        "--good_csv",
        type=str,
        default=str(_repo_root() / "data" / "classification" / "good_responders.csv"),
        help="Path to good_responders.csv",
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
        help="Output directory for between-subject group stats",
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
        # ACC / frontocentral-friendly candidates.
        # (F1/F2/FCz may not exist depending on montage; they are safe as candidates.)
        "Frontal": ["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"],
        "Visual": ["Oz", "O1", "O2"],
        # Some datasets may not have Pz; include CPz/CP1/CP2 as fallbacks.
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
