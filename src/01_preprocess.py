#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install requirements:
# pip install mne numpy pandas scipy matplotlib tqdm scikit-learn asrpy mne-icalabel pybv
#
# Notes:
# - `pybv` is only needed for exporting BrainVision; reading BrainVision via MNE does not require it.
# - Optional extras:
#   - Faster ICA: pip install python-picard
#   - ICLabel backend: pip install torch  OR  pip install onnxruntime

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import mne

try:
    from asrpy import ASR
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "asrpy is required for ASR. Install with: pip install asrpy"
    ) from exc


TRIG_PREFIX = "TRIG/"


@dataclass(frozen=True)
class AlignmentResult:
    csv_path: Path
    mode: str  # 'offset' or 'linear'
    window_start: int
    window_len: int
    mad: float
    offset: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    warnings.warn(msg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_vhdrs(subject_id: str) -> List[Path]:
    raw_dir = Path("data") / "raw" / subject_id
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    matches = sorted(raw_dir.glob(f"{subject_id}*.vhdr"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No VHDR files found under: {raw_dir} (pattern: {subject_id}*.vhdr)"
        )
    return matches


def _resolve_vhdr(subject_id: str) -> Path:
    matches = _resolve_vhdrs(subject_id)
    if len(matches) > 1:
        _log_warn(
            "Multiple VHDR files detected: "
            + ", ".join(m.name for m in matches)
            + f". Using: {matches[0].name}"
        )
    return matches[0]


def _resolve_trigger_csvs(subject_id: str) -> List[Path]:
    trigger_dir = Path("data") / "trigger" / subject_id
    if not trigger_dir.exists():
        raise FileNotFoundError(f"Trigger directory not found: {trigger_dir}")
    csvs = sorted(trigger_dir.glob("*_actichamp_trigger_session.csv"))
    if not csvs:
        raise FileNotFoundError(f"No trigger CSV found under: {trigger_dir}")
    return csvs


def _find_stimulus1_key(event_id: Dict[str, int]) -> str:
    # BrainVision descriptions often look like "Stimulus/S  1".
    # We match spaces flexibly and prefer the exact Stimulus marker.
    pat = re.compile(r"^Stimulus/S\s+1$")
    candidates = [k for k in event_id.keys() if pat.match(k)]
    if candidates:
        return candidates[0]

    # Fallback: sometimes spacing differs (e.g., one space)
    pat2 = re.compile(r"^Stimulus/S\s*1$")
    candidates2 = [k for k in event_id.keys() if pat2.match(k)]
    if candidates2:
        return candidates2[0]

    raise KeyError(
        "Could not find Stimulus/S  1 in annotations. Available keys include: "
        + ", ".join(list(event_id.keys())[:30])
    )


def _load_trigger_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "trigger_value", "annotation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trigger CSV missing columns {sorted(missing)}: {csv_path}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df[df["trigger_value"] == 1]
    if df.empty:
        raise ValueError(f"No rows with trigger_value==1 in: {csv_path}")

    t0 = df["timestamp"].iloc[0]
    df["t_rel_s"] = (df["timestamp"] - t0).dt.total_seconds().astype(float)
    return df.reset_index(drop=True)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _best_alignment_window(
    stim_times_s: np.ndarray,
    x_rel_s: np.ndarray,
    allow_linear: bool = True,
) -> Tuple[str, int, float, Dict[str, float]]:
    """Find best contiguous window aligning x_rel_s to stim_times_s.

    Returns:
      mode, start_index, mad_cost, params (offset or a,b)
    """
    stim_times_s = np.asarray(stim_times_s, dtype=float)
    x = np.asarray(x_rel_s, dtype=float)

    m = len(x)
    n = len(stim_times_s)
    if m < 2:
        raise ValueError("Need at least 2 trigger points in CSV for alignment")
    if n < m:
        raise ValueError(
            f"Not enough Stimulus/S  1 events in EEG (n={n}) to align CSV (m={m})."
        )

    best_cost = np.inf
    best_start = 0
    best_mode = "offset"
    best_params: Dict[str, float] = {}

    for start in range(0, n - m + 1):
        y = stim_times_s[start : start + m]

        # Offset-only
        offset = float(np.median(y - x))
        resid0 = y - (x + offset)
        cost0 = _mad(resid0)

        mode = "offset"
        cost = cost0
        params: Dict[str, float] = {"offset": offset}

        # Linear drift
        if allow_linear and m >= 3:
            b, a = np.polyfit(x, y, 1)  # y ≈ b*x + a
            resid1 = y - (a + b * x)
            cost1 = _mad(resid1)
            if cost1 < cost0 * 0.9:
                mode = "linear"
                cost = cost1
                params = {"a": float(a), "b": float(b)}

        if cost < best_cost:
            best_cost = cost
            best_start = start
            best_mode = mode
            best_params = params

    return best_mode, best_start, float(best_cost), best_params


def _remove_existing_trigger_annotations(raw: mne.io.BaseRaw, base_labels: Sequence[str]) -> None:
    if raw.annotations is None or len(raw.annotations) == 0:
        return

    labels = set(base_labels)

    keep_onset = []
    keep_dur = []
    keep_desc = []
    keep_ch = []

    for onset, dur, desc, ch in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        raw.annotations.ch_names,
    ):
        desc_str = str(desc)
        is_base = desc_str in labels
        is_prefixed = desc_str.startswith("TRIG/") or desc_str.startswith("TRIGGER/") or desc_str.startswith("TRIG:")
        if is_base or is_prefixed:
            continue
        keep_onset.append(float(onset))
        keep_dur.append(float(dur))
        keep_desc.append(desc_str)
        keep_ch.append(ch)

    raw.set_annotations(
        mne.Annotations(
            onset=np.array(keep_onset, float),
            duration=np.array(keep_dur, float),
            description=np.array(keep_desc, dtype=str),
            orig_time=raw.annotations.orig_time,
            ch_names=keep_ch,
        )
    )


def _add_trig_annotations(
    raw: mne.io.BaseRaw,
    df_trig: pd.DataFrame,
    align: AlignmentResult,
) -> None:
    x = df_trig["t_rel_s"].to_numpy(dtype=float)
    if align.mode == "linear":
        assert align.a is not None and align.b is not None
        onsets_s = align.a + align.b * x
    else:
        assert align.offset is not None
        onsets_s = x + align.offset

    desc = (TRIG_PREFIX + df_trig["annotation"].astype(str)).to_numpy(dtype=str)
    ann = mne.Annotations(
        onset=onsets_s,
        duration=np.zeros_like(onsets_s),
        description=desc,
        orig_time=raw.annotations.orig_time if raw.annotations is not None else None,
    )

    if raw.annotations is None:
        raw.set_annotations(ann)
    else:
        raw.set_annotations(raw.annotations + ann)


def _events_for_trig_labels(raw: mne.io.BaseRaw, labels: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    # Restrict to exact labels to avoid accidental matches.
    escaped = [re.escape(TRIG_PREFIX + lab) for lab in labels]
    regexp = "^(" + "|".join(escaped) + ")$"
    return mne.events_from_annotations(raw, regexp=regexp)


def _estimate_session_times_from_csv(
    df_trig: pd.DataFrame,
    align: AlignmentResult,
) -> Tuple[float, float]:
    def _pick_first(label: str) -> float:
        rows = df_trig.index[df_trig["annotation"] == label].to_list()
        if not rows:
            raise ValueError(f"CSV missing required annotation '{label}'")
        return float(df_trig.loc[rows[0], "t_rel_s"])

    t_start_rel = _pick_first("Session_Start")
    t_stop_rel = _pick_first("Session_Stop")
    if t_stop_rel <= t_start_rel:
        raise ValueError("Session_Stop occurs before/at Session_Start in CSV")

    if align.mode == "linear":
        assert align.a is not None and align.b is not None
        start_s = align.a + align.b * t_start_rel
        stop_s = align.a + align.b * t_stop_rel
    else:
        assert align.offset is not None
        start_s = t_start_rel + align.offset
        stop_s = t_stop_rel + align.offset

    return float(start_s), float(stop_s)


def _fit_ica_with_optional_iclabel(
    raw: mne.io.BaseRaw,
    random_state: int = 97,
    n_components: float = 0.999,
    prob_threshold: float = 0.90,
) -> mne.preprocessing.ICA:
    from mne.preprocessing import ICA

    method = "picard"
    fit_params: Dict[str, object] = {"ortho": False, "extended": True}
    try:
        __import__("picard")
    except Exception:
        method = "fastica"
        fit_params = {}

    ica = ICA(n_components=n_components, method=method, random_state=random_state, fit_params=fit_params)
    ica.fit(raw, picks="eeg")

    # Conservative ICLabel exclusion:
    # - target labels: 'eye blink' and 'muscle artifact'
    # - exclude only if predicted label is those AND y_pred_proba >= 0.90
    try:
        from mne_icalabel import label_components

        out = label_components(raw, ica, method="iclabel")
        labels = list(out.get("labels", []))
        y_pred_proba = np.asarray(out.get("y_pred_proba", []), dtype=float)
        if len(labels) != ica.n_components_ or y_pred_proba.shape[0] != ica.n_components_:
            _log_warn(
                "ICLabel output shape mismatch; skipping component exclusion. "
                f"labels={len(labels)} proba={y_pred_proba.shape} n_components={ica.n_components_}"
            )
            return ica

        targets = {"eye blink", "muscle artifact"}
        exclude = [
            i
            for i, (lab, p) in enumerate(zip(labels, y_pred_proba))
            if (lab in targets) and (float(p) >= prob_threshold)
        ]
        if exclude:
            _log_info(f"ICLabel excluding {len(exclude)} components: {exclude}")
            ica.exclude = list(exclude)
            ica.apply(raw)
        else:
            _log_info("ICLabel: no components met conservative exclusion rule")

    except Exception as exc:
        _log_warn(f"ICLabel not available or failed ({type(exc).__name__}: {exc}). Skipping exclusion.")

    return ica


def preprocess_one_subject(
    subject_id: str,
    vhdr_path: Optional[Path],
    trigger_dir: Optional[Path],
    out_root: Path,
    buffer_s: float = 15.0,
    target_sfreq: float = 250.0,
    asr_cutoff: float = 20.0,
) -> None:
    if vhdr_path is None:
        vhdr_paths = _resolve_vhdrs(subject_id)
    else:
        vhdr_paths = [Path(vhdr_path)]

    if trigger_dir is None:
        try:
            csv_paths = _resolve_trigger_csvs(subject_id)
        except FileNotFoundError as exc:
            _log_warn(
                f"Trigger CSVs not found for subject_id='{subject_id}'. Skipping preprocessing. ({exc})"
            )
            return
    else:
        trigger_dir = Path(trigger_dir)
        if not trigger_dir.exists():
            _log_warn(
                f"trigger_dir does not exist for subject_id='{subject_id}': {trigger_dir}. Skipping preprocessing."
            )
            return
        csv_paths = sorted(trigger_dir.glob("*_actichamp_trigger_session.csv"))
        if not csv_paths:
            _log_warn(
                f"No trigger CSVs under: {trigger_dir} for subject_id='{subject_id}'. Skipping preprocessing."
            )
            return

    out_dir = out_root / subject_id
    _ensure_dir(out_dir)

    if len(vhdr_paths) == 1:
        _log_info(f"Reading BrainVision: {vhdr_paths[0]}")
        raw_full = mne.io.read_raw_brainvision(str(vhdr_paths[0]), preload=True)
    else:
        _log_warn(
            f"Multiple VHDR files detected for subject_id='{subject_id}'. "
            f"Concatenating in filename order: {', '.join(p.name for p in vhdr_paths)}"
        )
        raws = [mne.io.read_raw_brainvision(str(p), preload=True) for p in vhdr_paths]
        raw_full = mne.concatenate_raws(raws)

    sfreq = float(raw_full.info["sfreq"])
    _log_info(f"Raw sfreq={sfreq} Hz")

    # Anchor events: Stimulus/S  1
    events_all, event_id_all = mne.events_from_annotations(raw_full)
    stim_key = _find_stimulus1_key(event_id_all)
    stim_code = event_id_all[stim_key]
    stim_events = events_all[events_all[:, 2] == stim_code]
    if len(stim_events) == 0:
        raise RuntimeError("No Stimulus/S  1 events found in EEG")
    stim_times_s = stim_events[:, 0] / sfreq

    # For each CSV: fit alignment and estimate session start/stop in EEG time.
    session_records = []
    for csv_path in csv_paths:
        df = _load_trigger_csv(csv_path)

        mode, win_start, mad_cost, params = _best_alignment_window(
            stim_times_s=stim_times_s,
            x_rel_s=df["t_rel_s"].to_numpy(dtype=float),
            allow_linear=True,
        )

        if mode == "linear":
            align = AlignmentResult(
                csv_path=csv_path,
                mode=mode,
                window_start=win_start,
                window_len=len(df),
                mad=mad_cost,
                a=params["a"],
                b=params["b"],
            )
        else:
            align = AlignmentResult(
                csv_path=csv_path,
                mode=mode,
                window_start=win_start,
                window_len=len(df),
                mad=mad_cost,
                offset=params["offset"],
            )

        start_s, stop_s = _estimate_session_times_from_csv(df, align)

        session_records.append(
            {
                "csv_path": csv_path,
                "df": df,
                "align": align,
                "start_s": start_s,
                "stop_s": stop_s,
            }
        )

        _log_info(
            f"Aligned {csv_path.name}: mode={mode} MAD={mad_cost:.6f}s start={start_s:.3f}s stop={stop_s:.3f}s"
        )

    # Sort sessions by estimated start time (time-true ordering)
    session_records.sort(key=lambda r: r["start_s"])

    # Process each session independently
    for sess_idx, rec in enumerate(session_records, start=1):
        csv_path = rec["csv_path"]
        df = rec["df"]
        align = rec["align"]
        start_s = float(rec["start_s"])
        stop_s = float(rec["stop_s"])

        sess_tag = f"sess{sess_idx:02d}"
        _log_info(f"Processing {subject_id} {sess_tag} (from {csv_path.name})")

        # Work on a fresh copy of the full raw to avoid annotation accumulation
        raw = raw_full.copy()

        # Remove old TRIG annotations and add new ones for this CSV
        _remove_existing_trigger_annotations(raw, base_labels=df["annotation"].unique().tolist())
        _add_trig_annotations(raw, df, align)

        # Rebuild TRIG events for this session only
        labels = df["annotation"].unique().tolist()
        events, event_id = _events_for_trig_labels(raw, labels=labels)

        # Find session boundary onsets from annotations (TRIG/Session_Start/Stop)
        ann = raw.annotations
        if ann is None or len(ann) == 0:
            raise RuntimeError("No annotations found after adding TRIG")

        def _first_onset(desc: str) -> float:
            idx = np.where(ann.description == desc)[0]
            if len(idx) == 0:
                raise RuntimeError(f"Missing annotation {desc} after TRIG injection")
            return float(ann.onset[idx[0]])

        start_onset = _first_onset(TRIG_PREFIX + "Session_Start")
        stop_onset = _first_onset(TRIG_PREFIX + "Session_Stop")
        if stop_onset <= start_onset:
            raise RuntimeError("Session_Stop onset <= Session_Start onset after alignment")

        # (1) Initial crop with buffer
        tmin = max(0.0, start_onset - buffer_s)
        tmax = min(raw.times[-1], stop_onset + buffer_s)
        raw.crop(tmin=tmin, tmax=tmax)

        # Recompute events after crop to ensure sample coordinates match this cropped raw
        events, event_id = _events_for_trig_labels(raw, labels=labels)

        # (2) Preprocessing
        # Resample with events coordinate transform
        raw, events = raw.resample(sfreq=target_sfreq, events=events)

        # Basic bandpass for ASR/ICA stability (matches notebook intent)
        raw.filter(l_freq=1.0, h_freq=40.0, picks="eeg")

        # Average reference
        raw.set_eeg_reference("average")

        # ASR on EEG channels
        _log_info(f"ASR (cutoff={asr_cutoff})")
        asr = ASR(sfreq=float(raw.info["sfreq"]), cutoff=asr_cutoff)
        asr.fit(raw, picks="eeg")
        raw = asr.transform(raw, picks="eeg")

        # ICA + conservative ICLabel exclusion (graceful fallback)
        _log_info("ICA fit (+ optional ICLabel exclusion)")
        _fit_ica_with_optional_iclabel(raw)

        # (3) Secondary crop: remove buffer strictly to Session_Start..Session_Stop
        # IMPORTANT: TRIG annotation onsets may remain in the original (recording-wide)
        # time base because BrainVision sets annotation orig_time.
        # Secondary crop must use the *segment-relative* time base after the initial crop.
        start_rel = max(0.0, start_onset - tmin)
        stop_rel = stop_onset - tmin
        if stop_rel <= start_rel:
            raise RuntimeError("Secondary crop invalid (stop_rel <= start_rel)")
        if stop_rel > raw.times[-1] + 1e-6:
            raise RuntimeError(
                f"Secondary crop stop_rel ({stop_rel:.6f}s) exceeds segment max time ({raw.times[-1]:.6f}s)."
            )
        raw.crop(tmin=start_rel, tmax=stop_rel)

        # Crop can drop annotations exactly at the boundary (floating tolerance).
        # For downstream consistency, ensure Session_Start/Stop exist in the saved segment.
        ann3 = raw.annotations
        if ann3 is None:
            ann3 = mne.Annotations(onset=[], duration=[], description=[])

        descs = set(map(str, ann3.description))
        add_onsets: List[float] = []
        add_desc: List[str] = []

        if (TRIG_PREFIX + "Session_Start") not in descs:
            add_onsets.append(0.0)
            add_desc.append(TRIG_PREFIX + "Session_Start")

        if (TRIG_PREFIX + "Session_Stop") not in descs:
            # Place at the last sample time (ensure within range)
            sf = float(raw.info["sfreq"])
            last_onset = float(raw.times[-1])
            if last_onset > 0:
                last_onset = max(0.0, last_onset - 1.0 / sf)
            add_onsets.append(last_onset)
            add_desc.append(TRIG_PREFIX + "Session_Stop")

        if add_onsets:
            ann_add = mne.Annotations(
                onset=np.array(add_onsets, dtype=float),
                duration=np.zeros(len(add_onsets), dtype=float),
                description=np.array(add_desc, dtype=str),
                orig_time=ann3.orig_time,
            )
            raw.set_annotations(ann3 + ann_add)

        # Normalize time base for saving:
        # - Make the saved FIF start at sample 0 (first_samp=0)
        # - Make annotation onsets relative to the start of this segment
        # This keeps saved events aligned to the saved Raw (sample 0 == t=0).
        sf = float(raw.info["sfreq"])
        first_time_abs = float(raw.first_samp) / sf
        ann_norm = raw.annotations
        if ann_norm is None:
            ann_norm = mne.Annotations(onset=[], duration=[], description=[])
        onsets_rel = np.asarray(ann_norm.onset, dtype=float) - first_time_abs
        # Keep within the segment bounds (tiny numeric tolerance)
        onsets_rel = np.clip(onsets_rel, 0.0, float(raw.times[-1]))
        ann_rel = mne.Annotations(
            onset=onsets_rel,
            duration=np.asarray(ann_norm.duration, dtype=float),
            description=np.asarray(ann_norm.description, dtype=str),
            orig_time=None,
        )
        data = raw.get_data()
        info = raw.info.copy()
        raw = mne.io.RawArray(data, info, first_samp=0)
        raw.set_annotations(ann_rel)

        # (4) Save: regenerate events after secondary crop to guarantee sample-0 alignment
        events_final, event_id_final = _events_for_trig_labels(raw, labels=labels)

        fif_name = f"{subject_id}_{sess_tag}_clean.fif"
        ev_name = f"{subject_id}_{sess_tag}_events.npy"

        fif_path = out_dir / fif_name
        ev_path = out_dir / ev_name

        _log_info(f"Saving FIF: {fif_path}")
        raw.save(str(fif_path), overwrite=True)

        _log_info(f"Saving events: {ev_path}")
        np.save(str(ev_path), events_final)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="actiCHamp EEG preprocessing (BrainVision + trigger CSV alignment)")

    p.add_argument("--subject_id", type=str, default=None, help="Subject ID (e.g., 251222_AK)")
    p.add_argument("--vhdr", type=str, default=None, help="Path to BrainVision .vhdr (overrides subject_id search)")
    p.add_argument(
        "--trigger_dir",
        type=str,
        default=None,
        help="Directory containing trigger CSVs (overrides default data/trigger/<subject_id>)",
    )

    p.add_argument("--out_root", type=str, default="data/processed", help="Output root directory")

    p.add_argument("--buffer_s", type=float, default=15.0, help="Crop buffer seconds before/after session")
    p.add_argument("--target_sfreq", type=float, default=250.0, help="Target sampling frequency (Hz)")
    p.add_argument("--asr_cutoff", type=float, default=20.0, help="ASR cutoff (typical 20)")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.vhdr is None and args.subject_id is None:
        raise ValueError("Provide either --subject_id or --vhdr")

    if args.vhdr is not None and args.subject_id is None:
        # Derive subject_id from folder name as best-effort
        subject_id = Path(args.vhdr).parent.name
        _log_warn(f"--subject_id not provided; derived subject_id='{subject_id}' from vhdr parent folder")
    else:
        subject_id = str(args.subject_id)

    vhdr_path = Path(args.vhdr) if args.vhdr is not None else None
    trigger_dir = Path(args.trigger_dir) if args.trigger_dir is not None else None

    preprocess_one_subject(
        subject_id=subject_id,
        vhdr_path=vhdr_path,
        trigger_dir=trigger_dir,
        out_root=Path(args.out_root),
        buffer_s=float(args.buffer_s),
        target_sfreq=float(args.target_sfreq),
        asr_cutoff=float(args.asr_cutoff),
    )


if __name__ == "__main__":
    main()
