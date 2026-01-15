#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HEP-TFR Analysis Script.
Computes Time-Frequency Representations (TFR) locked to R-peaks (ECG).

Reference:
- Logic merged from src/02_analysis_tfr.py (TFR computation)
- and src/03_analysis_hep.py (ECG event generation).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import mne

# ---------------------------------------------------------------------------
# Constants & Helpers (Shared/Adapted)
# ---------------------------------------------------------------------------

TARGET_SESSION_TYPES = {"increase", "decrease", "inc", "dec"}
CONTROL_SESSION_TYPES = {"control"}

def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _processed_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "processed" / subject_id

def _load_conditions_manifest(subject_id: str) -> Dict:
    manifest_path = _processed_dir(subject_id) / "conditions_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"conditions_manifest.json not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _session_type_to_condition(session_type: str) -> str:
    s = (session_type or "").strip().lower()
    if s in TARGET_SESSION_TYPES:
        return "Target"
    if s in CONTROL_SESSION_TYPES:
        return "Control"
    raise ValueError(f"Unknown session type: {session_type}")

def _session_tag_to_condition(subject_id: str, sess_tag: str) -> str:
    manifest = _load_conditions_manifest(subject_id)
    m = manifest.get("session_condition_map")
    if not isinstance(m, dict) or not m:
        raise ValueError(f"Invalid session_condition_map for {subject_id}")
    if sess_tag not in m:
        raise KeyError(f"session_condition_map has no key '{sess_tag}'")
    return _session_type_to_condition(str(m[sess_tag]))

def _extract_sess_tag(fif_path: Path) -> str:
    m = re.search(r"_sess(\d{2})_", fif_path.name)
    if not m:
        raise ValueError(f"Could not infer sessXX from: {fif_path.name}")
    return f"sess{m.group(1)}"

# ---------------------------------------------------------------------------
# File Resolution (From src/03_analysis_hep.py)
# ---------------------------------------------------------------------------

def _resolve_input_fifs(subject_id: str) -> List[Path]:
    proc_dir = _processed_dir(subject_id)
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc_dir}")
    fifs = sorted(proc_dir.glob("*_clean.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_clean.fif files found under: {proc_dir}")
    return fifs

def _resolve_ecg_session_jsons(ecg_dir: Path) -> List[Path]:
    if not ecg_dir.exists():
        raise FileNotFoundError(f"ECG dir not found: {ecg_dir}")
    js = sorted(ecg_dir.glob("*.json"))
    
    pat1 = re.compile(r"Session_0?1", re.IGNORECASE)
    pat2 = re.compile(r"Session_0?2", re.IGNORECASE)

    s1 = [p for p in js if pat1.search(p.name) is not None]
    s2 = [p for p in js if pat2.search(p.name) is not None]

    if len(s1) != 1 or len(s2) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve Session_01/02 ECG JSONs in {ecg_dir}. "
            f"Found: s1={len(s1)}, s2={len(s2)}"
        )
    return [s1[0], s2[0]]

def _load_peak_indices(json_path: Path) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "peak_indices" not in obj:
        raise ValueError(f"Missing 'peak_indices' in: {json_path}")
    return np.asarray(obj["peak_indices"], dtype=float)

# ---------------------------------------------------------------------------
# Event Generation (From src/03_analysis_hep.py)
# ---------------------------------------------------------------------------

def _make_hep_events_from_peaks(
    peak_indices_500hz: np.ndarray,
    event_id: int,
    n_times_raw: int,
) -> np.ndarray:
    # ECG peaks are 500Hz, EEG is 250Hz -> divide by 2
    samples_250 = np.rint(peak_indices_500hz / 2.0).astype(int)
    samples_250 = samples_250[(samples_250 >= 0) & (samples_250 < n_times_raw)]

    events = np.zeros((len(samples_250), 3), dtype=int)
    events[:, 0] = samples_250
    events[:, 2] = int(event_id)
    return events

# ---------------------------------------------------------------------------
# TFR Computation (From src/02_analysis_tfr.py)
# ---------------------------------------------------------------------------

def _compute_avg_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    decim: int,
    average: bool,
    use_fft: bool,
    n_jobs: int,
) -> mne.time_frequency.AverageTFR:
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=use_fft,
        return_itc=False,
        average=average,
        decim=decim,
        n_jobs=n_jobs,
        picks="eeg",
        output="power",
        verbose=False,
    )
    if not isinstance(power, mne.time_frequency.AverageTFR):
        raise TypeError(f"Expected AverageTFR, got {type(power)}")
    
    # Optimize size
    power.data = np.asarray(power.data, dtype=np.float32)
    return power

def _weighted_average_tfr(tfrs: List[mne.time_frequency.AverageTFR]) -> mne.time_frequency.AverageTFR:
    if not tfrs:
        raise ValueError("No TFRs provided for averaging")
    if len(tfrs) == 1:
        return tfrs[0]

    ref = tfrs[0]
    weights = np.array([max(int(getattr(t, "nave", 0) or 0), 1) for t in tfrs], dtype=float)
    wsum = float(weights.sum())

    data = np.zeros_like(ref.data, dtype=np.float64)
    for t, w in zip(tfrs, weights):
        if t.ch_names != ref.ch_names or not np.allclose(t.times, ref.times) or not np.allclose(t.freqs, ref.freqs):
             raise ValueError("TFR mismatch (channels/times/freqs) across sessions")
        data += w * t.data.astype(np.float64)
    data /= wsum

    out = ref.copy()
    out.data = data.astype(np.float32)
    out.nave = int(round(wsum))
    return out

def _save_average_tfr(out_path: Path, tfr: mne.time_frequency.AverageTFR, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        _log_info(f"Exists, skip: {out_path.name}")
        return
    mne.time_frequency.write_tfrs(str(out_path), tfr, overwrite=overwrite)
    _log_info(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main Processing Logic
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    ecg_dir_arg: Optional[str],
    overwrite: bool,
    tmin: float,
    tmax: float,
    baseline: Optional[Tuple[float, float]],
    reject_eeg: Optional[float],
    fmin: float,
    fmax: float,
    fstep: float,
    n_cycles_div: float,
    decim: int,
    n_jobs: int,
    use_fft: bool,
) -> None:
    # Resolve files
    fifs = _resolve_input_fifs(subject_id)
    
    if ecg_dir_arg:
        ecg_dir = Path(ecg_dir_arg)
    else:
        ecg_dir = _repo_root() / "data" / "ecg" / subject_id
    
    ecg_jsons = _resolve_ecg_session_jsons(ecg_dir)

    if len(fifs) != len(ecg_jsons):
         raise RuntimeError(
            f"Mismatch: FIF files ({len(fifs)}) vs ECG JSONs ({len(ecg_jsons)}). "
            "Ensure strict sess01/sess02 correspondence."
        )

    # Output paths
    out_dir = _processed_dir(subject_id)
    out_target = out_dir / f"{subject_id}_tfr_hep_target.h5"
    out_control = out_dir / f"{subject_id}_tfr_hep_control.h5"

    if not overwrite and out_target.exists() and out_control.exists():
        _log_info(f"Both target/control HEP-TFR exist. Skipping {subject_id}")
        return

    # Prepare freq grid
    freqs = np.arange(float(fmin), float(fmax) + 1e-9, float(fstep), dtype=float)
    n_cycles = np.maximum(freqs / float(n_cycles_div), 1.0)

    target_tfrs: List[mne.time_frequency.AverageTFR] = []
    control_tfrs: List[mne.time_frequency.AverageTFR] = []

    # Process pairs
    for fif_path, ecg_json in zip(sorted(fifs), sorted(ecg_jsons)):
        sess_tag = _extract_sess_tag(fif_path)
        cond = _session_tag_to_condition(subject_id, sess_tag)

        # Skip if output exists
        if not overwrite:
            if cond == "Target" and out_target.exists():
                continue
            if cond == "Control" and out_control.exists():
                continue

        _log_info(f"Processing: {fif_path.name} ({cond}) + {ecg_json.name}")

        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        peak_idx = _load_peak_indices(ecg_json)
        
        # Create HEP events (Event ID 999 is arbitrary but consistent)
        hep_events = _make_hep_events_from_peaks(peak_idx, 999, raw.n_times)
        
        if len(hep_events) == 0:
            _log_warn(f"No HEP events for {fif_path.name}. Skipping.")
            continue

        epochs = mne.Epochs(
            raw,
            events=hep_events,
            event_id={"R": 999},
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks="eeg",
            reject={"eeg": reject_eeg} if reject_eeg else None,
            preload=True,
            detrend=None,
            reject_by_annotation=True,
            verbose=False,
        )
        
        _log_info(f"  Epochs: {len(epochs)}")
        if len(epochs) == 0:
            _log_warn("  All epochs dropped. Skipping TFR calculation.")
            continue

        tfr = _compute_avg_tfr(
            epochs=epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=int(decim),
            average=True,
            use_fft=bool(use_fft),
            n_jobs=int(n_jobs),
        )
        
        tfr.comment = f"{subject_id} {sess_tag} {cond} (HEP-TFR)"

        if cond == "Target":
            target_tfrs.append(tfr)
        else:
            control_tfrs.append(tfr)

    # Save Results
    if target_tfrs:
        tfr_target = _weighted_average_tfr(target_tfrs)
        _save_average_tfr(out_target, tfr_target, overwrite=overwrite)
    elif not out_target.exists():
        _log_warn(f"No Target TFRs generated for {subject_id}")

    if control_tfrs:
        tfr_control = _weighted_average_tfr(control_tfrs)
        _save_average_tfr(out_control, tfr_control, overwrite=overwrite)
    elif not out_control.exists():
        _log_warn(f"No Control TFRs generated for {subject_id}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HEP-TFR Analysis: R-peak locked Time-Frequency Analysis")
    p.add_argument("--subject_id", type=str, required=True, help="Subject ID to process")
    p.add_argument("--ecg_dir", type=str, default=None, help="Directory containing ECG JSONs (optional)")
    p.add_argument("--overwrite", action="store_true")
    
    # TFR / Epoch params
    p.add_argument("--tmin", type=float, default=-0.3, help="Epoch start (s) relative to R-peak")
    p.add_argument("--tmax", type=float, default=0.8, help="Epoch end (s) relative to R-peak")
    p.add_argument("--baseline", type=float, nargs=2, default=[-0.2, -0.05], help="Baseline interval (s)")
    p.add_argument("--reject_eeg", type=float, default=None, help="EEG rejection (Volts). E.g., 150e-6")
    
    p.add_argument("--fmin", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=45.0)
    p.add_argument("--fstep", type=float, default=1.0)
    p.add_argument("--n_cycles_div", type=float, default=2.0)
    p.add_argument("--decim", type=int, default=1)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--use_fft", action="store_true")

    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    baseline = None
    if args.baseline:
        baseline = (float(args.baseline[0]), float(args.baseline[1]))

    process_subject(
        subject_id=args.subject_id,
        ecg_dir_arg=args.ecg_dir,
        overwrite=bool(args.overwrite),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        baseline=baseline,
        reject_eeg=float(args.reject_eeg) if args.reject_eeg else None,
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        fstep=float(args.fstep),
        n_cycles_div=float(args.n_cycles_div),
        decim=int(args.decim),
        n_jobs=int(args.n_jobs),
        use_fft=bool(args.use_fft),
    )

if __name__ == "__main__":
    main()