#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install requirements:
# pip install mne numpy pandas scipy matplotlib tqdm scikit-learn asrpy mne-icalabel pybv h5py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import mne


TRIG_PREFIX = "TRIG/"
FEEDBACK_DESCRIPTIONS = (
    f"{TRIG_PREFIX}Feedback_High",
    f"{TRIG_PREFIX}Feedback_Low",
)

TARGET_SESSION_TYPES = {"increase", "decrease", "inc", "dec"}
CONTROL_SESSION_TYPES = {"control"}


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _repo_root() -> Path:
    # src/02_analysis_tfr.py -> repo root is parent of src
    return Path(__file__).resolve().parents[1]


def _processed_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "processed" / subject_id


def _resolve_input_fifs(subject_id: str) -> List[Path]:
    proc_dir = _processed_dir(subject_id)
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc_dir}")

    fifs = sorted(proc_dir.glob(f"{subject_id}_sess*_clean.fif"))
    if not fifs:
        # fallback to any *_clean.fif (for historical naming)
        fifs = sorted(proc_dir.glob("*_clean.fif"))

    if not fifs:
        raise FileNotFoundError(f"No *_clean.fif files found under: {proc_dir}")
    return fifs


def _resolve_fif_path_maybe_under_processed(fif_arg: str) -> Path:
    p = Path(fif_arg)
    if p.exists():
        return p

    name = p.name
    candidates = sorted((_repo_root() / "data" / "processed").glob(f"**/{name}"))
    if len(candidates) == 1:
        _log_warn(f"FIF not found at '{p}'. Using: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"FIF not found: {p}. Multiple candidates under data/processed: "
            + ", ".join(str(c) for c in candidates)
        )
    return p


def _infer_subject_id_from_fif_path(fif_path: Path) -> Optional[str]:
    # Common: data/processed/<subject_id>/<subject_id>_sess01_clean.fif
    parent = fif_path.parent
    if parent.name and (parent.parent / parent.name).exists():
        # not a strong signal; just allow a simple heuristic: folder name looks like an ID.
        return parent.name
    return None


def _extract_sess_tag(fif_path: Path) -> str:
    m = re.search(r"_sess(\d{2})_", fif_path.name)
    if not m:
        raise ValueError(
            f"Could not infer session tag (sessXX) from FIF name: {fif_path.name}. "
            "Expected pattern like '<subject>_sess01_clean.fif'."
        )
    return f"sess{m.group(1)}"


def _load_conditions_manifest(subject_id: str) -> Dict:
    manifest_path = _processed_dir(subject_id) / "conditions_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"conditions_manifest.json not found: {manifest_path}. "
            "Run src/00_batch_run.py or ensure the manifest exists."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _session_type_to_condition(session_type: str) -> str:
    s = (session_type or "").strip().lower()
    if s in TARGET_SESSION_TYPES:
        return "Target"
    if s in CONTROL_SESSION_TYPES:
        return "Control"
    raise ValueError(
        "Unknown session type in conditions_manifest.json: "
        f"'{session_type}'. Expected one of Increase/Decrease/Control (or Inc/Dec)."
    )


def _session_tag_to_condition(subject_id: str, sess_tag: str) -> str:
    manifest = _load_conditions_manifest(subject_id)
    m = manifest.get("session_condition_map")
    if not isinstance(m, dict) or not m:
        raise ValueError(
            f"Invalid or missing session_condition_map in conditions_manifest.json for {subject_id}"
        )
    if sess_tag not in m:
        raise KeyError(
            f"session_condition_map has no key '{sess_tag}' for {subject_id}. Keys={list(m.keys())}"
        )
    return _session_type_to_condition(str(m[sess_tag]))


def _make_feedback_epochs(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    reject_eeg: Optional[float],
) -> mne.Epochs:
    # Use annotation *strings* (robust) rather than numeric event IDs.
    event_id = {desc: i + 1 for i, desc in enumerate(FEEDBACK_DESCRIPTIONS)}

    try:
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create events from annotations. "
            "Ensure preprocessing injected TRIG/Feedback_* annotations into *_clean.fif."
        ) from exc

    if events.size == 0:
        raise RuntimeError("No feedback events found from TRIG/Feedback_High/Low annotations")

    reject: Optional[Dict[str, float]]
    if reject_eeg is None or reject_eeg <= 0:
        reject = None
    else:
        reject = {"eeg": float(reject_eeg)}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=float(tmin),
        tmax=float(tmax),
        baseline=baseline,
        picks="eeg",
        reject=reject,
        preload=True,
        detrend=None,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


def _compute_avg_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    decim: int,
    average: bool,
    use_fft: bool,
    n_jobs: int,
) -> mne.time_frequency.AverageTFR:
    # AverageTFR (not EpochsTFR) to keep file sizes reasonable.
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
        # In case average=False or API changes.
        raise TypeError(
            f"Expected AverageTFR but got {type(power)}. Ensure average=True when calling tfr_morlet."
        )

    # Reduce on-disk size.
    power.data = np.asarray(power.data, dtype=np.float32)
    return power


def _weighted_average_tfr(tfrs: List[mne.time_frequency.AverageTFR]) -> mne.time_frequency.AverageTFR:
    if not tfrs:
        raise ValueError("No TFRs provided for averaging")
    if len(tfrs) == 1:
        return tfrs[0]

    ref = tfrs[0]
    for t in tfrs[1:]:
        if t.ch_names != ref.ch_names:
            raise ValueError("TFR channel sets differ; cannot average across sessions")
        if not np.allclose(t.times, ref.times):
            raise ValueError("TFR times differ; cannot average across sessions")
        if not np.allclose(t.freqs, ref.freqs):
            raise ValueError("TFR freqs differ; cannot average across sessions")

    weights = np.array([max(int(getattr(t, "nave", 0) or 0), 1) for t in tfrs], dtype=float)
    wsum = float(weights.sum())

    data = np.zeros_like(ref.data, dtype=np.float64)
    for t, w in zip(tfrs, weights):
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

    # Most robust across MNE versions: write_tfrs
    mne.time_frequency.write_tfrs(str(out_path), tfr, overwrite=overwrite)
    _log_info(f"Saved: {out_path}")


def process_subject(
    subject_id: str,
    overwrite: bool,
    tmin: float,
    tmax: float,
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    reject_eeg: Optional[float],
    fmin: float,
    fmax: float,
    fstep: float,
    n_cycles_div: float,
    decim: int,
    n_jobs: int,
    use_fft: bool,
) -> None:
    fifs = _resolve_input_fifs(subject_id)

    out_dir = _processed_dir(subject_id)
    out_target = out_dir / f"{subject_id}_tfr_target.h5"
    out_control = out_dir / f"{subject_id}_tfr_control.h5"

    # If both already exist and overwrite is False, skip everything.
    if not overwrite and out_target.exists() and out_control.exists():
        _log_info(f"Both target/control TFR exist. Skipping subject: {subject_id}")
        return

    freqs = np.arange(float(fmin), float(fmax) + 1e-9, float(fstep), dtype=float)
    if freqs.size < 2:
        raise ValueError("Frequency grid too small; check --fmin/--fmax/--fstep")

    # A simple, commonly used rule: n_cycles increases with frequency.
    n_cycles = freqs / float(n_cycles_div)
    n_cycles = np.maximum(n_cycles, 1.0)

    target_tfrs: List[mne.time_frequency.AverageTFR] = []
    control_tfrs: List[mne.time_frequency.AverageTFR] = []

    for fif_path in sorted(fifs):
        sess_tag = _extract_sess_tag(fif_path)
        cond = _session_tag_to_condition(subject_id, sess_tag)

        # Per-session skip: if that condition output exists and overwrite False, skip.
        if not overwrite:
            if cond == "Target" and out_target.exists():
                _log_info(f"Target output already exists; skipping {fif_path.name}")
                continue
            if cond == "Control" and out_control.exists():
                _log_info(f"Control output already exists; skipping {fif_path.name}")
                continue

        _log_info(f"Loading: {fif_path.name} ({sess_tag} -> {cond})")
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

        epochs = _make_feedback_epochs(
            raw=raw,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject_eeg=reject_eeg,
        )
        _log_info(f"Epochs: n={len(epochs)} (feedback events)")

        tfr = _compute_avg_tfr(
            epochs=epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=int(decim),
            average=True,
            use_fft=bool(use_fft),
            n_jobs=int(n_jobs),
        )

        # Add a descriptive comment for provenance.
        comment = f"{subject_id} {sess_tag} {cond} (Feedback-locked)"
        try:
            tfr.comment = comment
        except Exception:
            pass

        if cond == "Target":
            target_tfrs.append(tfr)
        else:
            control_tfrs.append(tfr)

    if target_tfrs:
        tfr_target = _weighted_average_tfr(target_tfrs)
        _save_average_tfr(out_target, tfr_target, overwrite=overwrite)
    else:
        _log_warn(f"No Target sessions found for {subject_id}; not saving {out_target.name}")

    if control_tfrs:
        tfr_control = _weighted_average_tfr(control_tfrs)
        _save_average_tfr(out_control, tfr_control, overwrite=overwrite)
    else:
        _log_warn(f"No Control sessions found for {subject_id}; not saving {out_control.name}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Time-frequency analysis (Morlet TFR) locked to feedback triggers. "
            "Target/Control are defined by session type via data/processed/<id>/conditions_manifest.json."
        )
    )

    p.add_argument("--subject_id", type=str, default=None, help="Subject ID to process")
    p.add_argument("--fif", type=str, default=None, help="Single *_clean.fif to process")
    p.add_argument(
        "--session_type",
        type=str,
        default=None,
        help=(
            "Only for --fif mode when conditions_manifest.json is not available. "
            "One of Increase/Decrease/Control (case-insensitive)."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_tfr_target.h5 / *_tfr_control.h5 outputs",
    )

    p.add_argument("--tmin", type=float, default=-0.5, help="Epoch start (s) relative to feedback")
    p.add_argument("--tmax", type=float, default=1.5, help="Epoch end (s) relative to feedback")
    p.add_argument(
        "--baseline",
        type=float,
        nargs=2,
        default=None,
        help=(
            "Optional baseline interval in seconds. Example: --baseline -0.3 0.0. "
            "Use 'None' by omitting this arg."
        ),
    )
    p.add_argument(
        "--reject_eeg",
        type=float,
        default=None,
        help=(
            "Optional EEG rejection threshold in Volts (e.g., 150e-6). "
            "Omit to disable amplitude-based rejection."
        ),
    )

    p.add_argument("--fmin", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=45.0)
    p.add_argument("--fstep", type=float, default=1.0)
    p.add_argument(
        "--n_cycles_div",
        type=float,
        default=2.0,
        help="n_cycles = freqs / n_cycles_div (min 1). Default: 2 (e.g., 10Hz->5 cycles).",
    )
    p.add_argument(
        "--decim",
        type=int,
        default=1,
        help="Decimation factor applied during TFR to reduce time resolution and file size.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for MNE TFR (set to -1 for all cores).",
    )
    p.add_argument(
        "--use_fft",
        action="store_true",
        help="Use FFT-based convolution (faster for longer epochs)",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.subject_id is not None and args.fif is not None:
        raise ValueError("Provide only one of --subject_id or --fif")
    if args.subject_id is None and args.fif is None:
        raise ValueError("Provide either --subject_id or --fif")

    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None
    if args.baseline is not None:
        baseline = (float(args.baseline[0]), float(args.baseline[1]))

    if args.subject_id is not None:
        process_subject(
            subject_id=str(args.subject_id),
            overwrite=bool(args.overwrite),
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            baseline=baseline,
            reject_eeg=float(args.reject_eeg) if args.reject_eeg is not None else None,
            fmin=float(args.fmin),
            fmax=float(args.fmax),
            fstep=float(args.fstep),
            n_cycles_div=float(args.n_cycles_div),
            decim=int(args.decim),
            n_jobs=int(args.n_jobs),
            use_fft=bool(args.use_fft),
        )
        return

    # --fif mode: process a single session and write a session-typed output.
    fif_path = _resolve_fif_path_maybe_under_processed(str(args.fif))
    if not fif_path.exists():
        raise FileNotFoundError(f"FIF not found: {fif_path}")

    subject_id = _infer_subject_id_from_fif_path(fif_path)
    sess_tag = _extract_sess_tag(fif_path)

    cond: Optional[str] = None
    if subject_id is not None:
        try:
            cond = _session_tag_to_condition(subject_id, sess_tag)
        except Exception:
            cond = None

    if cond is None:
        if args.session_type is None:
            raise ValueError(
                "Could not infer Target/Control from conditions_manifest.json. "
                "Provide --session_type Increase/Decrease/Control."
            )
        cond = _session_type_to_condition(str(args.session_type))

    # Write into the FIF's parent directory by default.
    out_dir = fif_path.parent
    stem_subject = subject_id or fif_path.stem.split("_sess")[0]

    out_path = out_dir / f"{stem_subject}_tfr_{cond.lower()}.h5"
    if out_path.exists() and not args.overwrite:
        _log_info(f"Exists, skip: {out_path}")
        return

    _log_info(f"Loading: {fif_path.name} ({sess_tag} -> {cond})")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    epochs = _make_feedback_epochs(
        raw=raw,
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        baseline=baseline,
        reject_eeg=float(args.reject_eeg) if args.reject_eeg is not None else None,
    )

    freqs = np.arange(float(args.fmin), float(args.fmax) + 1e-9, float(args.fstep), dtype=float)
    n_cycles = np.maximum(freqs / float(args.n_cycles_div), 1.0)

    tfr = _compute_avg_tfr(
        epochs=epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=int(args.decim),
        average=True,
        use_fft=bool(args.use_fft),
        n_jobs=int(args.n_jobs),
    )

    try:
        tfr.comment = f"{stem_subject} {sess_tag} {cond} (Feedback-locked)"
    except Exception:
        pass

    _save_average_tfr(out_path, tfr, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
