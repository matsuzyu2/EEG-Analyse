#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install requirements:
# pip install mne numpy pandas scipy matplotlib tqdm scikit-learn asrpy mne-icalabel pybv

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

import mne


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _resolve_input_fifs(subject_id: str) -> List[Path]:
    proc_dir = Path("data") / "processed" / subject_id
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc_dir}")
    fifs = sorted(proc_dir.glob("*_clean.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_clean.fif files found under: {proc_dir}")
    return fifs


def _resolve_ecg_jsons(ecg_dir: Path) -> List[Path]:
    if not ecg_dir.exists():
        raise FileNotFoundError(f"ECG dir not found: {ecg_dir}")
    js = sorted(ecg_dir.glob("*.json"))
    if not js:
        raise FileNotFoundError(f"No .json files found under: {ecg_dir}")
    return js


def _resolve_ecg_session_jsons(ecg_dir: Path) -> List[Path]:
    """Resolve ECG JSONs for HEP sessions.

    This project stores many ECG peak JSONs per subject (baseline, resting, tasks, etc.).
    HEP should use only the session ECG peaks:
      - *Session_01*.json  -> sess01
      - *Session_02*.json  -> sess02
    """
    js = _resolve_ecg_jsons(ecg_dir)

    pat1 = re.compile(r"Session_0?1", re.IGNORECASE)
    pat2 = re.compile(r"Session_0?2", re.IGNORECASE)

    s1 = [p for p in js if pat1.search(p.name) is not None]
    s2 = [p for p in js if pat2.search(p.name) is not None]

    if len(s1) != 1 or len(s2) != 1:
        raise RuntimeError(
            "Could not uniquely resolve ECG JSONs for Session_01/Session_02. "
            f"Session_01 matches={len(s1)} Session_02 matches={len(s2)} under: {ecg_dir}. "
            "Expected exactly one JSON containing 'Session_01' (or 'Session_1') and one containing "
            "'Session_02' (or 'Session_2')."
        )

    return [s1[0], s2[0]]


def _out_fig_dir_for_fif(fif_path: Path, out_dir: Optional[Path]) -> Path:
    if out_dir is None:
        base = fif_path.parent
    else:
        base = out_dir
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _load_peak_indices(json_path: Path) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "peak_indices" not in obj:
        raise ValueError(f"Missing 'peak_indices' in ECG JSON: {json_path}")
    peak_indices = np.asarray(obj["peak_indices"], dtype=float)
    if peak_indices.ndim != 1 or peak_indices.size == 0:
        raise ValueError(f"Invalid peak_indices in: {json_path}")
    return peak_indices


def _select_roi(raw: mne.io.BaseRaw, roi: Optional[List[str]]) -> List[str]:
    if roi is None or len(roi) == 0:
        roi = ["Fz", "Cz", "Pz", "C3", "C4"]

    existing = set(raw.ch_names)
    picked = [ch for ch in roi if ch in existing]
    if not picked:
        # fallback: at least include Cz if possible, else first EEG channel
        if "Cz" in existing:
            picked = ["Cz"]
        else:
            eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
            if len(eeg_picks) == 0:
                raise RuntimeError("No EEG channels found in data")
            picked = [raw.ch_names[int(eeg_picks[0])]]
    return picked


def _make_hep_events_from_peaks(
    peak_indices_500hz: np.ndarray,
    event_id: int,
    n_times_raw: int,
) -> np.ndarray:
    # Spec: ECG peaks are already cropped to Session_Start..Stop and sampled at 500 Hz.
    # EEG is saved at 250 Hz => sample conversion is just /2 (no offset correction).
    samples_250 = np.rint(peak_indices_500hz / 2.0).astype(int)
    samples_250 = samples_250[(samples_250 >= 0) & (samples_250 < n_times_raw)]

    events = np.zeros((len(samples_250), 3), dtype=int)
    events[:, 0] = samples_250
    events[:, 2] = int(event_id)
    return events


def _plot_roi_evoked(evoked: mne.Evoked, roi_chs: List[str], title: str) -> plt.Figure:
    data = evoked.copy().pick(roi_chs).data  # (n_roi, n_times)
    mean = data.mean(axis=0)
    times_ms = evoked.times * 1e3

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times_ms, mean * 1e6, lw=2)
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title + f" (ROI mean: {', '.join(roi_chs)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def process_one_pair(
    fif_path: Path,
    ecg_json: Path,
    out_dir: Optional[Path],
    roi: Optional[List[str]],
    tmin: float,
    tmax: float,
    baseline: Tuple[float, float],
    reject_eeg: float,
    event_id: int,
) -> None:
    _log_info(f"Loading EEG: {fif_path}")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True)

    _log_info(f"Loading ECG peaks: {ecg_json}")
    peak_idx = _load_peak_indices(ecg_json)

    hep_events = _make_hep_events_from_peaks(
        peak_indices_500hz=peak_idx,
        event_id=event_id,
        n_times_raw=raw.n_times,
    )
    if len(hep_events) == 0:
        raise RuntimeError("No valid HEP events after conversion/clipping")

    _log_info(f"HEP events: {len(hep_events)}")

    epochs = mne.Epochs(
        raw,
        events=hep_events,
        event_id={"R": event_id},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks="eeg",
        reject={"eeg": reject_eeg},
        preload=True,
        detrend=None,
        reject_by_annotation=True,
    )

    if len(epochs) == 0:
        fig_dir = _out_fig_dir_for_fif(fif_path, out_dir)
        stem = fif_path.stem.replace("_clean", "")

        # Save drop log for debugging
        fig_drop = epochs.plot_drop_log(show=False)
        out_png = fig_dir / f"{stem}_hep_drop_log.png"
        out_svg = fig_dir / f"{stem}_hep_drop_log.svg"
        fig_drop.savefig(out_png, dpi=300, bbox_inches="tight")
        fig_drop.savefig(out_svg, bbox_inches="tight")
        plt.close(fig_drop)

        raise RuntimeError(
            "All epochs were dropped. Common causes: reject threshold too strict or noisy channels. "
            f"A drop-log figure was saved to: {out_png} and {out_svg}. "
            "Try increasing --reject_eeg (e.g., 200e-6 or 300e-6) or inspect channel quality."
        )

    evoked = epochs.average()

    roi_chs = _select_roi(raw, roi)

    fig_dir = _out_fig_dir_for_fif(fif_path, out_dir)
    stem = fif_path.stem.replace("_clean", "")

    # Butterfly plot
    fig_butter = evoked.plot(spatial_colors=True, show=False)
    out_png = fig_dir / f"{stem}_hep.png"
    out_svg = fig_dir / f"{stem}_hep.svg"
    fig_butter.savefig(out_png, dpi=300, bbox_inches="tight")
    fig_butter.savefig(out_svg, bbox_inches="tight")
    plt.close(fig_butter)

    # ROI mean plot
    fig_roi = _plot_roi_evoked(evoked, roi_chs, title=f"HEP - {stem}")
    out_png2 = fig_dir / f"{stem}_hep_roi.png"
    out_svg2 = fig_dir / f"{stem}_hep_roi.svg"
    fig_roi.savefig(out_png2, dpi=300, bbox_inches="tight")
    fig_roi.savefig(out_svg2, bbox_inches="tight")
    plt.close(fig_roi)

    _log_info(
        "Saved: "
        + ", ".join([out_png.name, out_svg.name, out_png2.name, out_svg2.name])
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HEP (heartbeat-evoked potential) analysis")

    p.add_argument("--fif", type=str, default=None, help="Single EEG .fif to process")
    p.add_argument("--subject_id", type=str, default=None, help="Subject ID for batch processing")

    p.add_argument("--ecg_file", type=str, default=None, help="ECG peaks JSON for --fif mode")
    p.add_argument(
        "--ecg_dir",
        type=str,
        default=None,
        help=(
            "Directory containing ECG peaks JSONs for --subject_id mode (default: data/ecg/<id>/). "
            "Batch mode will auto-select only *Session_01*.json and *Session_02*.json."
        ),
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures when using --fif (default: <fif_dir>/figures).",
    )

    p.add_argument(
        "--roi",
        nargs="+",
        default=None,
        help="ROI channel names, e.g. --roi Fz Cz Pz. Default uses [Fz,Cz,Pz,C3,C4] if present.",
    )

    p.add_argument("--tmin", type=float, default=-0.3)
    p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument(
        "--baseline",
        type=float,
        nargs=2,
        default=[-0.2, -0.05],
        help="Baseline interval in seconds (must not include 0ms). Default: -0.2 -0.05",
    )
    p.add_argument(
        "--reject_eeg",
        type=float,
        default=100e-6,
        help="EEG rejection threshold in Volts. Default: 100e-6 (100 µV)",
    )
    p.add_argument("--event_id", type=int, default=999, help="Event id for HEP epochs")

    return p


def _resolve_fif_path_maybe_under_processed(fif_arg: str) -> Path:
    """Resolve a FIF path; if missing, try data/processed/**/<name>.

    This is a convenience for common invocations like:
      --fif 251222_AK_sess02_clean.fif
    when the file actually lives in data/processed/<subject_id>/.
    """
    p = Path(fif_arg)
    if p.exists():
        return p

    # Only try fuzzy search when user provided a bare name or relative path
    name = p.name
    candidates = sorted((Path("data") / "processed").glob(f"**/{name}"))
    if len(candidates) == 1:
        _log_warn(f"FIF not found at '{p}'. Using: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"FIF not found: {p}. Multiple candidates under data/processed: "
            + ", ".join(str(c) for c in candidates)
        )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.fif is not None and args.subject_id is not None:
        raise ValueError("Provide only one of --fif or --subject_id (mutually exclusive)")
    if args.fif is None and args.subject_id is None:
        raise ValueError("Provide either --fif or --subject_id")

    out_dir = Path(args.out_dir) if args.out_dir is not None else None

    if args.fif is not None:
        if args.ecg_file is None:
            raise ValueError("--ecg_file is required when using --fif")

        fif_path = _resolve_fif_path_maybe_under_processed(args.fif)
        ecg_path = Path(args.ecg_file)
        if not fif_path.exists():
            raise FileNotFoundError(
                f"FIF not found: {fif_path}. Try an explicit path like data/processed/<id>/<name>.fif"
            )
        if not ecg_path.exists():
            raise FileNotFoundError(f"ECG JSON not found: {ecg_path}")

        process_one_pair(
            fif_path=fif_path,
            ecg_json=ecg_path,
            out_dir=out_dir,
            roi=args.roi,
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            baseline=(float(args.baseline[0]), float(args.baseline[1])),
            reject_eeg=float(args.reject_eeg),
            event_id=int(args.event_id),
        )
        return

    # Batch mode
    subject_id = str(args.subject_id)
    fif_paths = _resolve_input_fifs(subject_id)

    if args.ecg_dir is None:
        ecg_dir = Path("data") / "ecg" / subject_id
    else:
        ecg_dir = Path(args.ecg_dir)

    # Select only session ECG peaks (Session_01/Session_02)
    ecg_jsons = _resolve_ecg_session_jsons(ecg_dir)

    if len(fif_paths) != len(ecg_jsons):
        raise RuntimeError(
            "Processing mismatch error: number of FIF files and ECG JSON files differ. "
            f"fif={len(fif_paths)} json={len(ecg_jsons)}. "
            "Ensure 1:1 correspondence for sess01/sess02 <-> Session_01/Session_02."
        )

    # In batch mode, figures go under data/processed/<id>/figures by default
    out_dir_batch = Path("data") / "processed" / subject_id

    for fif_path, ecg_json in zip(sorted(fif_paths), sorted(ecg_jsons)):
        _log_info(f"Pairing: {fif_path.name} <-> {ecg_json.name}")
        process_one_pair(
            fif_path=fif_path,
            ecg_json=ecg_json,
            out_dir=out_dir_batch,
            roi=args.roi,
            tmin=float(args.tmin),
            tmax=float(args.tmax),
            baseline=(float(args.baseline[0]), float(args.baseline[1])),
            reject_eeg=float(args.reject_eeg),
            event_id=int(args.event_id),
        )


if __name__ == "__main__":
    main()
