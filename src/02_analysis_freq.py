#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install requirements:
# pip install mne numpy pandas scipy matplotlib tqdm scikit-learn asrpy mne-icalabel pybv

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

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


def _out_fig_dir_for_fif(fif_path: Path, out_dir: Optional[Path]) -> Path:
    if out_dir is None:
        base = fif_path.parent
    else:
        base = out_dir
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _plot_psd_curve(freqs: np.ndarray, psd: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, 10 * np.log10(psd), lw=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def process_one_fif(
    fif_path: Path,
    out_dir: Optional[Path],
    fmin: float,
    fmax: float,
    n_fft: int,
    n_overlap: int,
) -> None:
    _log_info(f"Loading: {fif_path}")
    raw = mne.io.read_raw_fif(str(fif_path), preload=False)

    # Continuous Welch PSD
    spec = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks="eeg",
        n_fft=n_fft,
        n_overlap=n_overlap,
        reject_by_annotation=True,
    )
    psd, freqs = spec.get_data(return_freqs=True)  # (n_ch, n_freq)

    # Average across channels for a quick curve
    psd_mean = psd.mean(axis=0)

    fig_dir = _out_fig_dir_for_fif(fif_path, out_dir)
    stem = fif_path.stem.replace("_clean", "")

    fig_curve = _plot_psd_curve(freqs, psd_mean, title=f"PSD (Welch) - {stem}")
    out_png = fig_dir / f"{stem}_psd.png"
    out_svg = fig_dir / f"{stem}_psd.svg"
    fig_curve.savefig(out_png, dpi=300, bbox_inches="tight")
    fig_curve.savefig(out_svg, bbox_inches="tight")
    plt.close(fig_curve)

    # Topomap at a representative band (use average across freqs to mimic broad topography)
    # We plot mean PSD across all frequencies in [fmin, fmax] for a stable map.
    topo_vals = psd.mean(axis=1)
    fig_topo, ax = plt.subplots(figsize=(4.5, 4.5))
    mne.viz.plot_topomap(
        topo_vals,
        raw.info,
        axes=ax,
        contours=0,
        show=False,
    )
    ax.set_title(f"Topomap mean PSD ({fmin}-{fmax} Hz) - {stem}")
    fig_topo.tight_layout()

    out_png2 = fig_dir / f"{stem}_psd_topo.png"
    out_svg2 = fig_dir / f"{stem}_psd_topo.svg"
    fig_topo.savefig(out_png2, dpi=300, bbox_inches="tight")
    fig_topo.savefig(out_svg2, bbox_inches="tight")
    plt.close(fig_topo)

    _log_info(f"Saved: {out_png.name}, {out_svg.name}, {out_png2.name}, {out_svg2.name}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frequency-domain check (Welch PSD) for preprocessed EEG")

    p.add_argument("--fif", type=str, default=None, help="Single .fif file to process")
    p.add_argument("--subject_id", type=str, default=None, help="Subject ID to batch-process data/processed/<id>/*_clean.fif")

    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures when using --fif (default: <fif_dir>/figures).",
    )

    p.add_argument("--fmin", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=45.0)
    p.add_argument(
        "--window_s",
        type=float,
        default=2.0,
        help="Welch window length in seconds (affects frequency resolution)",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Welch overlap fraction (0-<1).",
    )

    return p


def _resolve_fif_path_maybe_under_processed(fif_arg: str) -> Path:
    p = Path(fif_arg)
    if p.exists():
        return p
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

    fif_paths: List[Path]
    out_dir: Optional[Path]

    if args.fif is not None:
        fif_paths = [_resolve_fif_path_maybe_under_processed(args.fif)]
        if not fif_paths[0].exists():
            raise FileNotFoundError(f"FIF not found: {fif_paths[0]}")
        out_dir = Path(args.out_dir) if args.out_dir is not None else None
    else:
        fif_paths = _resolve_input_fifs(str(args.subject_id))
        # For batch mode, figures go under data/processed/<id>/figures
        out_dir = Path("data") / "processed" / str(args.subject_id)

    # Convert window seconds to n_fft/n_overlap (uses sfreq from each file)
    for fif in fif_paths:
        raw = mne.io.read_raw_fif(str(fif), preload=False)
        sfreq = float(raw.info["sfreq"])
        n_fft = int(round(args.window_s * sfreq))
        n_fft = max(n_fft, 256)
        n_overlap = int(round(args.overlap * n_fft))
        process_one_fif(
            fif_path=fif,
            out_dir=out_dir,
            fmin=float(args.fmin),
            fmax=float(args.fmax),
            n_fft=n_fft,
            n_overlap=n_overlap,
        )


if __name__ == "__main__":
    main()
