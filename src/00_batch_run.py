#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConditionRow:
    session_id: str
    group: str
    set1_cond: str
    set2_cond: str


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _repo_root() -> Path:
    # src/00_batch_run.py -> repo root is parent of src
    return Path(__file__).resolve().parents[1]


def _default_conditions_csv() -> Path:
    return _repo_root() / "data" / "conditions" / "conditions.csv"


def _read_conditions(csv_path: Path) -> List[ConditionRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"conditions.csv not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"conditions.csv has no header: {csv_path}")

        required = {"session_id", "group", "Set1_Cond", "Set2_Cond"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"conditions.csv missing columns {sorted(missing)}: {csv_path}")

        rows: List[ConditionRow] = []
        for i, row in enumerate(reader, start=2):
            sid = (row.get("session_id") or "").strip()
            grp = (row.get("group") or "").strip()
            s1 = (row.get("Set1_Cond") or "").strip()
            s2 = (row.get("Set2_Cond") or "").strip()
            if not sid:
                _log_warn(f"Skipping empty session_id at line {i}: {csv_path}")
                continue
            rows.append(ConditionRow(session_id=sid, group=grp, set1_cond=s1, set2_cond=s2))

    if not rows:
        raise ValueError(f"No valid rows found in: {csv_path}")
    return rows


def _processed_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "processed" / subject_id


def _raw_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "raw" / subject_id


def _trigger_dir(subject_id: str) -> Path:
    return _repo_root() / "data" / "trigger" / subject_id


def _has_trigger_csvs(subject_id: str) -> bool:
    tdir = _trigger_dir(subject_id)
    if not tdir.exists():
        return False
    return any(tdir.glob("*_actichamp_trigger_session.csv"))


def _has_clean_fifs(subject_id: str) -> bool:
    pdir = _processed_dir(subject_id)
    if not pdir.exists():
        return False
    return any(pdir.glob("*_clean.fif"))


def _write_conditions_manifest(subject_id: str, cond: ConditionRow) -> Path:
    out_dir = _processed_dir(subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "session_id": cond.session_id,
        "group": cond.group,
        "Set1_Cond": cond.set1_cond,
        "Set2_Cond": cond.set2_cond,
        "session_condition_map": {
            "sess01": cond.set1_cond,
            "sess02": cond.set2_cond,
        },
    }
    out_path = out_dir / "conditions_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out_path


def _run_script(script: Path, argv: List[str], cwd: Path) -> Tuple[int, str]:
    cmd = [sys.executable, str(script), *argv]
    _log_info("Running: " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode), " ".join(cmd)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Batch runner: iterate data/conditions/conditions.csv session_id and run pipelines "
            "(01_preprocess -> 02_analysis_tfr -> 03_analysis_hep)."
        )
    )
    p.add_argument(
        "--conditions_csv",
        type=str,
        default=str(_default_conditions_csv()),
        help="Path to conditions.csv (default: data/conditions/conditions.csv)",
    )
    p.add_argument(
        "--check_only",
        action="store_true",
        help=(
            "Do not run pipelines. Only scan outputs under data/processed/<id> and print failure list."
        ),
    )
    return p


def _has_file(pdir: Path, patterns: Sequence[str]) -> bool:
    return any(pdir.glob(pat) for pat in patterns)


def _scan_subject_status(subject_id: str) -> List[str]:
    issues: List[str] = []

    if not _raw_dir(subject_id).exists():
        issues.append("missing_raw")
        return issues

    pdir = _processed_dir(subject_id)
    if not pdir.exists():
        issues.append("missing_processed_dir")
        return issues

    if not (pdir / "conditions_manifest.json").exists():
        issues.append("missing_conditions_manifest")

    # Expect 2 sessions (sess01/sess02)
    for sess in ("sess01", "sess02"):
        if not (pdir / f"{subject_id}_{sess}_clean.fif").exists():
            issues.append(f"missing_{sess}_clean_fif")
        if not (pdir / f"{subject_id}_{sess}_events.npy").exists():
            issues.append(f"missing_{sess}_events")

        # HEP per-session figures (kept)
        if not _has_file(pdir, [f"{subject_id}_{sess}_hep.png", f"{subject_id}_{sess}_hep.svg"]):
            issues.append(f"missing_{sess}_hep")
        if not _has_file(pdir, [f"{subject_id}_{sess}_hep_roi.png", f"{subject_id}_{sess}_hep_roi.svg"]):
            issues.append(f"missing_{sess}_hep_roi")

    # TFR outputs (subject-level, session-typed)
    if not (pdir / f"{subject_id}_tfr_target.h5").exists():
        issues.append("missing_tfr_target")
    if not (pdir / f"{subject_id}_tfr_control.h5").exists():
        issues.append("missing_tfr_control")

    # HEP evoked outputs (subject-level, session-typed)
    if not (pdir / f"{subject_id}_hep_target-ave.fif").exists():
        issues.append("missing_hep_target_evoked")
    if not (pdir / f"{subject_id}_hep_control-ave.fif").exists():
        issues.append("missing_hep_control_evoked")

    return issues


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    root = _repo_root()
    conditions_csv = Path(args.conditions_csv)
    if not conditions_csv.is_absolute():
        conditions_csv = (root / conditions_csv).resolve()

    rows = _read_conditions(conditions_csv)

    script_dir = Path(__file__).resolve().parent
    preprocess_py = script_dir / "01_preprocess.py"
    tfr_py = script_dir / "02_analysis_tfr.py"
    hep_py = script_dir / "03_analysis_hep.py"

    failures: Dict[str, List[str]] = {}

    if args.check_only:
        for cond in rows:
            sid = cond.session_id
            issues = _scan_subject_status(sid)
            if issues:
                failures[sid] = issues

        if failures:
            _log_warn("Check-only: outputs missing or incomplete:")
            for sid, reasons in failures.items():
                _log_warn(f"  {sid}: {', '.join(reasons)}")
            raise SystemExit(1)

        _log_info("Check-only: all subjects look complete")
        return

    for cond in rows:
        sid = cond.session_id
        _log_info(f"=== Subject: {sid} ===")

        if not _raw_dir(sid).exists():
            _log_warn(f"Raw directory missing; skipping: data/raw/{sid}")
            failures.setdefault(sid, []).append("missing_raw")
            continue

        manifest_path = _write_conditions_manifest(sid, cond)
        _log_info(f"Wrote manifest: {manifest_path}")

        # 01_preprocess: run only when triggers are present (robustness). If missing, skip.
        if _has_trigger_csvs(sid):
            code, cmdline = _run_script(preprocess_py, ["--subject_id", sid], cwd=root)
            if code != 0:
                _log_warn(f"Preprocess failed (exit={code}). Continuing to next subject.\n  cmd: {cmdline}")
                failures.setdefault(sid, []).append(f"preprocess_exit_{code}")
                continue
        else:
            _log_warn(f"Trigger CSVs missing; skipping preprocess: data/trigger/{sid}")

        # Downstream analyses require *_clean.fif
        if not _has_clean_fifs(sid):
            _log_warn(f"No *_clean.fif under data/processed/{sid}; skipping analyses")
            failures.setdefault(sid, []).append("missing_clean_fif")
            continue

        code, cmdline = _run_script(tfr_py, ["--subject_id", sid], cwd=root)
        if code != 0:
            _log_warn(f"TFR analysis failed (exit={code}). Continuing.\n  cmd: {cmdline}")
            failures.setdefault(sid, []).append(f"tfr_exit_{code}")
            continue

        # HEP needs ECG jsons too; the script itself will validate.
        code, cmdline = _run_script(hep_py, ["--subject_id", sid], cwd=root)
        if code != 0:
            _log_warn(f"HEP analysis failed (exit={code}). Continuing.\n  cmd: {cmdline}")
            failures.setdefault(sid, []).append(f"hep_exit_{code}")
            continue

    if failures:
        _log_warn("Batch finished with issues:")
        for sid, reasons in failures.items():
            _log_warn(f"  {sid}: {', '.join(reasons)}")
        raise SystemExit(1)

    _log_info("Batch finished successfully")


if __name__ == "__main__":
    main()
