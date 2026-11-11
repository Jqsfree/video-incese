"""Import fully human-reviewed positive MP4 clips as supervised temporal windows.

No VLM, no external services. Predicted labels are never trusted: only clips
explicitly confirmed by the operator may generate positive labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from video_active import CSV_COLUMNS
from video_temporal_cv import FEATURE_VERSION, scan_temporal


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def import_positives(
    clips: list[Path],
    output: Path,
    reports_dir: Path,
    *,
    confirmed_reviewed: bool = False,
    temporal_step: float = 1.0,
    temporal_fps: float = 4.0,
    edge_trim: float = .25,
) -> dict:
    """Create report-backed labels without needing uncut originals.

    A manually cut clip is positive at the CLIP level. Frames near cut
    boundaries and windows with no detected layout are left 'unsure' rather
    than being silently converted into confident window-level supervision.
    """
    if not confirmed_reviewed:
        raise ValueError("Explicit --confirmed-reviewed required: clips must have been reviewed as positives")
    if not clips:
        raise ValueError("At least one --clip is required")
    if not (0.25 <= temporal_step <= 5 and 2 <= temporal_fps <= 16
            and math.isfinite(edge_trim) and 0 <= edge_trim <= 5):
        raise ValueError("Invalid temporal_step, temporal_fps, or edge_trim")
    from video_incese import Settings, classify, probe

    output = output.expanduser().resolve()
    reports_dir = reports_dir.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("Output should be a .csv file")
    unique_paths = [p.expanduser().resolve(strict=True) for p in clips]
    if len(set(unique_paths)) != len(unique_paths):
        raise ValueError("Duplicate positive clip path")
    if output in unique_paths:
        raise ValueError("Do not overwrite an input MP4")
    settings = Settings(
        content_mode="temporal-cv", temporal_step=temporal_step,
        temporal_fps=temporal_fps, min_duration=.25, merge_gap=0,
    )
    rows: list[dict] = []
    reports: list[dict] = []
    label_count = 0
    unsure_count = 0

    for idx, path in enumerate(unique_paths, 1):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            raise ValueError(f"Expected a video file: {path}")
        meta = probe(path)
        duration = float(meta["duration"])
        fingerprint = _sha256(path)
        prefix = re.sub(r"[^\w.-]+", "_", path.stem)[:75]
        report_path = reports_dir / f"{idx:02d}_{prefix}_{fingerprint[:10]}.json"
        if report_path == output:
            raise ValueError("Output CSV path conflicts with report path")
        accepted, windows, stats = scan_temporal(path, [(0.0, duration)], settings, classify)
        if not windows:
            raise RuntimeError(f"No temporal windows from positive clip: {path}")
        annotated = 0
        unresolved = 0
        for row in windows:
            # Classification is model-independent and cannot decide if the
            # operator really regards every single window as a positive.
            # Only interior/valid-layout windows inherit the human clip label.
            valid_geometry = row["reason"] not in {"frame_error", "no_comparison_layout"}
            internal = (row["start"] >= edge_trim - 1e-6
                        and row["end"] <= duration - edge_trim + 1e-6)
            label = "keep" if valid_geometry and internal else "unsure"
            annotated += label == "keep"
            unresolved += label == "unsure"
            rows.append({
                "report_path": str(report_path), "window_id": row["window_id"],
                "start": row["start"], "end": row["end"],
                "suggested": "keep" if row["accepted"] else "reject",
                "confidence": row["heuristic_score"], "label": label,
                "note": ("whole_clip_human_positive" if label == "keep" else
                         "manual_check_layout_or_cut_boundary"),
                "preview": str(path),
            })
        record = {
            "input": str(path),
            "video": meta, "source_sha256": fingerprint,
            "manual_ground_truth": {
                "origin": "human_edited_clip", "scope": "whole_clip_positive",
                "edge_trim_seconds": edge_trim,
                "note": "NOT aligned to original uncut video; no negative truth supplied.",
            },
            "settings": {"temporal_step": temporal_step, "temporal_fps": temporal_fps},
            "temporal": stats,
            "temporal_windows": windows,
        }
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append({"clip": str(path), "report": str(report_path),
                        "sha256": fingerprint, "duration_seconds": duration,
                        "windows": len(windows), "label_keep": annotated,
                        "label_unsure": unresolved,
                        "geometry_missed_windows": sum(
                            w["reason"] == "no_comparison_layout" for w in windows),
                        "rule_rejected_but_human_positive_windows": sum(
                            not w["accepted"] and w["reason"] not in {"frame_error", "no_comparison_layout"}
                            for w in windows)})
        label_count += annotated
        unsure_count += unresolved
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "labels_csv": str(output), "report_dir": str(reports_dir),
        "positive_clips": len(reports), "positive_windows": label_count,
        "needs_manual_review_windows": unsure_count,
        "negative_windows": 0,
        "warning": "Positive-only labels cannot train the balanced model. "
                   "Add human-confirmed reject windows from complete original video.",
        "clips": reports,
    }
