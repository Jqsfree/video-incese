"""Human-in-the-loop active learning for temporal video windows.

Learning is explicit: no unreviewed prediction becomes a label. Reports, CSV
annotations, and versioned model JSON form a reproducible training contract.
No internet/GPU/LLM or scikit-learn dependency is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from video_temporal_cv import FEATURE_NAMES, FEATURE_VERSION

CSV_COLUMNS = ["report_path", "window_id", "start", "end", "suggested",
               "confidence", "label", "note", "preview"]
LABELS = {"keep": 1, "reject": 0}
MODEL_VERSION = "active-logistic-1"


def predict_proba(model: dict, features: dict[str, float]) -> float:
    if model.get("model_version") != MODEL_VERSION or model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Incompatible active-learning model/schema version")
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Active-learning feature order mismatch")
    x = np.asarray([float(features[n]) for n in FEATURE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite active-learning features")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if mean.shape != x.shape or scale.shape != x.shape or weights.shape != x.shape or np.any(scale <= 0):
        raise ValueError("Malformed active-learning model dimensions")
    z = float(((x - mean) / scale) @ weights + float(model["bias"]))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def _load_report(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve(strict=True)
    with path.open(encoding="utf-8") as f:
        report = json.load(f)
    if report.get("temporal", {}).get("mode") != FEATURE_VERSION:
        raise ValueError(f"Not a {FEATURE_VERSION} report: {path}")
    windows = report.get("temporal_windows")
    if not isinstance(windows, list):
        raise ValueError(f"Temporal windows missing: {path}")
    return path, report


def _choose_windows(windows: list[dict], budget: int) -> list[dict]:
    """Uncertainty plus feature-space/time diversity, with both decision classes."""
    if budget < 1:
        raise ValueError("budget must be positive")
    candidates = [w for w in windows if w["reason"] != "frame_error"]
    if not candidates:
        return []
    items = sorted(candidates, key=lambda w: (abs(
        (w["learned_probability"] if w.get("learned_probability") is not None
         else w["heuristic_score"]) - .5), w["window_id"]))
    selected = []
    for accepted in (True, False):
        hit = next((w for w in items if w["accepted"] == accepted), None)
        if hit is not None and len(selected) < budget:
            selected.append(hit)

    # Greedy diversity adds most learning value; separated windows from a
    # single scene are not treated as independent examples by default.
    x = np.array([[float(w["features"][n]) for n in FEATURE_NAMES] for w in candidates])
    scale = np.maximum(np.ptp(x, axis=0), .01)
    by_id = {w["window_id"]: vec for w, vec in zip(candidates, x)}
    ids = {w["window_id"] for w in selected}
    while len(selected) < min(budget, len(candidates)):
        best = None
        best_score = -float("inf")
        for w in candidates:
            if w["window_id"] in ids:
                continue
            p = w["learned_probability"] if w.get("learned_probability") is not None else w["heuristic_score"]
            uncertain = 1 - 2 * abs(float(p) - .5)
            diversity = (min(
                float(np.linalg.norm((by_id[w["window_id"]] - by_id[s["window_id"]]) / scale))
                for s in selected
            ) / math.sqrt(len(FEATURE_NAMES))) if selected else 1.0
            time_gap = min((abs(w["start"] - s["start"]) for s in selected), default=100.0)
            adjacent = .12 if time_gap < 2 else 0.0
            merit = .65 * uncertain + .35 * diversity - adjacent
            if merit > best_score:
                best_score = merit
                best = w
        if best is None:
            break
        ids.add(best["window_id"])
        selected.append(best)
    return sorted(selected, key=lambda w: w["window_id"])


def _preview(source: Path, output: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for --preview-dir")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end-start:.3f}",
           "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "24", "-pix_fmt", "yuv420p", str(output)]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Preview MP4 export failed: {p.stderr}")


def queue(report_path: Path, output: Path, budget: int, preview_dir: Path | None = None) -> dict:
    path, report = _load_report(report_path)
    if output.expanduser().resolve() == path:
        raise ValueError("The CSV output must not overwrite report.json")
    windows = _choose_windows(report["temporal_windows"], budget)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(report["input"])
    rows = []
    for w in windows:
        destination = ""
        if preview_dir is not None:
            filename = f"review_{w['window_id']:05d}_{w['start']:.3f}-{w['end']:.3f}.mp4"
            clip = preview_dir.expanduser().resolve() / filename
            _preview(source, clip, w["start"], w["end"])
            destination = str(clip)
        prob = w["learned_probability"] if w.get("learned_probability") is not None else w["heuristic_score"]
        rows.append({
            "report_path": str(path), "window_id": w["window_id"],
            "start": w["start"], "end": w["end"],
            "suggested": "keep" if w["accepted"] else "reject",
            "confidence": round(float(prob), 4), "label": "", "note": "", "preview": destination,
        })
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"queue": str(output), "windows": len(rows), "previews": preview_dir is not None}


def train(label_paths: list[Path], output: Path) -> dict:
    """L2-regularized balanced logistic regression; explicitly curated labels only."""
    data = []
    seen: dict[tuple[str, int], int] = {}
    cache: dict[str, dict] = {}
    for input_path in label_paths:
        with input_path.expanduser().open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"report_path", "window_id", "label"}.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"CSV lacks required fields: {input_path}")
            for row in reader:
                raw = row["label"].strip().lower()
                if raw in ("", "unsure", "skip"):
                    continue
                if raw not in LABELS:
                    raise ValueError(f"Invalid label {raw!r} in {input_path}; use keep/reject/unsure")
                path = str(Path(row["report_path"]).expanduser().resolve(strict=True))
                if path not in cache:
                    cache[path] = _load_report(Path(path))[1]
                index = int(row["window_id"])
                windows = cache[path]["temporal_windows"]
                if index < 1 or index > len(windows):
                    raise ValueError(f"Window ID not found: {path}#{index}")
                w = windows[index - 1]
                if w["window_id"] != index or w["reason"] == "frame_error":
                    raise ValueError(f"Invalid or undecodable labeled window: {path}#{index}")
                key = (path, index)
                if key in seen:
                    if seen[key] != LABELS[raw]:
                        raise ValueError(f"Conflicting labels for {path}#{index}")
                    continue
                seen[key] = LABELS[raw]
                data.append((key, w["features"], LABELS[raw]))
    y = np.asarray([r[2] for r in data], dtype=np.float64)
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    if len(data) < 12 or min(positives, negatives) < 4:
        raise ValueError(f"Need >=12 manually labeled windows, >=4 per class; "
                         f"received {positives} keep and {negatives} reject")
    x = np.asarray([[float(feats[n]) for n in FEATURE_NAMES] for _, feats, _ in data],
                   dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Training features contain non-finite values")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .03)
    normalized = (x - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    balances = np.where(y > 0, len(y)/(2*positives), len(y)/(2*negatives))
    for _ in range(900):
        p = 1 / (1 + np.exp(-np.clip(normalized @ weights + bias, -30, 30)))
        diff = balances * (p-y)
        weights -= .16 * (normalized.T @ diff / len(y) + .05 * weights)
        bias -= .16 * float(np.mean(diff))
    training_prediction = 1 / (1 + np.exp(-np.clip(normalized @ weights + bias, -30, 30)))
    records = [[list(key), int(label)] for key, _, label in sorted(data)]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    result = {
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES), "created_utc": datetime.now(timezone.utc).isoformat(),
        "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
        "bias": float(bias), "training_samples": len(data), "labeled_keep": positives,
        "labeled_reject": negatives, "label_fingerprint": fingerprint,
        "train_set_accuracy": round(float(np.mean((training_prediction >= .5) == y)), 4),
        "evaluation_warning": "Training-set fit only; not a held-out or independent quality metric.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output)
    return {"model": str(output), "samples": len(data), "keep": positives,
            "reject": negatives, "label_fingerprint": fingerprint}


def load_model(path: Path | None) -> dict | None:
    if path is None:
        return None
    result = json.loads(path.expanduser().read_text(encoding="utf-8"))
    # Check before processing the first video frame, not after inference.
    predict_proba(result, {name: 0.0 for name in FEATURE_NAMES})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review queue and human-labeled active learning")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("queue", help="Select uncertain/diverse temporal windows for annotation")
    q.add_argument("--report", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--budget", type=int, default=24)
    q.add_argument("--preview-dir", type=Path, help="Optional MP4 clips for reviewed windows")
    t = sub.add_parser("train", help="Train from edited human labels; no pseudo-labels")
    t.add_argument("--labels", type=Path, action="append", required=True)
    t.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "queue":
            result = queue(args.report, args.output, args.budget, args.preview_dir)
        else:
            result = train(args.labels, args.output)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
