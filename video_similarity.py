"""Positive-example video retrieval, not a calibrated semantic classifier.

Builds a case-balanced descriptor library from HUMAN-reviewed clip boundaries,
then scans ALL windows of a new video (including layout=none). Ranks candidate
segments by feature similarity. No unreviewed interval becomes a negative label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from video_content import _regions
from video_temporal_cv import FEATURE_VERSION, calculate_features, rule_score

PROFILE_VERSION = "positive-reference-v1"
FEATURES = (
    "reference_motion", "rendered_motion",
    "reference_flow", "rendered_flow",
    "reference_coherence", "rendered_coherence",
    "reference_edges", "rendered_edges",
    "dual_motion_fraction", "scene_cut_fraction",
    "reference_neutral_ratio", "reference_brightness",
    "rendered_neutral_ratio",
)
# These are engineering priors, NOT fitted feature importances.
WEIGHTS = {
    "reference_motion": 1.3, "rendered_motion": 1.0,
    "reference_flow": 1.4, "rendered_flow": 1.0,
    "reference_coherence": .35, "rendered_coherence": .25,
    "reference_edges": 1.0, "rendered_edges": .45,
    "dual_motion_fraction": .85, "scene_cut_fraction": .4,
    "reference_neutral_ratio": 1.0, "reference_brightness": .4,
    "rendered_neutral_ratio": .3,
}


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: numeric value required")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{context}: finite value required")
    return num


def load_positives(path: Path, *, expected_filename: str | None = None,
                   duration: float | None = None) -> list[dict]:
    """Support v0.5 truth JSON AND manually-verified original2 clip-ms JSON.

    Gaps in either file remain UNKNOWN; they are never synthesized as rejects.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "positives" in data:
        raw = data["positives"]
        source = data.get("source")
        if isinstance(source, dict):
            source = source.get("file")
        if not isinstance(raw, list):
            raise ValueError("positives must be a list")
        cases = [(x["start"], x["end"]) for x in raw]
    elif "clips" in data and "timeline" in data:
        raw = data["clips"]
        source = data.get("source", {}).get("file")
        if not isinstance(raw, list):
            raise ValueError("clips must be a list")
        cases = [(x["start_ms"] / 1000.0, x["end_ms"] / 1000.0)
                 for x in raw]
    else:
        raise ValueError("Expected reviewed positives or verified clips+timeline JSON")
    if expected_filename and source and Path(str(source)).name != expected_filename:
        raise ValueError(f"Annotations are for {source}, not {expected_filename}")
    if not cases:
        raise ValueError("No manually reviewed positive cases")
    spans = []
    for i, (start, end) in enumerate(cases, 1):
        a, b = _finite(start, f"case {i} start"), _finite(end, f"case {i} end")
        if a < 0 or b <= a:
            raise ValueError(f"case {i}: expected 0 <= start < end")
        if duration is not None and b > duration + .08:
            raise ValueError(f"case {i}: exceeds source duration {duration:.3f}s")
        spans.append((a, b))
    spans.sort()
    for (a, b), (c, d) in zip(spans, spans[1:]):
        if b > c + 1e-6:
            raise ValueError("Human positive cases overlap")
    return [{"case_id": i, "start": round(a, 3), "end": round(b, 3)}
            for i, (a, b) in enumerate(spans, 1)]


def _windows(start: float, end: float, step: float) -> list[tuple[float, float]]:
    if end - start < .75:
        return []
    n = int(math.ceil((end - start) / step))
    spans = []
    for i in range(n):
        a, b = start + i * step, min(end, start + (i + 1) * step)
        if b - a < .75:
            if spans:
                spans[-1] = (spans[-1][0], b)
            break
        spans.append((round(a, 4), round(b, 4)))
    return spans


def _read_window(cap: cv2.VideoCapture, start: float, end: float, fps: float
                 ) -> list[np.ndarray]:
    duration = end - start
    count = max(4, int(math.ceil(duration * fps)) + 1)
    edge = min(.03, duration / 8)
    result = []
    for t in np.linspace(start + edge, end - edge, count):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0., float(t)) * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode video at {t:.3f}s")
        result.append(frame)
    return result


def _appearance(frames: list[np.ndarray], layout: str, settings: object) -> dict[str, float]:
    sampled = (frames[0], frames[len(frames)//2], frames[-1])
    values = {"reference_neutral_ratio": [], "reference_brightness": [],
              "rendered_neutral_ratio": []}
    for frame in sampled:
        reference, rendered = _regions(frame, layout, settings.pip_x, settings.pip_y)
        ref = cv2.cvtColor(cv2.resize(reference, (96, 64)), cv2.COLOR_BGR2HSV)
        ren = cv2.cvtColor(cv2.resize(rendered, (96, 64)), cv2.COLOR_BGR2HSV)
        values["reference_neutral_ratio"].append(float(
            np.mean((ref[:, :, 1] <= 40) & (ref[:, :, 2] >= 45))))
        values["reference_brightness"].append(float(np.median(ref[:, :, 2]) / 255.0))
        values["rendered_neutral_ratio"].append(float(
            np.mean((ren[:, :, 1] <= 40) & (ren[:, :, 2] >= 45))))
    return {k: round(float(np.median(v)), 5) for k, v in values.items()}


def _descriptor(frames: list[np.ndarray], layout: str, settings: object) -> dict[str, float]:
    features = calculate_features(frames, layout, settings)
    features.update(_appearance(frames, layout, settings))
    return {name: features[name] for name in FEATURES}


def _geometry(frames: list[np.ndarray], settings: object) -> str:
    from video_incese import classify
    layout, _, _ = classify(frames[len(frames)//2], settings)
    return layout


def _positive_layouts(frames: list[np.ndarray], settings: object) -> tuple[list[str], str]:
    geometry = _geometry(frames, settings)
    if geometry in {"pip", "split"}:
        return [geometry], "geometry"
    # Verified positive: layout is NOT actually absent; geometry is uncertain.
    # Keep both possibilities as separate hypotheses, with provenance.
    return ["pip", "split"], "forced_positive_hypotheses"


def _extract_profile_cases(source: Path, positives: list[dict], settings: object,
                           *, step: float, fps: float) -> tuple[list[dict], list[dict]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open original positive source: {source}")
    cases = []
    failures = []
    try:
        for case in positives:
            examples = []
            for start, end in _windows(case["start"], case["end"], step):
                try:
                    frames = _read_window(cap, start, end, fps)
                    layouts, origin = _positive_layouts(frames, settings)
                    for layout in layouts:
                        examples.append({"start": start, "end": end,
                                         "layout": layout, "layout_evidence": origin,
                                         "features": _descriptor(frames, layout, settings)})
                except (RuntimeError, ValueError, cv2.error) as exc:
                    failures.append({"case_id": case["case_id"], "start": start,
                                     "end": end, "error": str(exc)})
            cases.append({**case, "examples": examples,
                          "status": "featured" if examples else "no_decodable_features"})
    finally:
        cap.release()
    return cases, failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _feature_scales(cases: list[dict]) -> dict[str, float]:
    # Cases contribute equally, regardless of the number of video windows.
    means = []
    for case in cases:
        items = [x["features"] for x in case["examples"]]
        if items:
            means.append([float(np.median([x[n] for x in items])) for n in FEATURES])
    if not means:
        raise ValueError("No examples decoded from human cases")
    x = np.asarray(means, dtype=np.float64)
    median = np.median(x, axis=0)
    spread = 1.4826 * np.median(np.abs(x - median), axis=0)
    return {name: round(float(max(.08, s)), 5) for name, s in zip(FEATURES, spread)}


def build_profile(source: Path, truth: Path, settings: object, *,
                  step: float = 2.0, fps: float = 4.0) -> dict:
    if not (.75 <= step <= 10 and 2 <= fps <= 10):
        raise ValueError("Training window step must be .75..10s; fps must be 2..10")
    from video_incese import probe
    info = probe(source)
    positives = load_positives(truth, expected_filename=source.name, duration=info["duration"])
    cases, failures = _extract_profile_cases(source, positives, settings, step=step, fps=fps)
    if any(not c["examples"] for c in cases):
        missing = [x["case_id"] for x in cases if not x["examples"]]
        raise ValueError(f"Cannot feature every reviewed case {missing}; check decode and layout")
    annotated_bytes = truth.read_bytes()
    return {
        "profile_version": PROFILE_VERSION, "feature_version": FEATURE_VERSION,
        "training_source": source.name, "training_duration": round(info["duration"], 3),
        "training_file_sha256": _sha256(source),
        "annotation_sha256": hashlib.sha256(annotated_bytes).hexdigest(),
        "annotation_basename": truth.name,
        "selection": "positive-only_per-case_nearest_examples",
        "annotation_policy": "Unlabeled gaps are UNKNOWN, never automatically negative",
        "window_seconds": step, "sample_fps": fps,
        "weights": WEIGHTS, "scales": _feature_scales(cases),
        "features": list(FEATURES), "cases": cases, "decode_failures": failures,
        "limitations": [
            "Positive-only nearest neighbors rank visual/temporal similarity, not semantic correspondence.",
            "Forced PiP/split candidates for previously missed layouts may have wrong crops.",
            "Scoring the same source used to create a profile is retrieval/development, not independent validation.",
        ],
    }


def validate_profile(profile: dict) -> None:
    if profile.get("profile_version") != PROFILE_VERSION:
        raise ValueError("Unsupported similarity profile version")
    if tuple(profile.get("features", ())) != FEATURES:
        raise ValueError("Feature names or order incompatible")
    cases = profile.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Profile missing reviewed cases")
    for name in FEATURES:
        if (_finite(profile["scales"][name], name) <= 0
                or _finite(profile["weights"][name], name) <= 0):
            raise ValueError(f"Invalid scale/weight for {name}")
    for case in cases:
        if not case.get("examples"):
            raise ValueError("Profile contains empty reviewed case")
        for example in case["examples"]:
            if example.get("layout") not in {"pip", "split"}:
                raise ValueError("Unknown layout in profile")
            for name in FEATURES:
                _finite(example["features"][name], name)


def _score_features(descriptor: dict[str, float], profile: dict, layout: str) -> tuple[float, list[dict]]:
    """Case-balanced nearest positive similarity; ranking score is NOT P(keep)."""
    scores = []
    scale = profile["scales"]
    weights = profile["weights"]
    weight_sum = sum(weights[n] for n in FEATURES)
    for case in profile["cases"]:
        examples = [x for x in case["examples"] if x["layout"] == layout]
        # Cross-layout matching is permitted only as uncalibrated fallback.
        cross_layout = not examples
        if cross_layout:
            examples = case["examples"]
        nearest = float("inf")
        for ex in examples:
            distance = sum(weights[n] * min(3., abs(descriptor[n] - ex["features"][n]) / scale[n])
                           for n in FEATURES) / weight_sum
            nearest = min(nearest, distance)
        sim = float(math.exp(-.55 * nearest))
        if cross_layout:
            sim *= .92  # Unknown layout mismatch is a hypothesis, not proof.
        scores.append({"case_id": case["case_id"], "similarity": round(sim, 5),
                       "cross_layout": cross_layout})
    scores.sort(key=lambda x: (-x["similarity"], x["case_id"]))
    best = scores[:3]
    score = .8 * best[0]["similarity"] + .2 * sum(x["similarity"] for x in best) / len(best)
    return round(score, 5), best


def scan_windows(source: Path, profile: dict, settings: object, *,
                 step: float | None = None, fps: float | None = None,
                 exclude_reviewed: bool = False) -> tuple[list[dict], dict]:
    from video_incese import probe
    validate_profile(profile)
    info = probe(source)
    step = float(step or profile["window_seconds"])
    fps = float(fps or profile["sample_fps"])
    if not (.75 <= step <= 10 and 2 <= fps <= 10):
        raise ValueError("Scan step must be .75..10s; fps 2..10")
    same_source = (profile["training_source"] == source.name
                   and abs(float(profile["training_duration"]) - info["duration"]) <= .1
                   and profile.get("training_file_sha256") == _sha256(source))
    if exclude_reviewed and not same_source:
        raise ValueError("--exclude-reviewed requires the original training source")
    reviewed = [(c["start"], c["end"]) for c in profile["cases"]] if exclude_reviewed else []
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for similarity scan: {source}")
    windows = []
    failures = []
    try:
        for start, end in _windows(0., info["duration"], step):
            # Exclude any overlapping human-reviewed positive windows from same-source rediscovery.
            if any(max(0., min(end, b) - max(start, a)) > 0 for a, b in reviewed):
                continue
            try:
                frames = _read_window(cap, start, end, fps)
                geometry = _geometry(frames, settings)
                best = None
                # Both hypotheses run even when the geometric classifier returns none
                # or the wrong region; original layout gating is NOT repeated here.
                for layout in ("pip", "split"):
                    features = _descriptor(frames, layout, settings)
                    similarity, cases = _score_features(features, profile, layout)
                    _, passes_rule = rule_score({
                        **{name: features[name] for name in (
                            "reference_motion", "rendered_motion", "reference_flow",
                            "rendered_flow", "dual_motion_fraction", "scene_cut_fraction")}
                    })
                    candidate = {
                        "start": start, "end": end,
                        "layout": layout, "geometry_layout": geometry,
                        "layout_agrees": geometry == layout,
                        "similarity": similarity, "nearest_cases": cases,
                        "features": features, "old_motion_rule_pass": passes_rule,
                    }
                    if best is None or (candidate["similarity"], candidate["layout_agrees"]) > (
                            best["similarity"], best["layout_agrees"]):
                        best = candidate
                windows.append(best)
            except (ValueError, RuntimeError, cv2.error) as exc:
                failures.append({"start": start, "end": end, "error": str(exc)})
    finally:
        cap.release()
    return windows, {
        "input": str(source.resolve()), "duration": info["duration"],
        "windows_scored": len(windows), "unreadable_windows": failures,
        "geometry_none_but_scored": sum(w["geometry_layout"] == "none" for w in windows),
        "alternative_layout_selected": sum(w["geometry_layout"] != w["layout"] for w in windows),
        "same_source_as_training": same_source,
        "exclude_reviewed": exclude_reviewed,
        "score_is_probability": False,
    }


def _contiguous(windows: list[dict], minimum: float) -> list[tuple[float, float, int]]:
    out = []
    start = end = None
    group = 0
    for w in sorted(windows, key=lambda x: x["start"]):
        if w["similarity"] < minimum:
            if start is not None:
                group += 1
                out.append((start, end, group))
                start = end = None
            continue
        if start is None:
            start, end = w["start"], w["end"]
        elif abs(w["start"]-end) < 1e-3:
            end = w["end"]
        else:
            group += 1
            out.append((start, end, group))
            start, end = w["start"], w["end"]
    if start is not None:
        group += 1
        out.append((start, end, group))
    return out


def rank_segments(windows: list[dict], settings: object,
                  events: list[dict], *, min_similarity: float,
                  min_duration: float) -> list[dict]:
    from video_boundaries import split_spans
    if not (0 <= min_similarity <= 1 and min_duration >= .75):
        raise ValueError("min similarity must be 0..1, duration at least .75s")
    accepted = _contiguous(windows, min_similarity)
    slices = split_spans(accepted, events)
    ranked = []
    for piece in slices:
        a, b = piece["start"], piece["end"]
        if b-a + 1e-8 < min_duration:
            continue
        covered = [(max(0., min(b, w["end"])-max(a, w["start"])), w)
                   for w in windows]
        active = [(d, w) for d, w in covered if d > 0]
        if not active:
            continue
        similarity = sum(d*w["similarity"] for d, w in active) / sum(d for d, _ in active)
        closest = {}
        for d, w in active:
            for match in w["nearest_cases"]:
                cid = match["case_id"]
                closest[cid] = max(closest.get(cid, 0), match["similarity"])
        matched = sorted(closest.items(), key=lambda x: (-x[1], x[0]))[:3]
        layout_counts = Counter(w["layout"] for _, w in active)
        ranked.append({
            **piece, "duration": round(b-a, 3),
            "layout": layout_counts.most_common(1)[0][0],
            "similarity": round(similarity, 5),
            "top_reference_cases": [{"case_id": cid, "similarity": score}
                                    for cid, score in matched],
            "supported_windows": len(active),
            "selected_for_review": True,
            "label": "unknown_needs_human_review",
        })
    ranked.sort(key=lambda x: (-x["similarity"], x["start"]))
    for i, item in enumerate(ranked, 1):
        item["rank"] = i
    return ranked


def _write_csv(path: Path, ranked: list[dict]) -> None:
    fields = ["rank", "start", "end", "duration", "similarity", "layout",
              "top_reference_cases", "label", "output", "export_error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({
                "rank": row["rank"], "start": row["start"], "end": row["end"],
                "duration": row["duration"], "similarity": row["similarity"],
                "layout": row.get("layout", "mixed"),
                "top_reference_cases": ";".join(
                    str(x["case_id"]) for x in row["top_reference_cases"]),
                "label": row["label"], "output": row.get("output", ""),
                "export_error": row.get("export_error", ""),
            })


def run_scan(source: Path, profile: dict, output: Path, settings: object, *,
             step: float | None = None, fps: float | None = None,
             min_similarity: float = .62, min_duration: float = 2.0,
             boundary_mode: str = "scene", export_top: int = 0,
             exclude_reviewed: bool = False) -> dict:
    if boundary_mode not in {"scene", "off"} or export_top < 0:
        raise ValueError("boundary mode must be scene/off; export_top cannot be negative")
    windows, scan = scan_windows(source, profile, settings, step=step, fps=fps,
                                exclude_reviewed=exclude_reviewed)
    accepted = _contiguous(windows, min_similarity)
    events = []
    bounds = {"mode": "off", "accepted_events": 0}
    if boundary_mode == "scene" and accepted:
        from video_incese import classify
        from video_boundaries import scan_boundaries
        try:
            events, bounds = scan_boundaries(source, accepted, settings, classify)
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            bounds = {"mode": "scene", "status": "scan_failed", "error": str(exc),
                      "accepted_events": 0}
    ranked = rank_segments(windows, settings, events, min_similarity=min_similarity,
                           min_duration=min_duration)
    if export_top:
        from video_incese import export_clip
        for item in ranked[:export_top]:
            target = output / "ranked-clips" / (
                f"rank_{item['rank']:04d}_sim{item['similarity']:.3f}_"
                f"{item['start']:.3f}-{item['end']:.3f}.mp4")
            try:
                export_clip(source, target, item["start"], item["end"])
                item["output"] = str(target)
            except RuntimeError as exc:
                item["export_error"] = str(exc)
    result = {
        "input": scan["input"], "profile_version": PROFILE_VERSION,
        "training_source": profile["training_source"],
        "settings": {"step": step or profile["window_seconds"],
                     "fps": fps or profile["sample_fps"],
                     "min_similarity": min_similarity,
                     "min_duration": min_duration,
                     "boundary_mode": boundary_mode,
                     "export_top": export_top,
                     "exclude_reviewed": exclude_reviewed},
        "scan": scan, "boundaries": bounds, "boundary_events": events,
        "ranked": ranked, "windows": windows,
        "warning": ("RANK ONLY: similarity is not calibrated probability; "
                    "do not treat these candidates as verified matches. "
                    "Unknown time is not negative. Visual cuts may not equal case boundaries."),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranking.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    _write_csv(output / "ranking.csv", ranked)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human-positive features -> full timeline -> similarity ranked clips")
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Extract features from ALL human-reviewed cases")
    build.add_argument("--source", required=True, type=Path, help="Original MP4 on annotated timeline")
    build.add_argument("--truth", required=True, type=Path, help="原片2.json or truth-original2.json")
    build.add_argument("--output", required=True, type=Path, help="Positive profile JSON")
    build.add_argument("--step", type=float, default=2.0)
    build.add_argument("--fps", type=float, default=4.0)

    scan = sub.add_parser("scan", help="Score ALL source windows, including layout=none, and rank clips")
    scan.add_argument("--input", required=True, type=Path, help="Raw source MP4")
    scan.add_argument("--profile", required=True, type=Path, help="Human positive profile JSON")
    scan.add_argument("--output", required=True, type=Path, help="Ranking output directory")
    scan.add_argument("--step", type=float, default=None)
    scan.add_argument("--fps", type=float, default=None)
    scan.add_argument("--min-similarity", type=float, default=.62,
                      help="Exploratory retrieval cutoff (not P(keep)); default .62")
    scan.add_argument("--min-duration", type=float, default=2.0)
    scan.add_argument("--boundary-mode", choices=("scene", "off"), default="scene",
                      help="Optional visual split after similarity retrieval, NOT semantic case segmentation")
    scan.add_argument("--export-top", type=int, default=0,
                      help="Export this many highest-ranked clips; 0=rank only")
    scan.add_argument("--exclude-reviewed", action="store_true",
                      help="Exclude annotated positive times when rescanning the EXACT training source")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from video_incese import Settings
    settings = Settings(content_mode="temporal-cv", boundary_threshold=.48,
                        boundary_min_segment=2.0)
    try:
        if args.command == "build":
            profile = build_profile(args.source, args.truth, settings,
                                    step=args.step, fps=args.fps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print(json.dumps({"profile": str(args.output),
                              "reviewed_cases": len(profile["cases"]),
                              "prototypes": sum(len(c["examples"]) for c in profile["cases"]),
                              "geometry_fallback_windows": sum(
                                  x["layout_evidence"] != "geometry"
                                  for c in profile["cases"] for x in c["examples"]),
                              "decode_failures": len(profile["decode_failures"])},
                             ensure_ascii=False, indent=2))
        else:
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
            result = run_scan(args.input, profile, args.output, settings,
                              step=args.step, fps=args.fps,
                              min_similarity=args.min_similarity,
                              min_duration=args.min_duration,
                              boundary_mode=args.boundary_mode,
                              export_top=args.export_top,
                              exclude_reviewed=args.exclude_reviewed)
            print(json.dumps({"ranking": str(args.output / "ranking.csv"),
                              "windows_scored": result["scan"]["windows_scored"],
                              "none_layout_scored": result["scan"]["geometry_none_but_scored"],
                              "ranked_segments": len(result["ranked"]),
                              "top_candidates": result["ranked"][:5]},
                             ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, cv2.error,
            json.JSONDecodeError) as exc:
        print(f"similarity error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
