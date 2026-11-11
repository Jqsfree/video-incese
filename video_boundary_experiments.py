"""Read-only boundary score experiments on recorded proposals and manual cases.

Only adjacent manual cases provide known internal cut labels. All gaps are
UNKNOWN. Replaying scores cannot recover cuts never proposed by the detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_similarity import load_positives


def contiguous_boundaries(positives: list[dict], gap: float = .075) -> list[dict]:
    cases = sorted(positives, key=lambda c: c["start"])
    return [
        {"time": round((left["end"] + right["start"]) / 2, 4),
         "before_case": left["case_id"], "after_case": right["case_id"]}
        for left, right in zip(cases, cases[1:])
        if abs(right["start"] - left["end"]) <= gap
    ]


def _observable(report: dict, t: float, margin: float = .1) -> bool:
    spans = [(float(c["start"]), float(c["end"])) for c in report.get("clips", [])
             if c.get("status") in {"exported", "dry_run"}]
    return (any(a <= t - margin < b for a, b in spans) and
            any(a < t + margin <= b for a, b in spans))


def _proposals(report: dict, threshold: float, min_sep: float) -> list[dict]:
    raw = [e for e in report.get("boundary_events", [])
           if e.get("reason") == "simultaneous_abrupt_scene_change"
           and float(e["score"]) >= threshold]
    chosen = []
    for e in sorted(raw, key=lambda x: (-float(x["score"]), float(x["time"]))):
        if any(
            (e["source_candidate"], e["span_start"], e["span_end"]) ==
            (previous["source_candidate"], previous["span_start"], previous["span_end"])
            and abs(float(e["time"])-float(previous["time"])) < min_sep
            for previous in chosen
        ):
            continue
        chosen.append(e)
    return sorted(chosen, key=lambda e: e["time"])


def analyze(report: dict, positives: list[dict], *,
            thresholds: tuple[float, ...] = (.42, .50, .52, .55, .60),
            tolerance: float = .5, interior_margin: float = .5,
            min_separation: float = .85) -> dict:
    if not 0 < tolerance <= 3 or not 0 < interior_margin <= 3:
        raise ValueError("Expected tolerance/interior margin in (0,3]")
    if not .1 <= min_separation <= 10:
        raise ValueError("Invalid minimum separation")
    if any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Thresholds must be in [0,1]")
    cases = sorted(positives, key=lambda c: c["start"])
    boundaries = contiguous_boundaries(cases)
    eligible = [{**b, "observable": _observable(report, b["time"])}
                for b in boundaries]
    known = [b for b in eligible if b["observable"]]
    comparisons = []
    for threshold in thresholds:
        events = _proposals(report, threshold, min_separation)
        unmatched = set(range(len(events)))
        matches = []
        for boundary in known:
            close = sorted(
                ((abs(events[i]["time"]-boundary["time"]), i)
                 for i in unmatched
                 if abs(events[i]["time"]-boundary["time"]) <= tolerance),
                key=lambda x: (x[0], x[1]))
            if close:
                err, i = close[0]
                unmatched.remove(i)
                matches.append({
                    "before_case": boundary["before_case"],
                    "human_time": boundary["time"],
                    "event_time": events[i]["time"],
                    "error_seconds": round(err, 4),
                    "event_score": events[i]["score"],
                })
        interior = []
        for event in events:
            match = next(
                (case for case in cases
                 if case["start"]+interior_margin < event["time"] <
                    case["end"]-interior_margin),
                None)
            if match is not None:
                interior.append({
                    "time": event["time"], "score": event["score"],
                    "inside_manual_case": match["case_id"],
                })
        comparisons.append({
            "threshold": threshold,
            "retained_proposals": len(events),
            "observable_true_boundaries": len(known),
            "matched_true_boundaries": len(matches),
            "missed_observable_boundaries": len(known)-len(matches),
            "interior_split_proposals": len(interior),
            "other_unverified_proposals": len(events)-len(matches)-len(interior),
            "matches": matches,
            "inside_manual_cases": interior,
        })
    return {
        "experiment_version": "case-boundary-proposals-v1",
        "source": report.get("input"),
        "total_contiguous_manual_boundaries": len(eligible),
        "observable_manual_boundaries": len(known),
        "unobservable_manual_boundaries": [b for b in eligible if not b["observable"]],
        "threshold_sweep": comparisons,
        "matching_tolerance_seconds": tolerance,
        "limitations": [
            "Only contiguous manually selected cases are positive boundary labels.",
            "All unannotated time is UNKNOWN; non-adjacent case edges are NOT known cuts.",
            "An observable boundary has accepted video on both adjacent sides.",
            "Scores are replayed from existing proposals, not from new video decoding.",
            "Higher thresholds cannot recover cuts absent from the original report.",
            "Interior cuts are possible conflicts with manual case completeness, "
            "not independently verified semantic errors.",
            "Results on the training source are development data, not independent validation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare proposed scene cuts to adjacent human case boundaries")
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[.42, .50, .52, .55, .60])
    p.add_argument("--tolerance", type=float, default=.5)
    args = p.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        positives = load_positives(
            args.truth, expected_filename=Path(report["input"]).name,
            duration=float(report["video"]["duration"]))
        result = analyze(report, positives, thresholds=tuple(args.thresholds),
                         tolerance=args.tolerance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "total_manual_boundaries": result["total_contiguous_manual_boundaries"],
            "observable_boundaries": result["observable_manual_boundaries"],
            "results": [{
                k: row[k] for k in ("threshold", "retained_proposals",
                                   "matched_true_boundaries", "interior_split_proposals")
            } for row in result["threshold_sweep"]],
        }, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        print(f"boundary experiment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
