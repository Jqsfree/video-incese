"""Offline, annotation-aware evaluation of exported video intervals.

No unlabelled time is silently treated as a negative. Positive and confirmed
negative intervals are explicit; the rest of the source is UNKNOWN.
Standard library only, so evaluation works without OpenCV/FFmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SUCCESS = {"exported", "dry_run"}


def _intervals(raw: object, name: str, *, allow_empty: bool = True) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list of {{start, end}} objects")
    if not allow_empty and not raw:
        raise ValueError(f"{name} must contain at least one interval")
    spans = []
    for i, value in enumerate(raw):
        if not isinstance(value, dict) or not all(k in value for k in ("start", "end")):
            raise ValueError(f"{name}[{i}] needs start and end")
        a, b = value["start"], value["end"]
        if (isinstance(a, bool) or isinstance(b, bool) or
                not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or
                not math.isfinite(a) or not math.isfinite(b) or
                a < 0 or b <= a):
            raise ValueError(f"{name}[{i}] must contain finite 0 <= start < end")
        spans.append((float(a), float(b)))
    return sorted(spans)


def _merged(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _length(spans: list[tuple[float, float]]) -> float:
    return sum(b-a for a, b in spans)


def _overlap(one: tuple[float, float], other: tuple[float, float]) -> float:
    return max(0.0, min(one[1], other[1]) - max(one[0], other[0]))


def _covered(one: list[tuple[float, float]], other: list[tuple[float, float]]) -> float:
    """Intersect unioned timelines, avoiding double-counted overlapping predictions."""
    left, right = _merged(one), _merged(other)
    i = j = 0
    amount = 0.0
    while i < len(left) and j < len(right):
        amount += _overlap(left[i], right[j])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return amount


def _distinct(spans: list[tuple[float, float]], name: str) -> None:
    for prior, nxt in zip(spans, spans[1:]):
        if prior[1] > nxt[0]:
            raise ValueError(f"{name} contains overlapping intervals; truth must be nonoverlapping")


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = _overlap(a, b)
    return intersection / (a[1]-a[0]+b[1]-b[0]-intersection)


def _matching(predicted: list[tuple[float, float]], positives: list[tuple[float, float]],
              threshold: float) -> list[dict]:
    """Maximum-cardinality one-to-one matching, not count of all intersecting pairs."""
    neighbors = [
        sorted((j for j, truth in enumerate(positives) if _iou(clip, truth) >= threshold),
               key=lambda j: (-_iou(clip, positives[j]), j))
        for clip in predicted
    ]
    claimed: dict[int, int] = {}

    def visit(i: int, seen: set[int]) -> bool:
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in claimed or visit(claimed[j], seen):
                claimed[j] = i
                return True
        return False

    for i in sorted(range(len(predicted)), key=lambda i: (-len(neighbors[i]), i)):
        visit(i, set())
    return [
        {"prediction_index": i, "positive_index": j,
         "iou": round(_iou(predicted[i], positives[j]), 5),
         "start_error_seconds": round(abs(predicted[i][0]-positives[j][0]), 5),
         "end_error_seconds": round(abs(predicted[i][1]-positives[j][1]), 5)}
        for j, i in sorted(claimed.items())
    ]


def evaluate(report: dict, truth: dict, *, iou_threshold: float = .5) -> dict:
    if not (0 < iou_threshold <= 1):
        raise ValueError("iou_threshold must be in (0, 1]")
    positives = _intervals(truth.get("positives"), "positives", allow_empty=False)
    negatives = _intervals(truth.get("negatives", []), "negatives")
    _distinct(positives, "positives")
    _distinct(negatives, "negatives")
    if _covered(positives, negatives) > 1e-8:
        raise ValueError("positives and verified negatives overlap")
    if not isinstance(report.get("clips"), list):
        raise ValueError("Report lacks clips list")
    source = report.get("input")
    expected_source = truth.get("source")
    if expected_source:
        if not source or Path(str(expected_source)).name != Path(str(source)).name:
            raise ValueError("Truth and report source video filenames do not match")
    duration = report.get("video", {}).get("duration")
    if duration is not None:
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("Invalid report.video.duration")
        if max(e for _, e in positives + negatives) > float(duration) + .05:
            raise ValueError("Truth intervals extend past video duration")
    predicted = _intervals([c for c in report["clips"]
                            if c.get("status") in SUCCESS], "exported clips")
    union = _merged(predicted)
    accepted = _length(union)
    gt_seconds = _length(positives)
    positive_hit = _covered(union, positives)
    negative_hit = _covered(union, negatives)
    matches = _matching(predicted, positives, iou_threshold)
    positive_details = []
    for i, p in enumerate(positives):
        hits = sum(_overlap(p, x) >= .25 for x in predicted)
        coverage = _covered([p], union) / (p[1]-p[0])
        positive_details.append({
            "index": i, "start": p[0], "end": p[1], "coverage_fraction": round(coverage, 5),
            "overlapping_clips": hits,
        })
    merges = sum(sum(_overlap(x, p) >= .25 for p in positives) > 1
                 for x in predicted)
    mae = None if not matches else round(
        sum(x["start_error_seconds"]+x["end_error_seconds"] for x in matches)
        / (2 * len(matches)), 5)
    known_precision = None if not negatives else (
        None if positive_hit + negative_hit <= 0 else
        round(positive_hit / (positive_hit + negative_hit), 5))
    return {
        "source": source,
        "annotation_policy": "Only positives and verified negatives are labeled; all other time is UNKNOWN.",
        "prediction_statuses": sorted(SUCCESS),
        "predicted_clips": len(predicted),
        "predicted_union_seconds": round(accepted, 5),
        "annotated_positive_clips": len(positives),
        "annotated_positive_seconds": round(gt_seconds, 5),
        "positive_covered_seconds": round(positive_hit, 5),
        "positive_time_recall": round(positive_hit / gt_seconds, 5),
        "positive_clips_95pct_covered": sum(x["coverage_fraction"] >= .95 for x in positive_details),
        "confirmed_negative_seconds": round(_length(negatives), 5),
        "confirmed_negative_predicted_seconds": round(negative_hit, 5),
        "confirmed_negative_false_accept_rate": (
            round(negative_hit / _length(negatives), 5) if negatives else None),
        "precision_on_labeled_time_only": known_precision,
        "global_precision": None,
        "iou_threshold": iou_threshold,
        "one_to_one_iou_matches": len(matches),
        "matched_boundary_mae_seconds": mae,
        "split_positive_cases": sum(x["overlapping_clips"] > 1 for x in positive_details),
        "merged_positive_cases_in_prediction": merges,
        "matched_pairs": matches,
        "positive_cases": positive_details,
        "boundary_events_accepted": report.get("boundaries", {}).get("accepted_events"),
        "notes": [
            "Global precision is undefined because unannotated intervals are unknown, not negative.",
            "Matched boundary MAE is conditional on IoU-matched pairs, not a global boundary score.",
            "Do not treat validation on a training/development source as independent generalization.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a video-incese report against explicit reviewed positives/negatives")
    p.add_argument("--report", required=True, type=Path, help="report.json")
    p.add_argument("--truth", required=True, type=Path, help="reviewed truth JSON with positives, negatives")
    p.add_argument("--output", type=Path, help="Optional result JSON; otherwise stdout only")
    p.add_argument("--iou-threshold", type=float, default=.5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        result = evaluate(report, truth, iou_threshold=args.iou_threshold)
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
