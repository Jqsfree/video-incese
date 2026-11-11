"""Offline controlled retrieval experiments on one existing ranking.json.

Compares alternative positive-example strategies at an identical window budget.
No video decode, no fitting against the evaluation truth, no negative labels
inferred from unlabeled gaps, no unverified 'precision' reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from video_incese import Settings
from video_similarity import (
    SIMILARITY_STRATEGIES, _score_features, evaluate_ranking,
    load_positives, rank_segments, validate_profile,
)

PIPELINES = tuple((strategy, smoothing) for strategy in SIMILARITY_STRATEGIES
                  for smoothing in ("off", "median3"))


def _rescore_windows(windows: list[dict], profile: dict, strategy: str
                    ) -> tuple[list[dict], int]:
    """Replay the same feature vectors under another strategy.

    Old v0.6 scans kept only the chosen region hypothesis; for them
    re-ranking uses that same chosen layout rather than guessing a missing ROI.
    """
    rows = []
    incomplete = 0
    for original in windows:
        hypotheses = original.get("hypotheses")
        if not hypotheses:
            incomplete += 1
            hypotheses = [{"layout": original["layout"],
                           "features": original["features"]}]
        candidates = []
        for h in hypotheses:
            score, nearest = _score_features(
                h["features"], profile, h["layout"], strategy=strategy)
            agree = h["layout"] == original.get("geometry_layout")
            candidates.append((score, agree, h["layout"], nearest, h["features"]))
        if not candidates:
            raise ValueError("A scanned window has no feature hypotheses")
        score, _, layout, nearest, features = max(
            candidates, key=lambda c: (c[0], c[1]))
        rows.append({
            **original, "similarity": score, "nearest_cases": nearest,
            "layout": layout, "features": features,
            "scoring_strategy": strategy,
        })
    return rows, incomplete


def _median_smooth(windows: list[dict]) -> list[dict]:
    """Median-3 only on three consecutive windows; no bridging UNKNOWN gaps."""
    ordered = sorted(windows, key=lambda x: x["start"])
    out = []
    for i, item in enumerate(ordered):
        row = dict(item)
        if (0 < i < len(ordered)-1
                and abs(ordered[i-1]["end"] - item["start"]) < .002
                and abs(item["end"] - ordered[i+1]["start"]) < .002):
            scores = sorted([ordered[i-1]["similarity"], item["similarity"],
                             ordered[i+1]["similarity"]])
            row["similarity"] = round(scores[1], 5)
        out.append(row)
    return out


def _selected_windows(windows: list[dict], count: int) -> list[dict]:
    if not (1 <= count <= len(windows)):
        raise ValueError("Window budget must be 1..len(windows)")
    ranked = sorted(windows, key=lambda x: (-x["similarity"], x["start"], x["end"]))
    return sorted(ranked[:count], key=lambda x: (x["start"], x["end"]))


def audit_labels(truth: Path, labels: Path) -> dict:
    """Surface apparent overlapping human judgments, without choosing sides.

    A review.csv row contains report_path, not the original MP4 hash, so exact
    source identity may require human confirmation before calling it a conflict.
    """
    positives = load_positives(truth)
    with labels.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    issues = []
    counts = {"keep": 0, "reject": 0, "unsure": 0}
    for i, row in enumerate(rows, 1):
        label = str(row.get("label", "")).strip().lower()
        if label not in counts:
            continue
        counts[label] += 1
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"Review row {i} has invalid time columns")
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end):
            raise ValueError(f"Review row {i} has invalid time span")
        overlaps = []
        for case in positives:
            seconds = max(0.0, min(case["end"], end)-max(case["start"], start))
            if seconds > .001:
                overlaps.append({"case_id": case["case_id"],
                                 "overlap_seconds": round(seconds, 3)})
        if label == "reject" and overlaps:
            issues.append({"row": i, "start": start, "end": end,
                           "label": label, "overlapping_manual_cases": overlaps,
                           "issue": "potential_positive_reject_overlap"})
        # 'keep' outside the 28 chosen clips is NOT a contradiction:
        # positive-only annotations are not exhaustive.
    return {
        "reviewed_rows": len(rows), "labels": counts,
        "potential_conflicting_reject_rows": len(issues),
        "possible_time_axis_conflicts": issues,
        "source_identity_verified": False,
        "interpretation": (
            "Overlap is a possible annotation inconsistency ONLY if all input "
            "times refer to the same original video and same timeline. "
            "No labels are overwritten or promoted into training." ),
    }


def benchmark(profile: dict, ranking: dict, truth: Path, *,
              strategies: tuple[str, ...] = SIMILARITY_STRATEGIES,
              smoothing: tuple[str, ...] = ("off", "median3"),
              budget_windows: int | None = None,
              budgets: tuple[int, ...] | None = None,
              top_k: tuple[int, ...] = (10, 20, 50),
              min_duration: float = 2.0) -> dict:
    from video_evaluate import evaluate

    validate_profile(profile)
    if not isinstance(ranking.get("windows"), list) or not ranking["windows"]:
        raise ValueError("ranking.json must contain non-empty windows")
    if not isinstance(ranking.get("input"), str) or not ranking.get("scan", {}).get("duration"):
        raise ValueError("ranking.json is missing input or scan duration")
    if not (.75 <= min_duration <= 30):
        raise ValueError("min_duration must be .75..30s")
    if not top_k or any(k < 1 for k in top_k):
        raise ValueError("Provide positive --top-k budgets")
    if any(s not in SIMILARITY_STRATEGIES for s in strategies):
        raise ValueError("Unknown strategy selected")
    if any(mode not in {"off", "median3"} for mode in smoothing):
        raise ValueError("Unknown smoothing mode")
    if budget_windows is not None and budgets is not None:
        raise ValueError("Choose only one of --budgets and --budget-windows")

    original = ranking["windows"]
    scanned = len(original)
    baseline_threshold = float(ranking.get("settings", {}).get("min_similarity", .62))
    original_selected = sum(float(w["similarity"]) >= baseline_threshold for w in original)

    if budget_windows is not None:
        resolved_budgets = (budget_windows,)
    elif budgets is not None:
        resolved_budgets = tuple(dict.fromkeys(budgets))
    else:
        # A score threshold that admits 477/478 windows is NOT a useful
        # ranking experiment. Use sparse retrieval budgets instead.
        resolved_budgets = tuple(sorted({
            max(1, round(scanned * ratio)) for ratio in (.10, .20, .30, .40)
        }))
    if not resolved_budgets or any(not isinstance(b, int) or b < 1 or b > scanned
                                   for b in resolved_budgets):
        raise ValueError(f"Each window budget must be in 1..{scanned}")
    resolved_budgets = tuple(sorted(resolved_budgets))

    cases = load_positives(truth, expected_filename=Path(ranking["input"]).name,
                           duration=float(ranking["scan"]["duration"]))
    annotation = {"source": Path(ranking["input"]).name,
                  "positives": [{"start": x["start"], "end": x["end"]} for x in cases]}
    trials = []
    for strategy in strategies:
        scored, single_layout_count = _rescore_windows(original, profile, strategy)
        for smooth_mode in smoothing:
            current = _median_smooth(scored) if smooth_mode == "median3" else scored
            for budget in resolved_budgets:
                selected = _selected_windows(current, budget)
                # Window-level coverage measures the retrieval stage, before
                # segmentation and min-duration filtering change the evidence.
                selected_report = {
                    "input": ranking["input"],
                    "video": {"duration": float(ranking["scan"]["duration"])},
                    "clips": [{"start": w["start"], "end": w["end"],
                               "status": "dry_run"} for w in selected],
                }
                window_metrics = evaluate(selected_report, annotation)
                # No scene cuts: events of another segmentation cannot
                # be reused with different selected time spans.
                ranked = rank_segments(selected, Settings(), [],
                                       min_similarity=0., min_duration=min_duration)
                test = evaluate_ranking(
                    {"input": ranking["input"], "scan": ranking["scan"],
                     "ranked": ranked}, truth, top_k=top_k)
                trials.append({
                    "strategy": strategy, "smoothing": smooth_mode,
                    "selected_windows": len(selected),
                    "single_layout_hypothesis_windows": single_layout_count,
                    "selected_window_seconds": window_metrics["predicted_union_seconds"],
                    "selected_window_positive_time_recall": window_metrics["positive_time_recall"],
                    "candidate_segments": len(ranked),
                    "retrieved_covered_seconds": round(sum(p["duration"] for p in ranked), 3),
                    "metrics": test["results"],
                    "top_candidates": [
                        {key: row[key] for key in ("rank", "start", "end", "duration",
                                                  "similarity", "top_reference_cases")}
                        for row in ranked[:min(10, len(ranked))]
                    ],
                })
    warnings = []
    if original_selected >= .90 * scanned:
        warnings.append(
            f"Original similarity cutoff selects {original_selected}/{scanned} windows "
            f"({original_selected/scanned:.1%}); saturated threshold NOT used as default budget."
        )
    if max(resolved_budgets) >= .90 * scanned:
        warnings.append(
            "At least one explicit budget covers >=90% of scanned windows; "
            "full-time recall may be trivial while case IoU collapses."
        )
    return {
        "experiment_version": "retrieval-pipelines-v2",
        "source": ranking["input"], "training_source": profile.get("training_source"),
        "same_source_as_training": ranking.get("scan", {}).get("same_source_as_training"),
        "manually_selected_positive_cases": len(cases),
        "budget_window_count": (resolved_budgets[0] if len(resolved_budgets) == 1 else None),
        "budget_window_counts": list(resolved_budgets),
        "baseline_threshold": baseline_threshold,
        "baseline_selected_windows": original_selected,
        "scanned_window_count": scanned,
        "top_k": list(top_k), "min_duration_seconds": min_duration,
        "scene_boundary_mode": "off_for_controlled_retrieval_comparison",
        "trials": trials,
        "warnings": warnings,
        "limitations": [
            "Equal budgets compare selected WINDOW COUNT, not equal MP4 count or exactly equal seconds.",
            "Window-level positive recall is measured BEFORE min-duration filtering and segmentation.",
            "Top-K clip metrics measure segments AFTER grouping; longer segments may inflate time recall.",
            "v0.6 ranking JSON stores only selected ROI layout; old scans compare FIXED-LAYOUT hypotheses.",
            "Training-source retrieval metrics do NOT measure independent new-video performance.",
            "Unlabeled time is UNKNOWN, not negative: precision cannot be calculated.",
            "Scene boundaries omitted to isolate retrieval; case segmentation needs separate evaluation.",
        ],
    }


def _write_summary(path: Path, result: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        cols = ["strategy", "smoothing", "window_budget", "segments", "top_k",
                "retrieved_clips", "selected_window_seconds",
                "selected_window_positive_recall", "positive_time_recall",
                "one_to_one_iou_matches", "split_positive_cases", "merged_positive_cases"]
        out = csv.DictWriter(stream, fieldnames=cols)
        out.writeheader()
        for trial in result["trials"]:
            for m in trial["metrics"]:
                out.writerow({
                    "strategy": trial["strategy"], "smoothing": trial["smoothing"],
                    "window_budget": trial["selected_windows"],
                    "segments": trial["candidate_segments"],
                    "top_k": m["top_k"], "retrieved_clips": m["retrieved_clips"],
                    "selected_window_seconds": trial["selected_window_seconds"],
                    "selected_window_positive_recall": trial["selected_window_positive_time_recall"],
                    "positive_time_recall": m["annotated_positive_time_recall"],
                    "one_to_one_iou_matches": m["one_to_one_iou_matches"],
                    "split_positive_cases": m["split_positive_cases"],
                    "merged_positive_cases": m["merged_positive_cases_in_prediction"],
                })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare positive-example similarity pipelines without re-decoding MP4")
    p.add_argument("--ranking", required=True, type=Path, help="ranking.json from rank scan")
    p.add_argument("--profile", required=True, type=Path, help="positive-profile.json")
    p.add_argument("--truth", required=True, type=Path, help="Human positive timeline JSON")
    p.add_argument("--output", required=True, type=Path, help="Folder for experiment outputs")
    p.add_argument("--strategies", nargs="+", choices=SIMILARITY_STRATEGIES,
                   default=list(SIMILARITY_STRATEGIES))
    p.add_argument("--smoothing", nargs="+", choices=("off", "median3"),
                   default=["off", "median3"])
    p.add_argument("--budget-windows", type=int, default=None,
                   help="Legacy single budget, mutually exclusive with --budgets")
    p.add_argument("--budgets", nargs="+", type=int, default=None,
                   help="Window-count sweep, e.g. 50 100 150 200; default 10/20/30/40%")
    p.add_argument("--top-k", nargs="+", type=int, default=[10, 20, 50])
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--review-labels", type=Path,
                   help="Optional prior human review CSV for possible label conflict audit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        result = benchmark(profile, ranking, args.truth,
                           strategies=tuple(args.strategies),
                           smoothing=tuple(args.smoothing),
                           budget_windows=args.budget_windows,
                           budgets=tuple(args.budgets) if args.budgets else None,
                           top_k=tuple(args.top_k),
                           min_duration=args.min_duration)
        if args.review_labels:
            result["label_audit"] = audit_labels(args.truth, args.review_labels)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"experiments.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_summary(args.output/"experiments.csv", result)
        print(json.dumps({
            "experiment": str(args.output/"experiments.json"),
            "summary": str(args.output/"experiments.csv"),
            "source": result["source"], "window_budgets": result["budget_window_counts"],
            "original_saturated_cutoff_windows": result["baseline_selected_windows"],
            "scanned_windows": result["scanned_window_count"],
            "warnings": result["warnings"], "trials": len(result["trials"]),
            "label_conflicts": result.get("label_audit", {}).get(
                "potential_conflicting_reject_rows"),
        }, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, json.JSONDecodeError) as e:
        print(f"experiment error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
