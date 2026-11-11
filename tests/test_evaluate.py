"""Annotation-aware evaluation invariants, including unknown time and one-to-one IoU."""
import json

import pytest

import video_evaluate as ev


def _report(*intervals):
    return {
        "input": "/tmp/source.mp4", "video": {"duration": 60.0},
        "clips": [{"status": "exported", "start": a, "end": b} for a, b in intervals],
        "boundaries": {"accepted_events": 2},
    }


def test_exact_predictions_have_full_recall_and_matching():
    data = ev.evaluate(
        _report((2, 6), (10, 14)),
        {"source": "source.mp4",
         "positives": [{"start": 2, "end": 6}, {"start": 10, "end": 14}],
         "negatives": [{"start": 0, "end": 2}, {"start": 6, "end": 10}]})
    assert data["positive_time_recall"] == 1
    assert data["one_to_one_iou_matches"] == 2
    assert data["matched_boundary_mae_seconds"] == 0
    assert data["precision_on_labeled_time_only"] == 1
    assert data["global_precision"] is None
    assert data["confirmed_negative_predicted_seconds"] == 0


def test_unknown_not_counted_as_confirmed_negative():
    data = ev.evaluate(
        _report((0, 3), (10, 14)),
        {"positives": [{"start": 10, "end": 12}]})
    assert data["positive_time_recall"] == 1
    assert data["confirmed_negative_false_accept_rate"] is None
    assert data["precision_on_labeled_time_only"] is None
    assert data["global_precision"] is None
    assert data["predicted_union_seconds"] == 7


def test_overlap_union_no_double_count_and_verified_negative_only():
    data = ev.evaluate(
        _report((0, 5), (3, 9)),
        {"positives": [{"start": 2, "end": 6}],
         "negatives": [{"start": 0, "end": 2}, {"start": 8, "end": 10}]})
    assert data["predicted_union_seconds"] == 9
    assert data["positive_covered_seconds"] == 4
    assert data["confirmed_negative_predicted_seconds"] == 3
    assert data["confirmed_negative_false_accept_rate"] == .75
    assert data["precision_on_labeled_time_only"] == round(4/7, 5)


def test_split_and_merged_cases_have_separate_errors():
    data = ev.evaluate(
        _report((2, 4), (4, 7), (10, 16)),
        {"positives": [{"start": 2, "end": 7}, {"start": 10, "end": 12},
                       {"start": 13, "end": 16}]})
    assert data["split_positive_cases"] == 1
    assert data["merged_positive_cases_in_prediction"] == 1
    assert data["one_to_one_iou_matches"] == 2


def test_max_cardinality_matching_not_cartesian():
    result = ev._matching(
        [(0, 10), (1, 9)],
        [(0, 9), (1, 10)], threshold=.5)
    assert len(result) == 2
    assert len({p["positive_index"] for p in result}) == 2
    assert len({p["prediction_index"] for p in result}) == 2


@pytest.mark.parametrize("truth", [
    {"positives": []},
    {"positives": [{"start": 2, "end": 1}]},
    {"positives": [{"start": 0, "end": float("nan")}]},
    {"positives": [{"start": 1, "end": 4}, {"start": 3, "end": 6}]},
    {"positives": [{"start": 1, "end": 4}], "negatives": [{"start": 3, "end": 5}]},
    {"source": "wrong.mp4", "positives": [{"start": 1, "end": 4}]},
    {"positives": [{"start": 59, "end": 61}]},
])
def test_bad_truth_rejected(truth):
    with pytest.raises(ValueError):
        ev.evaluate(_report((2, 3)), truth)


def test_cli_roundtrip_and_malformed_truth_exit(tmp_path, capsys):
    report = tmp_path / "report.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "metrics.json"
    report.write_text(json.dumps(_report((2, 5))))
    truth.write_text(json.dumps({"positives": [{"start": 2, "end": 6}]}))
    assert ev.main(["--report", str(report), "--truth", str(truth),
                    "--output", str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved["positive_time_recall"] == .75
    truth.write_text("{}")
    assert ev.main(["--report", str(report), "--truth", str(truth)]) == 2
    assert "evaluation error" in capsys.readouterr().err


def test_dry_run_and_failed_exports_not_confused():
    report = _report((1, 2))
    report["clips"] += [
        {"start": 2, "end": 3, "status": "dry_run"},
        {"start": 3, "end": 8, "status": "export_error"},
        {"start": 8, "end": 9, "status": "rejected_temporal"},
    ]
    result = ev.evaluate(report, {"positives": [{"start": 1, "end": 8}]})
    assert result["predicted_clips"] == 2
    assert result["positive_covered_seconds"] == 2
    assert result["positive_time_recall"] == round(2/7, 5)
