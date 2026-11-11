"""Scene-boundary proposal sweeps against only known adjacent human cases."""
import json

import pytest

import video_boundary_experiments as b


def _cases():
    return [
        {"case_id": 1, "start": 0., "end": 4.},
        {"case_id": 2, "start": 4., "end": 8.},
        {"case_id": 3, "start": 9., "end": 12.},
        {"case_id": 4, "start": 12., "end": 16.},
    ]


def _event(time, score, status="accepted"):
    return {
        "time": time, "score": score, "status": status,
        "reason": "simultaneous_abrupt_scene_change",
        "source_candidate": 1, "span_start": 0., "span_end": 16.,
    }


def _report():
    return {
        "input": "/tmp/source.mp4",
        "video": {"duration": 16.},
        "clips": [{"status": "exported", "start": 0., "end": 16.}],
        "boundary_events": [
            _event(4.02, .52),
            _event(5., .63),  # inside a known complete case
            _event(12.01, .61),
            _event(5.2, .44, "suppressed_nearby"),
        ],
    }


def test_only_adjacent_manual_cases_supply_known_boundaries():
    out = b.contiguous_boundaries(_cases())
    assert [(x["before_case"], x["after_case"]) for x in out] == [
        (1, 2), (3, 4)]
    assert [x["time"] for x in out] == [4, 12]


def test_observable_replay_threshold_sweep_and_inside_case():
    result = b.analyze(_report(), _cases(), thresholds=(.5, .55), tolerance=.1)
    assert result["total_contiguous_manual_boundaries"] == 2
    assert result["observable_manual_boundaries"] == 2
    low, high = result["threshold_sweep"]
    assert low["matched_true_boundaries"] == 2
    assert low["interior_split_proposals"] == 1
    assert high["matched_true_boundaries"] == 1
    assert high["interior_split_proposals"] == 1
    assert len(high["matches"]) == 1
    assert high["matches"][0]["human_time"] == 12.


def test_unobservable_manual_boundary_not_counted_as_detector_miss():
    report = _report()
    report["clips"] = [{"status": "exported", "start": 9., "end": 16.}]
    result = b.analyze(report, _cases(), thresholds=(.42,))
    assert result["observable_manual_boundaries"] == 1
    assert result["unobservable_manual_boundaries"][0]["time"] == 4.


def test_score_order_suppression_keeps_stronger_neighbor():
    report = _report()
    selected = b._proposals(report, .42, .85)
    assert any(e["time"] == 5 for e in selected)
    assert not any(e["time"] == 5.2 for e in selected)


@pytest.mark.parametrize("kwargs", [
    {"thresholds": (-.1,)},
    {"thresholds": (1.1,)},
    {"thresholds": (.5,), "tolerance": 0},
    {"thresholds": (.5,), "min_separation": 0},
])
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        b.analyze(_report(), _cases(), **kwargs)


def test_cli_reads_manual_ms_file_not_unreviewed_gaps(tmp_path):
    report = tmp_path / "report.json"
    truth = tmp_path / "原片2.json"
    output = tmp_path / "boundary-results.json"
    report.write_text(json.dumps(_report()))
    truth.write_text(json.dumps({
        "source": {"file": "source.mp4"},
        "timeline": {"start_ms": 0, "end_ms": 16000},
        "clips": [
            {"start_ms": int(x["start"]*1000), "end_ms": int(x["end"]*1000)}
            for x in _cases()
        ],
        "gaps": [{"from_ms": 8000, "to_ms": 9000}],
    }))
    assert b.main(["--report", str(report), "--truth", str(truth),
                   "--output", str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved["total_contiguous_manual_boundaries"] == 2
    assert len(saved["threshold_sweep"]) == 5
    assert not any(x.get("known_negative") for x in saved["threshold_sweep"])
