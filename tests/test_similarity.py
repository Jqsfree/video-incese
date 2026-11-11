"""v0.6 positive-only retrieval: annotation provenance, full-timeline scanning, ranking."""
import json
from pathlib import Path

import numpy as np
import pytest

import video_similarity as sim
import video_incese as vi
from video_temporal_cv import FEATURE_NAMES
from conftest import needs_ffmpeg


def _features(**extra):
    base = {name: .2 for name in sim.FEATURES}
    base.update(extra)
    return base


def _profile():
    return {
        "profile_version": sim.PROFILE_VERSION,
        "feature_version": sim.FEATURE_VERSION,
        "training_source": "original.mp4",
        "training_duration": 6.0,
        "training_file_sha256": "test-hash",
        "window_seconds": 2, "sample_fps": 4,
        "weights": sim.WEIGHTS,
        "scales": {n: .20 for n in sim.FEATURES},
        "features": list(sim.FEATURES),
        "cases": [
            {"case_id": 1, "start": 1, "end": 3,
             "examples": [{"start": 1, "end": 3, "layout": "pip",
                           "layout_evidence": "geometry",
                           "features": _features()}]},
            {"case_id": 2, "start": 3, "end": 5,
             "examples": [{"start": 3, "end": 5, "layout": "split",
                           "layout_evidence": "forced_positive_hypotheses",
                           "features": _features(reference_motion=.7)}]},
        ],
    }


def test_verified_manual_clip_file_as_ground_truth(tmp_path):
    path = tmp_path / "original2.json"
    payload = {"source": {"file": "original.mp4"}, "timeline": {"start_ms": 0},
               "clips": [
                   {"start_ms": 1000, "end_ms": 3100},
                   {"start_ms": 4000, "end_ms": 5500},
               ],
               "gaps": [{"from_ms": 0, "to_ms": 1000}]}
    path.write_text(json.dumps(payload))
    cases = sim.load_positives(path, expected_filename="original.mp4", duration=6)
    assert [(x["start"], x["end"]) for x in cases] == [(1, 3.1), (4, 5.5)]
    assert len(cases) == 2  # gaps never become "reject"
    with pytest.raises(ValueError, match="not wrong"):
        sim.load_positives(path, expected_filename="wrong.mp4", duration=6)


def test_positive_truth_contract_does_not_infer_negatives(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"source": "source.mp4",
                                 "positives": [{"start": 1.2, "end": 2.7}],
                                 "negatives": []}))
    result = sim.load_positives(truth, expected_filename="source.mp4")
    assert len(result) == 1
    truth.write_text(json.dumps({"positives": [{"start": 1, "end": 3},
                                              {"start": 2, "end": 4}]}))
    with pytest.raises(ValueError, match="overlap"):
        sim.load_positives(truth)


def test_case_balanced_scales_and_positive_similarity_ranking():
    profile = _profile()
    sim.validate_profile(profile)
    exact, matching = sim._score_features(_features(), profile, "pip")
    distant, _ = sim._score_features(_features(reference_flow=1.0, rendered_flow=1.0,
                                               reference_neutral_ratio=1.0), profile, "pip")
    assert exact > distant
    assert matching[0]["case_id"] == 1
    assert exact < 1  # per-case aggregation also considers other cases; not a probability
    assert len(matching) == 2


def test_candidate_ranges_rank_by_similarity_and_only_high_windows():
    def w(start, end, score, case):
        return {"start": start, "end": end, "layout": "split",
                "similarity": score,
                "nearest_cases": [{"case_id": case, "similarity": score}]}
    windows = [w(0, 2, .90, 1), w(2, 4, .86, 1), w(4, 6, .15, 2),
               w(6, 8, .71, 2), w(8, 10, .70, 2)]
    candidates = sim.rank_segments(windows, vi.Settings(), [], min_similarity=.60,
                                   min_duration=2)
    assert len(candidates) == 2
    assert [(p["start"], p["end"]) for p in candidates] == [(0, 4), (6, 10)]
    assert [p["rank"] for p in candidates] == [1, 2]
    assert all(p["label"] == "unknown_needs_human_review" for p in candidates)


def test_scene_cut_partitions_retrieval_but_preserves_temporal_coverage():
    windows = [{"start": i*2, "end": (i+1)*2, "layout": "pip",
                "similarity": .9,
                "nearest_cases": [{"case_id": 1, "similarity": .9}]}
               for i in range(5)]
    event = {"status": "accepted", "source_candidate": 1,
             "span_start": 0, "span_end": 10, "time": 4.5, "event_id": 1}
    ranked = sim.rank_segments(windows, vi.Settings(), [event],
                               min_similarity=.6, min_duration=2)
    assert [(p["start"], p["end"]) for p in ranked] == [(0, 4.5), (4.5, 10)]
    assert sum(p["duration"] for p in ranked) == 10
    assert ranked[0]["boundary_after"] == 1


def test_forced_layout_hypotheses_for_geometry_none(monkeypatch):
    monkeypatch.setattr(sim, "_geometry", lambda frames, settings: "none")
    candidates, source = sim._positive_layouts(
        [np.zeros((216, 384, 3), dtype=np.uint8)] * 5, vi.Settings())
    assert candidates == ["pip", "split"]
    assert source == "forced_positive_hypotheses"


def test_bad_profile_rejected_not_silently_accepted():
    profile = _profile()
    profile["features"] = ["wrong"]
    with pytest.raises(ValueError, match="Feature names"):
        sim.validate_profile(profile)


@needs_ffmpeg
def test_end_to_end_manual_positive_extraction_and_full_timeline_scanning(
        make_video, tmp_path):
    # Synthetic 10s video, manually reviewed time from 2-4s and 7-9s.
    # Other time is UNKNOWN even though an ordinary layout detector may reject it.
    path = make_video([(2, 4), (7, 9)], seconds=10, name="synthetic.mp4")
    truth = tmp_path / "original2.json"
    truth.write_text(json.dumps({
        "source": {"file": "synthetic.mp4"},
        "timeline": {"start_ms": 0, "end_ms": 10000},
        "clips": [
            {"start_ms": 2000, "end_ms": 4000},
            {"start_ms": 7000, "end_ms": 9000}],
        "gaps": [{"from_ms": 0, "to_ms": 2000}]}))
    settings = vi.Settings(content_mode="temporal-cv")
    profile = sim.build_profile(path, truth, settings, step=2, fps=3)
    assert len(profile["cases"]) == 2
    assert all(c["examples"] for c in profile["cases"])
    assert len(profile["training_file_sha256"]) == 64
    windows, stats = sim.scan_windows(path, profile, settings, step=2, fps=3)
    assert stats["windows_scored"] == 5
    assert len(windows) == 5  # entire 10s, not just the training positive spans
    assert "geometry_none_but_scored" in stats
    assert all("nearest_cases" in w for w in windows)
    assert all(w["similarity"] > 0 for w in windows)

    # Exclusion is explicit and bounded to verified same-source bytes.
    unseen, excluded = sim.scan_windows(path, profile, settings, step=2, fps=3,
                                        exclude_reviewed=True)
    assert excluded["same_source_as_training"]
    assert len(unseen) == 2
    path_copy = tmp_path / "copy.mp4"
    path_copy.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="original training source"):
        sim.scan_windows(path_copy, profile, settings, exclude_reviewed=True)


@needs_ffmpeg
def test_rank_outputs_are_auditable_not_automatically_verified(make_video, tmp_path):
    path = make_video([(1, 4)], seconds=6, name="synthetic.mp4")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"source": "synthetic.mp4",
                                 "positives": [{"start": 1, "end": 4}]}))
    settings = vi.Settings(content_mode="temporal-cv")
    profile = sim.build_profile(path, truth, settings, step=1.0, fps=3)
    out = tmp_path / "ranking"
    results = sim.run_scan(path, profile, out, settings,
                           step=1.0, fps=3, min_similarity=0,
                           min_duration=.75, boundary_mode="off", export_top=0)
    assert results["scan"]["windows_scored"] == 6
    assert results["ranked"]
    assert results["ranked"][0]["start"] == 0
    assert results["ranked"][0]["label"] == "unknown_needs_human_review"
    assert results["scan"]["score_is_probability"] is False
    assert (out / "ranking.json").is_file()
    assert (out / "ranking.csv").is_file()
    assert not (out / "ranked-clips").exists()
