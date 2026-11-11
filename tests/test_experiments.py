"""Experimental ranking pipelines: no labeled negatives inferred from gaps."""
import csv
import json

import pytest

import video_experiments as e
import video_similarity as s


def _f(**kwargs):
    d = {n: .2 for n in s.FEATURES}
    d.update(kwargs)
    return d


def _profile():
    return {
        "profile_version": s.PROFILE_VERSION,
        "features": list(s.FEATURES),
        "scales": {n: .2 for n in s.FEATURES},
        "weights": s.WEIGHTS,
        "cases": [
            {"case_id": 1, "start": 1, "end": 3, "examples": [
                {"layout": "pip", "features": _f(reference_motion=.5)},
                {"layout": "pip", "features": _f(reference_motion=.6)},
                {"layout": "pip", "features": _f(reference_motion=.7)},
            ]},
            {"case_id": 2, "start": 6, "end": 8, "examples": [
                {"layout": "split", "features": _f(reference_flow=.8)}
            ]},
        ],
        "training_source": "source.mp4",
        "training_duration": 10,
    }


def _ranking(with_hypotheses=True):
    rows = []
    for i in range(5):
        f = _f(reference_motion=.5 if i in (1, 2) else .1,
               reference_flow=.8 if i == 3 else .2)
        hypothesis = [
            {"layout": "pip", "features": f},
            {"layout": "split", "features": _f(reference_motion=.1,
                                            reference_flow=.8)}
        ]
        w = {"start": i*2, "end": (i+1)*2,
             "layout": "pip", "geometry_layout": "none",
             "features": f, "similarity": .8 if i in (1, 2, 3) else .3,
             "nearest_cases": [{"case_id": 1, "similarity": .8}]}
        if with_hypotheses:
            w["hypotheses"] = hypothesis
        rows.append(w)
    return {"input": "/tmp/source.mp4",
            "scan": {"duration": 10, "same_source_as_training": True},
            "settings": {"min_similarity": .62}, "windows": rows}


def test_score_strategies_distinct_and_constrained():
    prof = _profile()
    descriptor = _f(reference_motion=.5, reference_neutral_ratio=1.0)
    scores = {}
    for name in s.SIMILARITY_STRATEGIES:
        value, examples = s._score_features(
            descriptor, prof, "pip", strategy=name)
        assert 0 <= value <= 1
        assert examples[0]["case_id"] in {1, 2}
        scores[name] = value
    assert scores["nearest"] != scores["top3"]
    assert scores["motion"] > scores["balanced"]
    with pytest.raises(ValueError, match="Unknown"):
        s._score_features(descriptor, prof, "pip", strategy="magic")


def test_median_window_smoothing_does_not_bridge_time_gap():
    x = [{"start": 0, "end": 2, "similarity": .8},
         {"start": 2, "end": 4, "similarity": .2},
         {"start": 4, "end": 6, "similarity": .9},
         {"start": 8, "end": 10, "similarity": .1}]
    y = e._median_smooth(x)
    assert y[1]["similarity"] == .8
    assert y[2]["similarity"] == .9
    assert y[3]["similarity"] == .1
    assert x[1]["similarity"] == .2


def test_fixed_budget_exact_not_threshold_tie():
    rows = [{"start": i, "end": i+1, "similarity": .8} for i in range(10)]
    selected = e._selected_windows(rows, 3)
    assert len(selected) == 3
    assert [r["start"] for r in selected] == [0, 1, 2]


def test_profile_rerank_keeps_alternate_hypotheses():
    rows, fallback = e._rescore_windows(_ranking()["windows"], _profile(), "motion")
    assert fallback == 0
    assert len(rows) == 5
    assert all(r["scoring_strategy"] == "motion" for r in rows)
    old, fixed = e._rescore_windows(
        _ranking(with_hypotheses=False)["windows"], _profile(), "top3")
    assert fixed == 5
    assert all(r["layout"] == "pip" for r in old)


def test_controlled_benchmark_no_precision_from_unknown(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "source": "source.mp4", "positives": [
            {"start": 2., "end": 6.}, {"start": 6., "end": 8.}]}))
    result = e.benchmark(_profile(), _ranking(), truth,
                         strategies=("balanced", "top3", "motion"),
                         smoothing=("off", "median3"), top_k=(1, 3),
                         budget_windows=3, min_duration=2)
    assert result["budget_window_count"] == 3
    assert len(result["trials"]) == 6
    assert result["same_source_as_training"] is True
    assert all(t["selected_windows"] == 3 for t in result["trials"])
    assert all(len(t["metrics"]) == 2 for t in result["trials"])
    assert all(m["precision_at_k"] is None
               for t in result["trials"] for m in t["metrics"])
    assert result["scene_boundary_mode"] == "off_for_controlled_retrieval_comparison"
    assert result["trials"][0]["single_layout_hypothesis_windows"] == 0


def test_audit_previous_human_reject_conflicts_and_no_negatives_inferred(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"source": "source.mp4", "positives": [
        {"start": 2, "end": 5}, {"start": 7, "end": 9}]}))
    labels = tmp_path / "review.csv"
    with labels.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "end", "label"])
        w.writeheader()
        w.writerow({"start": 3, "end": 4, "label": "reject"})
        w.writerow({"start": 5, "end": 6, "label": "reject"})  # UNKNOWN gap, not a conflict
        w.writerow({"start": 0, "end": 1, "label": "keep"})  # positive not exhaustive
        w.writerow({"start": 7, "end": 8, "label": "unsure"})
    result = e.audit_labels(truth, labels)
    assert result["potential_conflicting_reject_rows"] == 1
    assert result["source_identity_verified"] is False
    assert result["labels"] == {"keep": 1, "reject": 2, "unsure": 1}
    assert result["possible_time_axis_conflicts"][0]["overlapping_manual_cases"][0]["case_id"] == 1


def test_benchmark_cli_files_are_reproducible_without_video(tmp_path):
    profile = tmp_path / "profile.json"
    ranking = tmp_path / "ranking.json"
    truth = tmp_path / "truth.json"
    profile.write_text(json.dumps(_profile()))
    ranking.write_text(json.dumps(_ranking()))
    truth.write_text(json.dumps({"source": "source.mp4",
                                 "positives": [{"start": 2, "end": 6}]}))
    out = tmp_path / "experiment"
    rc = e.main(["--profile", str(profile), "--ranking", str(ranking),
                 "--truth", str(truth), "--output", str(out),
                 "--strategies", "balanced", "centroid", "--smoothing", "off",
                 "--top-k", "1", "2", "--budget-windows", "3"])
    assert rc == 0
    saved = json.loads((out / "experiments.json").read_text())
    assert saved["budget_window_count"] == 3
    assert len(saved["trials"]) == 2
    assert (out / "experiments.csv").is_file()


def test_wrong_truth_source_fails_closed(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"source": "different.mp4",
                                 "positives": [{"start": 2, "end": 4}]}))
    with pytest.raises(ValueError):
        e.benchmark(_profile(), _ranking(), truth)


def test_default_sparse_budget_sweep_avoids_near_entire_video(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "source": "source.mp4",
        "positives": [{"start": 2, "end": 6}]}))
    ranking = _ranking()
    # Simulates the uploaded experiment's failure mode: 99.8% of
    # baseline windows pass the old score threshold.
    for w in ranking["windows"]:
        w["similarity"] = .8
    result = e.benchmark(_profile(), ranking, truth,
                         strategies=("balanced", "motion"), smoothing=("off",),
                         top_k=(1, 3))
    assert result["baseline_selected_windows"] == 5
    assert result["budget_window_counts"] == [1, 2]
    assert result["budget_window_count"] is None
    assert len(result["trials"]) == 4
    assert all(t["selected_windows"] <= 2 for t in result["trials"])
    assert any("saturated" in warning for warning in result["warnings"])
    assert all("selected_window_positive_time_recall" in t for t in result["trials"])


def test_explicit_multiple_window_budgets_and_unknown_precision(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "source": "source.mp4",
        "positives": [{"start": 2, "end": 6}]}))
    result = e.benchmark(_profile(), _ranking(), truth,
                         budgets=(2, 4), strategies=("nearest",),
                         smoothing=("off", "median3"), top_k=(1, 2))
    assert result["budget_window_counts"] == [2, 4]
    assert result["budget_window_count"] is None
    assert len(result["trials"]) == 4
    assert {t["selected_windows"] for t in result["trials"]} == {2, 4}
    assert all(m["precision_at_k"] is None
               for t in result["trials"] for m in t["metrics"])


@pytest.mark.parametrize("budgets, single", [
    ((1, 2), 3), ((), None), ((0,), None), ((6,), None),
])
def test_budget_contract_rejects_conflicts_and_out_of_range(tmp_path, budgets, single):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "source": "source.mp4",
        "positives": [{"start": 2, "end": 6}]}))
    with pytest.raises(ValueError):
        e.benchmark(_profile(), _ranking(), truth,
                    budgets=budgets, budget_windows=single)
