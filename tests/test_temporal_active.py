"""v0.3 CPU-only temporal filtering + human-feedback active learning."""
import csv
import json

import cv2
import numpy as np
import pytest

import video_incese as v
import video_temporal_cv as cvmode
import video_active as active
from conftest import needs_ffmpeg


def _animated_frame(index: int, move_reference=True, move_rendered=True):
    frame = np.full((216, 384, 3), (65, 85, 90), dtype=np.uint8)
    frame[:73, 251:] = (20, 100, 170)
    rx = 278 + (index * 5 if move_reference else 0)
    gx = 37 + (index * 7 if move_rendered else 0)
    cv2.rectangle(frame, (rx, 19), (rx + 23, 53), (190, 220, 40), -1)
    cv2.rectangle(frame, (gx, 100), (gx + 35, 157), (240, 240, 240), -1)
    return frame


def test_static_reference_rejected_without_model():
    frames = [_animated_frame(i, move_reference=False) for i in range(5)]
    features = cvmode.calculate_features(frames, "pip", v.Settings(), 2.5)
    score, accepted = cvmode.rule_score(features)
    assert features["reference_motion"] == 0
    assert features["rendered_motion"] > .012
    assert not accepted
    assert 0 <= score <= 1


def test_two_animated_regions_have_motion_evidence():
    frames = [_animated_frame(i) for i in range(5)]
    features = cvmode.calculate_features(frames, "pip", v.Settings(), 2.5)
    score, accepted = cvmode.rule_score(features)
    assert features["reference_motion"] > .012
    assert features["rendered_motion"] > .012
    assert features["dual_motion_fraction"] > .45
    assert accepted, (score, features)


def test_rejected_window_never_bridged():
    windows = [
        {"start": 0, "end": 1, "accepted": True},
        {"start": 1, "end": 2, "accepted": True},
        {"start": 2, "end": 3, "accepted": False},
        {"start": 3, "end": 4, "accepted": True},
        {"start": 4, "end": 5, "accepted": True},
    ]
    assert cvmode._spans(windows, .8) == [(0, 2), (3, 5)]


def _report(path, count=30):
    windows = []
    for i in range(count):
        feats = {key: 0.0 for key in cvmode.FEATURE_NAMES}
        feats["reference_motion"] = i / (count * 2)
        feats["rendered_motion"] = (count-i) / (count * 3)
        feats["dual_motion_fraction"] = .1 + (i % 8) / 10
        feats["reference_flow"] = i / count / 4
        feats["rendered_flow"] = (count-i) / count / 4
        windows.append({
            "window_id": i + 1, "source_candidate": 1, "start": float(i), "end": float(i+1),
            "layout": "pip", "features": feats,
            "heuristic_score": i / count, "learned_probability": None,
            "accepted": i % 2 == 0, "reason": "dual_motion" if i % 2 == 0 else "insufficient_temporal_evidence",
        })
    path.write_text(json.dumps({"input": "/tmp/nonexistent.mp4",
                                 "temporal": {"mode": cvmode.FEATURE_VERSION},
                                 "temporal_windows": windows}, ensure_ascii=False))
    return windows


def test_active_queue_diversity_and_train_roundtrip(tmp_path):
    report_path = tmp_path / "report.json"
    _report(report_path)
    labels_path = tmp_path / "review.csv"
    result = active.queue(report_path, labels_path, budget=18)
    assert result["windows"] == 18
    with labels_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(not row["label"] for row in rows)
    assert {"keep", "reject"}.issubset({row["suggested"] for row in rows})
    for i, row in enumerate(rows):
        row["label"] = "keep" if i < 9 else "reject"
    with labels_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=active.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    model_path = tmp_path / "active-v1.json"
    meta = active.train([labels_path], model_path)
    assert meta["samples"] == 18
    model = active.load_model(model_path)
    assert model["training_samples"] == 18
    assert model["labeled_keep"] == 9
    score = active.predict_proba(model, _report(report_path)[0]["features"])
    assert 0 <= score <= 1


def test_labels_ignored_unless_human_verified(tmp_path):
    report = tmp_path / "report.json"
    _report(report)
    labels = tmp_path / "review.csv"
    active.queue(report, labels, 12)
    with pytest.raises(ValueError, match="Need >=12"):
        active.train([labels], tmp_path / "model.json")
    assert not (tmp_path / "model.json").exists()


def test_version_mismatch_rejected():
    example = {k: 0.0 for k in cvmode.FEATURE_NAMES}
    with pytest.raises(ValueError, match="Incompatible"):
        active.predict_proba({"model_version": "wrong", "feature_version": "wrong"}, example)


@needs_ffmpeg
def test_static_pip_scan_finds_no_motion(make_video):
    src = make_video([(1, 6)], seconds=8)
    settings = v.Settings(content_mode="temporal-cv", temporal_step=1,
                          temporal_fps=4, min_duration=.8)
    spans, windows, stats = cvmode.scan_temporal(src, [(1, 6)], settings, v.classify)
    assert spans == []
    assert len(windows) == 5
    assert all(not w["accepted"] for w in windows)
    assert stats["mode"] == cvmode.FEATURE_VERSION


@needs_ffmpeg
def test_default_temporal_mode_exports_only_mp4(make_video, tmp_path, monkeypatch):
    src = make_video([(1, 7)], seconds=8)
    def fake_detector(path, candidates, settings, classify, model=None):
        return [(2, 4, 1)], [{
            "window_id": 1, "source_candidate": 1, "start": 2, "end": 4,
            "layout": "pip", "features": {k: 0.0 for k in cvmode.FEATURE_NAMES},
            "heuristic_score": .8, "learned_probability": None, "accepted": True,
            "reason": "dual_motion",
        }], {"mode": cvmode.FEATURE_VERSION, "elapsed_seconds": .01,
             "windows": 1, "accepted_windows": 1, "failed_windows": [],
             "boundary_uncertainty_seconds": 1}
    monkeypatch.setattr(v, "scan_temporal", fake_detector)
    monkeypatch.setattr(v, "verify_ollama_frames", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("default mode must never call Ollama")))
    dest = tmp_path / "out"
    code = v.main(["--input", str(src), "--output", str(dest)])
    assert code == 0
    report = json.loads((dest / "report.json").read_text())
    assert report["settings"]["content_mode"] == "temporal-cv"
    assert report["clips"][0]["status"] == "exported"
    assert len(list((dest/"clips").glob("*.mp4"))) == 1
    assert not (dest/"evidence").exists()
    assert all(c["output"].endswith(".mp4") for c in report["clips"]
               if c["status"] == "exported")


def test_learned_model_path_serializable(monkeypatch, tmp_path):
    import dataclasses
    settings = v.Settings(learned_model=tmp_path/"model.json")
    settings_dict = dataclasses.asdict(settings)
    assert settings_dict["learned_model"].name == "model.json"


@needs_ffmpeg
def test_real_animated_pip_video_keeps_motion_and_drops_static_reference(tmp_path):
    """Full chain: encoded frames -> geometry -> temporal CV -> exported MP4."""
    import subprocess

    frames = tmp_path / "frames"
    frames.mkdir()
    rng = np.random.default_rng(42)
    bg = rng.integers(65, 105, size=(216, 384, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 1.0)
    for n in range(80):  # 10 seconds @ 8 fps
        second = n / 8.0
        frame = bg.copy()
        gen_x = 42 + n % 18
        cv2.rectangle(frame, (gen_x, 105), (gen_x + 52, 164), (240, 235, 230), -1)
        if 2 <= second < 8:
            frame[:73, 251:] = (25, 100, 165)
            ref_x = 275 + ((n-16) % 12 if second < 6 else 7)
            cv2.rectangle(frame, (ref_x, 16), (ref_x + 25, 54), (220, 245, 25), -1)
        assert cv2.imwrite(str(frames / f"{n:03d}.png"), frame)
    source = tmp_path / "moving.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "8", "-i", str(frames / "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    out = tmp_path / "real-out"
    ret = v.main([
        "--input", str(source), "--output", str(out),
        "--content-mode", "temporal-cv", "--sample-fps", "4",
        "--temporal-fps", "4", "--temporal-step", "1",
    ])
    assert ret == 0
    data = json.loads((out / "report.json").read_text())
    exported = [c for c in data["clips"] if c["status"] == "exported"]
    assert len(exported) >= 1, data["temporal"]
    assert all(c["start"] >= 1.5 and c["end"] <= 7 for c in exported), exported
    assert len(list((out/"clips").glob("*.mp4"))) == len(exported)
    assert any(w["reason"] == "insufficient_temporal_evidence"
               for w in data["temporal_windows"]), "Static reference interval not rejected"


def _window(start, *, accepted=False, reason="insufficient_temporal_evidence",
            reference_motion=0.0, rendered_motion=.05, reference_edges=.08,
            rendered_edges=.08, layout="pip"):
    f = {k: 0.0 for k in cvmode.FEATURE_NAMES}
    f.update(reference_motion=reference_motion, rendered_motion=rendered_motion,
             reference_flow=.2 if reference_motion else 0.0,
             reference_edges=reference_edges, rendered_edges=rendered_edges)
    return {
        "start": float(start), "end": float(start+1), "accepted": accepted,
        "raw_accepted": accepted, "reason": reason, "raw_reason": reason,
        "layout": layout, "features": f, "learned_probability": None,
        "decision_source": "temporal_rules", "_alternative_features": {},
    }


def test_context_rescues_static_reference_inside_known_comparison():
    windows = [
        _window(0, accepted=True, reason="dual_motion", reference_motion=.3),
        _window(1, accepted=False),
        _window(2, accepted=False),
        _window(3, accepted=True, reason="dual_motion", reference_motion=.3),
    ]
    cvmode._restore_context(windows, 2, None, .60)
    assert all(w["accepted"] for w in windows)
    assert [w["reason"] for w in windows[1:3]] == ["context_motion_pause"]*2
    assert cvmode._spans(windows, .8) == [(0, 4)]


def test_context_rescues_weak_layout_only_when_alternatives_have_visual_evidence():
    original = _window(1, reason="no_comparison_layout", layout="none")
    alternative = dict(original["features"])
    alternative.update(reference_motion=.09, rendered_motion=.15,
                       reference_flow=.4, rendered_flow=.5, dual_motion_fraction=1.0)
    original["_alternative_features"] = {"pip": alternative}
    windows = [
        _window(0, accepted=True, reason="dual_motion", reference_motion=.3),
        original,
        _window(2, accepted=True, reason="dual_motion", reference_motion=.3),
    ]
    cvmode._restore_context(windows, 2, None, .60)
    assert original["accepted"] and original["layout"] == "pip"
    assert original["reason"] == "context_layout_recovered"
    assert original["raw_reason"] == "no_comparison_layout"


def test_context_does_not_bridge_unverified_or_long_gaps():
    hard = _window(1, reason="frame_error", layout="none")
    windows = [
        _window(0, accepted=True, reason="dual_motion"),
        hard,
        _window(2, accepted=True, reason="dual_motion"),
    ]
    cvmode._restore_context(windows, 2, None, .60)
    assert not hard["accepted"]
    assert cvmode._spans(windows, .8) == [(0, 1), (2, 3)]
    many = [
        _window(0, accepted=True, reason="dual_motion"),
        _window(1), _window(2), _window(3),
        _window(4, accepted=True, reason="dual_motion"),
    ]
    cvmode._restore_context(many, 2, None, .60)
    assert not any(w["accepted"] for w in many[1:4])


def test_terminal_pause_limited_and_recoverable():
    windows = [
        _window(0, accepted=True, reason="dual_motion"),
        _window(1, accepted=True, reason="dual_motion"),
        _window(2),
    ]
    cvmode._restore_context(windows, 2, None, .60)
    assert windows[-1]["accepted"]
    assert windows[-1]["reason"] == "context_motion_pause"
    assert cvmode._spans(windows, .8) == [(0, 3)]
    too_long = [
        _window(0, accepted=True, reason="dual_motion"),
        _window(1, accepted=True, reason="dual_motion"),
        _window(2), _window(3),
    ]
    cvmode._restore_context(too_long, 2, None, .60)
    assert not too_long[-1]["accepted"]


def test_context_can_be_disabled():
    windows = [
        _window(0, accepted=True, reason="dual_motion"),
        _window(1),
        _window(2, accepted=True, reason="dual_motion"),
    ]
    cvmode._restore_context(windows, 0, None, .60)
    assert not windows[1]["accepted"]


def test_last_timestamp_retry_is_not_general_error_mask():
    class FakeCapture:
        def __init__(self):
            self.position = 0
        def set(self, attr, value):
            assert attr == cv2.CAP_PROP_POS_MSEC
            self.position = value/1000
        def get(self, attr):
            assert attr == cv2.CAP_PROP_FPS
            return 30
        def read(self):
            if self.position <= 20.34:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            return False, None
    cap = FakeCapture()
    frame, recovered = cvmode._frame(cap, 20.357, near_file_end=True)
    assert frame.shape == (8, 8, 3) and recovered
    with pytest.raises(RuntimeError, match="20.357"):
        cvmode._frame(cap, 20.357, near_file_end=False)
