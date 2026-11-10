"""v0.2 temporal semantic filtering and boundary regression tests."""
import io
import json

import cv2
import numpy as np
import pytest

import video_content as content
import video_incese as v
from conftest import needs_ffmpeg


def _frame(n):
    img = np.zeros((216, 384, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (383, 215), (60, 70, 90), -1)
    # Strong PiP seams and distinct moving objects in both interiors.
    cv2.rectangle(img, (251, 0), (383, 73), (20, 100, 220), -1)
    cv2.rectangle(img, (30 + n * 2, 75), (90 + n * 2, 160), (220, 20, 20), -1)
    cv2.rectangle(img, (278+n, 14), (309+n, 45), (220, 220, 20), -1)
    return img


def test_motion_pair_rejects_static_reference():
    a = _frame(0)
    b = _frame(12)
    rm, gm = content.motion_pair(a, b, "pip", .655, .338)
    assert rm > .02 and gm > .02
    static = b.copy()
    static[:73, 251:] = a[:73, 251:]
    rm2, gm2 = content.motion_pair(a, static, "pip", .655, .338)
    assert rm2 == 0 and gm2 > .02


@pytest.mark.parametrize("raw,expected", [
    ({"match": True, "reference_video": True, "generated_video": True,
      "same_subject": True, "is_software_ui": False, "reason": "corresponding"}, True),
    ({"match": True, "reference_video": True, "generated_video": True,
      "same_subject": True, "is_software_ui": True, "reason": "website"}, False),
    ({"match": True, "reference_video": True, "generated_video": False,
      "same_subject": True, "is_software_ui": False}, False),
])
def test_ollama_consistency_gate(monkeypatch, raw, expected):
    body = {"message": {"content": json.dumps(raw)}}
    monkeypatch.setattr(v.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(body).encode()))
    assert v.verify_ollama_frames([_frame(i) for i in (0, 3, 6)], "mock")[0] == expected


def test_ollama_incomplete_fields_fail_closed(monkeypatch):
    body = {"message": {"content": json.dumps({"match": True, "reason": "x"})}}
    monkeypatch.setattr(v.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(body).encode()))
    with pytest.raises(RuntimeError, match="Malformed semantic decision"):
        v.verify_ollama_frames([_frame(0)]*3, "mock")


@needs_ffmpeg
def test_semantic_rejects_geometry_only_static_window(make_video, tmp_path):
    src = make_video([(2, 8)])
    # The synthetic PiP used by make_video is frozen over time.
    s = v.Settings(sample_fps=2, content_mode="semantic", ollama_model="fake")
    spans, trace = content.semantic_intervals(src, 2, 8, s, v.classify,
        lambda frames, model: (_ for _ in ()).throw(AssertionError("VLM should never run")))
    assert spans == []
    assert all(row["reason"] in {"missing_dual_video_motion", "no_comparison_layout"}
               for row in trace)


@needs_ffmpeg
def test_mixed_sequence_finds_separate_spans_and_no_bridging(make_video, tmp_path, monkeypatch):
    src = make_video([(0, 10)])
    # Synthetic frames are static; make motion prefilter pass to isolate time segmentation.
    monkeypatch.setattr(content, "motion_pair", lambda *args: (.5, .5))
    def semantic(frames, model):
        # Marker encoded by test monkeypatch to _capture from its requested timestamp.
        t = float(frames[1][0, 0, 0]) / 10.0
        return (2 <= t < 5 or 7 <= t < 9), "mock verdict"
    def capture(cap, t):
        image = np.full((216, 384, 3), min(255, round(t * 10)), dtype=np.uint8)
        image[:73, 251:] = (20, 100, 220)
        return image
    monkeypatch.setattr(content, "_capture", capture)
    spans, trace = content.semantic_intervals(
        src, 0, 10, v.Settings(content_mode="semantic", ollama_model="fake", min_duration=.8),
        lambda frame, settings: ("pip", 1, {}), semantic)
    assert len(spans) == 2, spans
    assert 1.5 <= spans[0][0] <= 2.5
    assert 4.5 <= spans[0][1] <= 5.5
    assert 6.5 <= spans[1][0] <= 7.5
    assert 8.5 <= spans[1][1] <= 9.5
    assert all(a < b for a, b in spans)
    assert any(row["phase"] == "boundary" for row in trace)


@needs_ffmpeg
def test_semantic_vlm_failure_reports_error_and_no_export(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 8)])
    monkeypatch.setattr(content, "motion_pair", lambda *args: (.4, .4))
    def broken(frames, model):
        raise RuntimeError("mock Ollama unavailable")
    monkeypatch.setattr(v, "verify_ollama_frames", broken)
    out = tmp_path / "out"
    result = v.process_file(src, out,
        v.Settings(sample_fps=2, content_mode="semantic", ollama_model="fake", dry_run=False))
    assert result["clips"][0]["status"] == "verification_error"
    assert "mock Ollama unavailable" in result["clips"][0]["error"]
    assert not (out / "clips").exists()


@needs_ffmpeg
def test_process_file_exports_two_refined_segments(make_video, tmp_path, monkeypatch):
    """Exercise FFmpeg integration after a single layout candidate is split."""
    src = make_video([(0, 10)], seconds=10)
    def approved_intervals(path, start, end, settings, classify, verify):
        return [(0.5, 2.5), (6.0, 8.0)], [
            {"time": 1.0, "match": True, "reason": "mock approved",
             "layout": "pip", "reference_motion": .3, "generated_motion": .4,
             "phase": "scan"}]
    monkeypatch.setattr(v, "semantic_intervals", approved_intervals)
    out = tmp_path / "out"
    result = v.process_file(src, out,
        v.Settings(sample_fps=2, content_mode="semantic", ollama_model="fake", min_duration=.8))
    exported = [c for c in result["clips"] if c["status"] == "exported"]
    assert len(exported) == 2, result["clips"]
    assert [(c["start"], c["end"]) for c in exported] == [(0.5, 2.5), (6.0, 8.0)]
    assert [c["source_candidate"] for c in exported] == [1, 1]
    assert result["source_candidates"] == 1
    assert len(result["verification_trace"]) == 1
    assert len(list((out/"clips").glob("*.mp4"))) == 2
    assert all(abs(v.probe(__import__("pathlib").Path(c["output"]))["duration"] - 2.0) < .15
               for c in exported)
