"""Regression tests: failures must be recorded and reported, never silently shorten or erase results."""
import io
import json

import pytest

import video_incese as v
from conftest import needs_ffmpeg


@needs_ffmpeg
def test_single_undecodable_sample_does_not_truncate_clip(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 18)])
    original = v.read_frame
    monkeypatch.setattr(v, "read_frame", lambda cap, t: None if 7.7 < t < 7.8 else original(cap, t))
    report = v.process_file(src, tmp_path / "out", v.Settings(sample_fps=2, dry_run=True))
    assert [(c["start"], c["end"]) for c in report["clips"]] == [(2.0, 18.0)]
    assert report["sampling"]["failed_times"] == [7.75]
    assert not report["sampling"]["incomplete"]
    assert report["warnings"] and "could not be decoded" in report["warnings"][0]


@needs_ffmpeg
def test_unreadable_tail_is_flagged_incomplete_with_exit_code_2(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 18)])
    original = v.read_frame
    monkeypatch.setattr(v, "read_frame", lambda cap, t: None if t > 10 else original(cap, t))
    out = tmp_path / "out"
    code = v.main(["--input", str(src), "--output", str(out), "--sample-fps", "2", "--content-mode", "layout", "--dry-run"])
    data = json.loads((out / "report.json").read_text())
    assert code == 2
    assert data["sampling"]["incomplete"] and data["sampling"]["stopped_early"]
    assert any("Analysis incomplete" in w for w in data["warnings"])


@needs_ffmpeg
def test_vlm_error_is_recorded_and_other_candidates_still_processed(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 6), (12, 16)])
    calls = {"n": 0}

    def flaky(frame, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated Ollama failure")
        return True, "ok"
    monkeypatch.setattr(v, "verify_ollama", flaky)
    out = tmp_path / "out"
    code = v.main(["--input", str(src), "--output", str(out), "--sample-fps", "2",
                   "--content-mode", "layout", "--dry-run", "--ollama-model", "fake"])
    data = json.loads((out / "report.json").read_text())
    assert [c["status"] for c in data["clips"]] == ["verification_error", "dry_run"]
    assert "simulated" in data["clips"][0]["error"]
    assert code == 1


@needs_ffmpeg
def test_vlm_stops_after_consecutive_errors(make_video, tmp_path, monkeypatch):
    src = make_video([(1, 3), (5, 7), (9, 11), (13, 15)])
    def always_fail(frame, model):
        raise RuntimeError("down")
    monkeypatch.setattr(v, "verify_ollama", always_fail)
    report = v.process_file(src, tmp_path / "out", v.Settings(sample_fps=2, dry_run=True, ollama_model="fake"))
    assert [c["status"] for c in report["clips"]] == (
        ["verification_error"] * v.MAX_CONSECUTIVE_VERIFY_ERRORS + ["verification_skipped"])


@needs_ffmpeg
def test_rejected_candidate_keeps_evidence_frame(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 8)])
    monkeypatch.setattr(v, "verify_ollama", lambda frame, model: (False, "unrelated"))
    out = tmp_path / "out"
    report = v.process_file(src, out, v.Settings(sample_fps=2, ollama_model="fake", evidence=True))
    clip = report["clips"][0]
    assert clip["status"] == "rejected_semantic"
    assert clip["evidence"].endswith("_rejected.jpg") and (out / "evidence").exists()
    assert not (out / "clips").exists()


@needs_ffmpeg
def test_export_failure_is_recorded_not_raised(make_video, tmp_path, monkeypatch):
    src = make_video([(2, 8)])
    def boom(*a, **k):
        raise RuntimeError("ffmpeg clip export failed: simulated")
    monkeypatch.setattr(v, "export_clip", boom)
    out = tmp_path / "out"
    code = v.main(["--input", str(src), "--output", str(out), "--sample-fps", "2", "--content-mode", "layout"])
    data = json.loads((out / "report.json").read_text())
    assert data["clips"][0]["status"] == "export_error" and code == 1


def _fake_urlopen(body):
    return lambda request, timeout=None: io.BytesIO(body if isinstance(body, bytes) else json.dumps(body).encode())


def _frame():
    import numpy as np
    return np.zeros((216, 384, 3), dtype=np.uint8)


def test_verify_ollama_accepts_valid_response(monkeypatch):
    body = {"message": {"content": json.dumps({"match": True, "reason": "same character"})}}
    monkeypatch.setattr(v.urllib.request, "urlopen", _fake_urlopen(body))
    assert v.verify_ollama(_frame(), "m") == (True, "same character")


@pytest.mark.parametrize("body", [
    {"message": {}},
    {"message": {"content": "not json"}},
    {"message": {"content": json.dumps({"match": "yes"})}},
    {"message": {"content": json.dumps([1, 2])}},
    b"<html>proxy error</html>",
])
def test_verify_ollama_bad_responses_become_runtime_error(monkeypatch, body):
    monkeypatch.setattr(v.urllib.request, "urlopen", _fake_urlopen(body))
    with pytest.raises(RuntimeError):
        v.verify_ollama(_frame(), "m")
