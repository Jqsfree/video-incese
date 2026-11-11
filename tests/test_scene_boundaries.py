"""v0.4 regression: content classification and scene segmentation are separate.

Synthetic clips exercise the boundary detector, while a monkeypatched
integration verifies that scene cuts partition (rather than create/delete)
accepted temporal intervals.
"""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

import video_incese as v
import video_boundaries as sb
from conftest import needs_ffmpeg


def _solid_signature(t: float, layout: str, color: tuple[int, int, int]) -> sb.Signature:
    frame = np.full((216, 384, 3), color, dtype=np.uint8)
    return sb._signature(frame, t, v.Settings(), lambda *_: (layout, 4.0, {}))


def test_boundary_change_requires_both_regions():
    a = _solid_signature(0, "pip", (10, 50, 70))
    b = _solid_signature(1, "pip", (180, 210, 230))
    strong = sb._distance(a, b)
    assert strong["score"] >= .4
    assert sb._abrupt_cut(strong, {"score": .02}, {"score": .03}, .40)
    assert not sb._abrupt_cut(strong, {"score": .9}, {"score": .9}, .40)


def test_switching_layout_without_scene_evidence_cannot_cut():
    pip = _solid_signature(0, "pip", (10, 50, 70))
    split = _solid_signature(1, "split", (180, 210, 230))
    assert sb._distance(pip, split) is None
    assert not sb._abrupt_cut(None, None, None, .4)


def test_split_spans_preserves_coverage_and_provenance():
    spans = [(10., 20., 1), (30., 40., 2)]
    events = [
        {"event_id": 1, "source_candidate": 1, "span_start": 10.,
         "span_end": 20., "status": "accepted", "time": 14.5},
        {"event_id": 2, "source_candidate": 1, "span_start": 10.,
         "span_end": 20., "status": "suppressed_nearby", "time": 15.},
        {"event_id": 3, "source_candidate": 2, "span_start": 30.,
         "span_end": 40., "status": "accepted", "time": 35.2},
    ]
    outputs = sb.split_spans(spans, events)
    assert [(x["start"], x["end"], x["source_candidate"]) for x in outputs] == [
        (10., 14.5, 1), (14.5, 20., 1),
        (30., 35.2, 2), (35.2, 40., 2)]
    assert sum(x["end"]-x["start"] for x in outputs) == pytest.approx(20)
    assert outputs[0]["boundary_after"] == outputs[1]["boundary_before"] == 1
    assert outputs[1]["boundary_after"] is None


def test_no_events_keeps_original_spans():
    accepted = [(2., 15., 1)]
    pieces = sb.split_spans(accepted, [])
    assert len(pieces) == 1
    assert (pieces[0]["start"], pieces[0]["end"]) == (2., 15.)


@needs_ffmpeg
def test_real_encoded_abrupt_cut_detected_but_one_panel_change_is_not(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    # Two separate 12-second clips, only the first changes both panels.
    def write_video(name, both):
        for i in range(96):  # 12 seconds @ 8 fps
            is_second = i >= 48
            frame = np.full((216, 384, 3), (30, 40, 60), dtype=np.uint8)
            frame[:73, 251:] = (40, 80, 180) if not is_second else (140, 210, 30)
            if not both:
                frame[:73, 251:] = (40, 80, 180)
            frame[73:, :235] = (65, 105, 155) if not is_second else (220, 30, 45)
            # Small moving objects should not create scene events.
            dx = i % 7
            cv2.rectangle(frame, (55+dx, 125), (78+dx, 155), (235, 235, 235), -1)
            cv2.rectangle(frame, (287+dx, 21), (305+dx, 48), (230, 220, 240), -1)
            assert cv2.imwrite(str(frames / f"{i:03d}.png"), frame)
        source = tmp_path / name
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "8",
            "-i", str(frames / "%03d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(source)
        ], check=True)
        return source

    settings = v.Settings(boundary_fps=3, boundary_threshold=.40,
                          boundary_min_separation=.85)
    fake_classifier = lambda frame, settings: ("pip", 4., {})
    source = write_video("two_panels.mp4", both=True)
    events, stats = sb.scan_boundaries(source, [(0, 12, 1)], settings, fake_classifier)
    accepted = [event for event in events if event["status"] == "accepted"]
    assert stats["mode"] == sb.BOUNDARY_VERSION
    assert stats["accepted_events"] >= 1, (events, stats)
    assert any(abs(e["time"] - 6) <= .4 for e in accepted)
    pieces = sb.split_spans([(0, 12, 1)], events)
    assert sum(p["end"] - p["start"] for p in pieces) == pytest.approx(12)
    source_one = write_video("one_panel.mp4", both=False)
    events_one, one_stats = sb.scan_boundaries(
        source_one, [(0, 12, 1)], settings, fake_classifier)
    assert one_stats["accepted_events"] == 0, events_one


@needs_ffmpeg
def test_content_evidence_and_boundary_events_reported_separately(make_video, tmp_path, monkeypatch):
    source = make_video([(1, 9)], seconds=10)
    def fake_temporal(path, candidates, settings, classify, model=None):
        return [(2., 8., 1)], [{
            "window_id": 1, "source_candidate": 1, "start": 2., "end": 8.,
            "accepted": True, "reason": "dual_motion", "layout": "pip",
            "features": {}, "heuristic_score": .8, "learned_probability": None,
        }], {"mode": "temporal-cv-1", "accepted_windows": 1, "failed_windows": []}
    event = {"event_id": 1, "time": 5., "source_candidate": 1,
             "span_start": 2., "span_end": 8., "status": "accepted",
             "reason": "simultaneous_abrupt_scene_change", "score": .88}
    called = []
    def fake_boundaries(path, spans, settings, classify):
        called.append(spans)
        return [event], {"mode": sb.BOUNDARY_VERSION, "accepted_events": 1,
                         "failed_samples": []}
    monkeypatch.setattr(v, "scan_temporal", fake_temporal)
    monkeypatch.setattr(v, "scan_boundaries", fake_boundaries)
    settings = v.Settings(content_mode="temporal-cv", dry_run=True, boundary_mode="scene")
    report = v.process_file(source, tmp_path / "split", settings)
    pieces = [c for c in report["clips"] if c["status"] == "dry_run"]
    assert [(c["start"], c["end"]) for c in pieces] == [(2., 5.), (5., 8.)]
    assert report["boundary_events"] == [event]
    assert report["temporal_windows"][0]["accepted"]
    assert report["boundaries"]["accepted_events"] == 1
    assert called == [[(2., 8., 1)]]
    assert pieces[0]["boundary_after"] == pieces[1]["boundary_before"] == 1

    disabled = v.process_file(source, tmp_path / "disabled",
                              v.Settings(content_mode="temporal-cv",
                                         boundary_mode="off", dry_run=True))
    assert len([c for c in disabled["clips"] if c["status"] == "dry_run"]) == 1
    assert len(called) == 1
    assert disabled["boundaries"]["mode"] == "off"


@needs_ffmpeg
def test_boundary_scanner_failure_is_reported_and_does_not_remove_content(make_video, tmp_path, monkeypatch):
    source = make_video([(2, 8)], seconds=10)
    def fake_temporal(path, candidates, settings, classify, model=None):
        return [(2., 8., 1)], [], {"mode": "temporal-cv-1", "failed_windows": []}
    def broken(*args):
        raise RuntimeError("simulated boundary decoder failure")
    monkeypatch.setattr(v, "scan_temporal", fake_temporal)
    monkeypatch.setattr(v, "scan_boundaries", broken)
    report = v.process_file(source, tmp_path / "out",
                            v.Settings(content_mode="temporal-cv", dry_run=True))
    assert report["boundaries"]["status"] == "scan_error"
    assert any("boundary analysis failed" in x for x in report["warnings"])
    assert [(c["start"], c["end"]) for c in report["clips"] if c["status"] == "dry_run"] == [
        (2., 8.)]
