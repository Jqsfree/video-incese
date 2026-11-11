"""v0.5 guards: avoid 1s overcut segments, keep inference model contract intact."""
from dataclasses import asdict

import cv2
import numpy as np

import video_boundaries as boundary
import video_incese as vi
import video_temporal_cv as temporal


def _paired_frames(positions: list[int], *, move_right=True):
    frames = []
    for x in positions:
        frame = np.full((216, 384, 3), 40, dtype=np.uint8)
        frame[:73, 251:] = (70, 90, 110)
        cv2.rectangle(frame, (270 + x, 13), (280 + x, 33), (220, 230, 230), -1)
        rx = (x if move_right else 0)
        cv2.rectangle(frame, (50 + rx, 90), (70 + rx, 130), (220, 230, 230), -1)
        frames.append(frame)
    return frames


def test_default_conservative_policy_is_configurable():
    defaults = vi.Settings()
    assert defaults.boundary_threshold == .55
    assert defaults.boundary_min_segment == 2.0
    assert defaults.motion_diagnostics is False
    assert "--boundary-min-segment" in vi.parser().format_help()
    assert "--motion-diagnostics" in vi.parser().format_help()
    assert "motion_diagnostics" in asdict(defaults)


def test_correspondence_is_diagnostic_not_semantic_classifier():
    frames = _paired_frames([0, 1, 1, 8, 8, 12, 12])
    positive = temporal.motion_correspondence(frames, "pip", vi.Settings())
    assert positive["correlation"] is not None
    assert .75 <= positive["correlation"] <= 1.0
    assert positive["pairs"] == len(frames)-1
    assert "not_a_match_classifier" in positive["status"]

    static_render = temporal.motion_correspondence(
        _paired_frames([0, 1, 1, 8, 8, 12, 12], move_right=False),
        "pip", vi.Settings())
    assert static_render["correlation"] is None
    assert static_render["status"] == "insufficient_motion_variation"


def test_no_implicit_model_v1_activation():
    settings = vi.Settings()
    assert settings.learned_model is None
    assert settings.content_mode == "layout"  # CLI default is independently temporal-cv
    assert vi.parser().parse_args(["--input", "x", "--output", "out"]).content_mode == "temporal-cv"


def test_guard_reduces_event_fragments_without_modifying_temporal_evidence():
    # Validate the partition contract against the post-filtered event sequence:
    # the accepted duration must remain unchanged after suppressing short cuts.
    raw_span = [(10.0, 20.0, 1)]
    events = [
        {"event_id": 1, "time": 11., "source_candidate": 1,
         "span_start": 10.0, "span_end": 20.0, "status": "suppressed_short_segment"},
        {"event_id": 2, "time": 14., "source_candidate": 1,
         "span_start": 10.0, "span_end": 20.0, "status": "accepted"},
        {"event_id": 3, "time": 15., "source_candidate": 1,
         "span_start": 10.0, "span_end": 20.0, "status": "suppressed_nearby"},
    ]
    pieces = boundary.split_spans(raw_span, events)
    assert [(x["start"], x["end"]) for x in pieces] == [(10., 14.), (14., 20.)]
    assert sum(x["end"]-x["start"] for x in pieces) == 10
