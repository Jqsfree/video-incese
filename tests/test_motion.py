"""Tests for video time continuity, static references, and whole-page scrolling.

Synthetic-only: these are contract regression tests, not production precision.
"""
import cv2
import numpy as np

from video_motion import temporal_motion


def frame_sequence(mode):
    rng = np.random.default_rng(19)
    h, w = 216, 384
    base = rng.integers(20, 130, size=(h, w, 3), dtype=np.uint8)
    frames = []
    for t in range(5):
        if mode == "scroll":
            frame = cv2.warpAffine(base,
                np.float32([[1, 0, 0], [0, 1, 2*t]]),
                (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            frame = np.full((h, w, 3), 70, dtype=np.uint8)
            frame[:73, 251:] = (100, 100, 100)
            if mode != "static":
                if mode != "static_reference":
                    cv2.circle(frame, (285 + 4*t, 25), 13, (230, 230, 230), -1)
                cv2.circle(frame, (112 + 7*t, 137), 25, (40, 160, 230), -1)
        frames.append(frame)
    return frames


def test_both_panels_show_continuous_local_motion():
    _, _, ok, details = temporal_motion(
        frame_sequence("both"), "pip", .655, .338,
        min_changed=.005, min_residual=.003, consistency=.5)
    assert ok, details
    assert details["reference"]["active_pairs"] >= 2
    assert details["rendered"]["active_pairs"] >= 2


def test_still_reference_with_moving_render_is_rejected():
    _, _, ok, details = temporal_motion(
        frame_sequence("static_reference"), "pip", .655, .338)
    assert not ok and details["reference"]["active_pairs"] == 0


def test_two_still_images_are_rejected():
    _, _, ok, details = temporal_motion(
        frame_sequence("static"), "pip", .655, .338)
    assert not ok and details["reference"]["active_pairs"] == 0
    assert details["rendered"]["active_pairs"] == 0


def test_global_scroll_alone_does_not_prove_animations():
    _, _, ok, details = temporal_motion(
        frame_sequence("scroll"), "pip", .655, .338,
        min_changed=.02, min_residual=.02, consistency=.5)
    assert not ok, details
