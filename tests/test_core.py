from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from video_incese import Observation, Settings, classify, main, time_intervals


def synthetic():
    # Contrasting textured panels, with sharp rectangular inset edges.
    rng = np.random.default_rng(123)
    frame = rng.integers(30, 200, size=(720, 1280, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (5, 5), 1)
    return frame


def test_pip_recognized():
    frame = synthetic()
    x, y = int(1280*.655), int(720*.338)
    frame[:y, x:] = (220, 40, 20)
    # Distinct reference texture inside inset.
    cv2.rectangle(frame, (x+45, 30), (x+130, 130), (40, 255, 60), -1)
    label, score, metrics = classify(frame, Settings(layout="pip"))
    assert label == "pip", metrics
    assert score >= 1


def test_flat_solo_rejected():
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    assert classify(frame, Settings())[0] == "none"


def test_split_recognized():
    frame = synthetic()
    frame[:, :640] = 20
    frame[:, 640:] = 215
    cv2.circle(frame, (160, 300), 100, (210, 180, 90), -1)
    cv2.rectangle(frame, (830, 300), (1000, 500), (60, 30, 210), -1)
    assert classify(frame, Settings(layout="split"))[0] == "split"


def test_merge_and_min_length():
    settings = Settings(sample_fps=2, min_duration=.8, merge_gap=.5)
    xs = [Observation(t, "pip" if t in {.25, .75, 1.75, 3.75} else "none", 1, {})
          for t in [.25, .75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]]
    assert time_intervals(xs, 4, settings) == [(0, 2)]
    assert time_intervals([Observation(.25, "pip", 1, {})], 1, settings) == []
    assert time_intervals([Observation(.25, "pip", 1, {}), Observation(.75, "pip", 1, {})], 1.19, settings) == [(0.0, 1.19)]


def test_bad_cli_arguments(tmp_path):
    assert main(["--input", str(tmp_path / "nonexistent.mp4"), "--output", str(tmp_path / "out")]) == 1
    assert main(["--input", str(__file__), "--output", str(tmp_path), "--sample-fps", "0"]) == 1
