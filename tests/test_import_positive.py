"""Human-cut positives remain manually grounded, traceable and training compatible."""
import csv
import json

import pytest

import video_active
import video_positive
from conftest import needs_ffmpeg


def test_confirmation_required(tmp_path):
    with pytest.raises(ValueError, match="confirmed-reviewed"):
        video_positive.import_positives(
            [tmp_path / "sample.mp4"], tmp_path / "gold.csv",
            tmp_path / "reports")


@needs_ffmpeg
def test_import_positive_actual_mp4_make_report_and_labels(make_video, tmp_path):
    clip = make_video([(1, 7)], seconds=8)
    out = tmp_path / "gold.csv"
    stats = video_positive.import_positives(
        [clip], out, tmp_path / "reports",
        confirmed_reviewed=True,
        temporal_step=1.0, temporal_fps=4, edge_trim=0.25,
    )
    assert stats["positive_clips"] == 1
    assert stats["negative_windows"] == 0
    assert out.exists()
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 7
    assert all(r["label"] in {"keep", "unsure"} for r in rows)
    assert all(r["preview"] == str(clip.resolve()) for r in rows)
    assert all(r["report_path"] == rows[0]["report_path"] for r in rows)
    assert rows[0]["label"] == "unsure"  # first cut boundary
    assert rows[-1]["label"] == "unsure"  # last cut boundary
    report_path, report = video_active._load_report(
        video_positive.Path(rows[0]["report_path"]))
    assert report_path.exists()
    assert report["manual_ground_truth"]["origin"] == "human_edited_clip"
    assert report["manual_ground_truth"]["scope"] == "whole_clip_positive"
    assert report["input"] == str(clip.resolve())
    assert len(report["temporal_windows"]) == len(rows)
    assert report["source_sha256"] == video_positive._sha256(clip)
    with pytest.raises(ValueError, match="Need >=12"):
        video_active.train([out], tmp_path / "should-not-train-positive-only.json")


@needs_ffmpeg
def test_import_rejects_duplicate_clips(make_video, tmp_path):
    clip = make_video([(1, 5)], seconds=7)
    with pytest.raises(ValueError, match="Duplicate positive clip"):
        video_positive.import_positives(
            [clip, clip], tmp_path / "gold.csv", tmp_path / "reports",
            confirmed_reviewed=True)


def test_explicit_cli_without_confirmation_is_error(tmp_path):
    rc = video_active.main([
        "import-positive", "--clip", str(tmp_path / "not-real.mp4"),
        "--output", str(tmp_path / "gold.csv"),
        "--reports-dir", str(tmp_path / "reports"),
    ])
    assert rc == 1
    assert not (tmp_path / "gold.csv").exists()
