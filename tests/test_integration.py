"""End-to-end test, including FFmpeg encode/decode and accurate time-range export."""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from video_incese import Settings, classify, process_file


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_video_pipeline_excludes_solo_and_keeps_pip(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    rng = np.random.default_rng(77)
    normal = np.full((216, 384, 3), 95, dtype=np.uint8)
    cv2.rectangle(normal, (30, 30), (125, 185), (210, 40, 100), -1)
    pip = rng.integers(35, 175, size=(216, 384, 3), dtype=np.uint8)
    pip[:73, 251:] = (20, 50, 160)
    cv2.circle(pip, (320, 35), 20, (205, 230, 70), -1)
    assert classify(normal, Settings())[0] == "none"
    assert classify(pip, Settings())[0] == "pip"
    for i in range(40):
        current = pip if 10 <= i < 30 else normal
        assert cv2.imwrite(str(frame_dir / f"frame_{i:03d}.png"), current)

    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
        "-i", str(frame_dir / "frame_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    original_size = source.stat().st_size
    target = tmp_path / "out"
    report = process_file(source, target, Settings(sample_fps=5, merge_gap=.6))
    assert len(report["clips"]) == 1, report["clips"]
    one = report["clips"][0]
    assert one["status"] == "exported"
    assert one["start"] == pytest.approx(1.0, abs=.1)
    assert one["end"] == pytest.approx(3.0, abs=.1)
    clip = target / "clips" / next((target / "clips").iterdir()).name
    assert clip.is_file() and clip.stat().st_size > 1000
    assert source.stat().st_size == original_size
    data = json.loads((target / "report.json").read_text())
    assert data["clips"][0]["status"] == "exported"
