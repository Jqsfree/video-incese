import shutil
import subprocess

import cv2
import numpy as np
import pytest

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                  reason="FFmpeg unavailable")


@pytest.fixture
def make_video(tmp_path):
    """Build a 10 fps MP4 whose frames are PiP-like inside the given (start_s, end_s) ranges."""
    def build(pip_ranges, seconds=20, name="source.mp4"):
        rng = np.random.default_rng(1)
        normal = np.full((216, 384, 3), 95, dtype=np.uint8)
        cv2.rectangle(normal, (30, 30), (125, 185), (210, 40, 100), -1)
        pip = rng.integers(35, 175, size=(216, 384, 3), dtype=np.uint8)
        pip[:73, 251:] = (20, 50, 160)
        cv2.circle(pip, (320, 35), 20, (205, 230, 70), -1)
        frames = tmp_path / f"frames_{name}"
        frames.mkdir()
        for i in range(seconds * 10):
            inside = any(a * 10 <= i < b * 10 for a, b in pip_ranges)
            assert cv2.imwrite(str(frames / f"{i:04d}.png"), pip if inside else normal)
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
                        "-i", str(frames / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                       check=True)
        return out
    return build
