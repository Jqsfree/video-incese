"""Detect and extract simultaneous reference/model + generated-video comparisons.

Deliberately conservative: CV detects compositing geometry, not semantic correspondence.
Optional local Ollama VLM checks semantic correspondence; it is not a guarantee.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Settings:
    sample_fps: float = 2.0
    min_duration: float = 0.8
    merge_gap: float = 1.5
    layout: str = "auto"
    pip_x: float = 0.655
    pip_y: float = 0.338
    pip_vertical: float = 4.0
    pip_horizontal: float = 3.0
    split_threshold: float = 26.0
    evidence: bool = False
    ollama_model: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class Observation:
    t: float
    layout: str
    score: float
    metrics: dict[str, float]

    @property
    def accepted(self) -> bool:
        return self.layout != "none"


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame")
    return cv2.cvtColor(cv2.resize(frame, (384, 216)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def classify(frame: np.ndarray, settings: Settings) -> tuple[str, float, dict[str, float]]:
    """Geometry-only classifier. Current baseline targets top-right PiP and centered split-screen.

    A positive is a *candidate*, not proof that model and generated content match.
    """
    g = _gray(frame)
    h, w = g.shape
    # At the inset top-left corner, a rectangular overlay creates two aligned seams.
    x = int(round(settings.pip_x * w))
    y = int(round(settings.pip_y * h))
    if not (int(.42 * w) < x < int(.85 * w) and int(.15 * h) < y < int(.55 * h)):
        raise ValueError("pip-x or pip-y outside supported normalized bounds")
    vertical = float(np.abs(g[5:y-3, x-2] - g[5:y-3, x+2]).mean())
    horizontal = float(np.abs(g[y-2, x+3:w-5] - g[y+2, x+3:w-5]).mean())
    inset_std = float(g[5:y-3, x+4:w-5].std())
    main_std = float(g[y+5:h-12, 8:w-8].std())
    # Top inset must contain non-flat visual information; weak dark cartoon insets are valid.
    pip_score = min(vertical / max(settings.pip_vertical, .01), horizontal / max(settings.pip_horizontal, .01))
    pip_ok = pip_score >= 1.0 and inset_std >= 6 and main_std >= 8

    # Vertical seam between two full-height panels, not a short inset edge.
    xs = list(range(int(w * .47), int(w * .54), 2))
    split_edges = [float(np.abs(g[int(.16*h):int(.78*h), xx-2] - g[int(.16*h):int(.78*h), xx+2]).mean()) for xx in xs]
    split_edge = max(split_edges)
    left_std = float(g[int(.14*h):int(.78*h), 10:int(.40*w)].std())
    right_std = float(g[int(.14*h):int(.78*h), int(.60*w):w-10].std())
    split_score = split_edge / max(settings.split_threshold, .01)
    split_ok = split_score >= 1.0 and left_std >= 9 and right_std >= 9

    metrics = {"pip_vertical": round(vertical, 2), "pip_horizontal": round(horizontal, 2),
               "inset_std": round(inset_std, 2), "main_std": round(main_std, 2),
               "split_edge": round(split_edge, 2), "left_std": round(left_std, 2),
               "right_std": round(right_std, 2)}
    if settings.layout in ("auto", "pip") and pip_ok:
        return "pip", round(float(pip_score), 3), metrics
    if settings.layout in ("auto", "split") and split_ok:
        return "split", round(float(split_score), 3), metrics
    return "none", round(float(max(pip_score, split_score)), 3), metrics


def time_intervals(observations: list[Observation], duration: float, settings: Settings) -> list[tuple[float, float]]:
    """Group positive sample cells; bridge short gaps only when bounded by positives."""
    dt = 1 / settings.sample_fps
    spans: list[list[float]] = []
    for o in observations:
        if not o.accepted:
            continue
        begin = max(0.0, o.t - dt / 2)
        finish = min(duration, o.t + dt / 2)
        if spans and begin - spans[-1][1] <= settings.merge_gap + 1e-8:
            spans[-1][1] = max(spans[-1][1], finish)
        else:
            spans.append([begin, finish])
    if spans and observations and observations[-1].accepted and duration - (observations[-1].t + dt / 2) <= dt / 2 + 1e-8:
        spans[-1][1] = duration
    return [(round(a, 3), round(b, 3)) for a, b in spans if b-a >= settings.min_duration - 1e-8]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=True)


def probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe missing; on macOS run: brew install ffmpeg")
    raw = _run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate", "-of", "json", str(path)]).stdout
    data = json.loads(raw)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise ValueError(f"No video stream: {path}")
    duration = float(data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {path}")
    return {"duration": duration, "width": video.get("width"), "height": video.get("height"),
            "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", []))}


def read_frame(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def sample_video(path: Path, duration: float, settings: Settings) -> list[Observation]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {path}")
    observations = []
    step = 1 / settings.sample_fps
    try:
        index = 0
        while True:
            t = (index + 0.5) * step
            if t >= duration:
                break
            frame = read_frame(cap, t)
            if frame is None:
                break
            layout, score, metrics = classify(frame, settings)
            observations.append(Observation(round(t, 4), layout, score, metrics))
            index += 1
    finally:
        cap.release()
    return observations


def verify_ollama(frame: np.ndarray, model: str) -> tuple[bool, str]:
    """Experimental, opt-in semantic verification using only loopback Ollama endpoint."""
    ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode verification frame")
    prompt = (
        "Inspect this video frame. Is a reference 3D model / animated reference video "
        "visible simultaneously with a separate GENERATED video depicting the SAME "
        "character or action, as a side-by-side or picture-in-picture comparison? "
        "Two unrelated videos, a single scene, overlays of text/logos, and ordinary "
        "video calls do NOT count. Answer exactly a JSON object with keys "
        "match (boolean) and reason (short string)."
    )
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
                         "messages": [{"role": "user", "content": prompt,
                                       "images": [base64.b64encode(jpg.tobytes()).decode("ascii")]}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            result = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Local Ollama unavailable ({model}): {exc}") from exc
    content = json.loads(result["message"]["content"])
    if type(content.get("match")) is not bool:
        raise RuntimeError(f"Ollama did not return boolean match: {content}")
    return content["match"], str(content.get("reason", ""))


def export_clip(source: Path, out: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing; on macOS run: brew install ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode intentionally: stream copy can produce imprecise cuts at non-keyframes.
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)]
    try:
        _run(command)
    except subprocess.CalledProcessError as exc:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg clip export failed: {exc.stderr}") from exc
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"Empty exported clip: {out}")


def process_file(source: Path, target: Path, settings: Settings) -> dict:
    info = probe(source)
    observations = sample_video(source, info["duration"], settings)
    candidates = time_intervals(observations, info["duration"], settings)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open for verification: {source}")
    clips = []
    try:
        for idx, (start, end) in enumerate(candidates, 1):
            entry: dict = {"start": start, "end": end, "duration": round(end-start, 3), "status": "candidate"}
            frame = None
            if settings.ollama_model or settings.evidence:
                frame = read_frame(cap, (start + end) / 2)
                if frame is None:
                    entry["status"] = "frame_error"
                    clips.append(entry)
                    continue
            if settings.ollama_model:
                match, reason = verify_ollama(frame, settings.ollama_model)
                entry["verification"] = {"match": match, "reason": reason, "model": settings.ollama_model}
                if not match:
                    entry["status"] = "rejected_semantic"
                    clips.append(entry)
                    continue
            entry["status"] = "dry_run" if settings.dry_run else "exported"
            out = target / "clips" / f"{source.stem}_{idx:04d}_{start:.3f}-{end:.3f}.mp4"
            if not settings.dry_run:
                export_clip(source, out, start, end)
                entry["output"] = str(out)
            if settings.evidence and frame is not None:
                evidence = target / "evidence" / f"{source.stem}_{idx:04d}.jpg"
                evidence.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(evidence), frame):
                    raise RuntimeError(f"Could not write evidence: {evidence}")
                entry["evidence"] = str(evidence)
            clips.append(entry)
    finally:
        cap.release()
    report = {"input": str(source.resolve()), "video": info, "settings": asdict(settings),
              "samples": [asdict(o) for o in observations], "clips": clips,
              "note": "Geometry-only detection identifies candidates, not confirmed semantic matches. "
                      "Optional local VLM checks a single midpoint frame, not the whole clip."}
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract simultaneous model/reference and generated-video comparisons (macOS-friendly)")
    p.add_argument("--input", required=True, type=Path, help="MP4 input file")
    p.add_argument("--output", required=True, type=Path, help="Output directory; source is untouched")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--min-duration", type=float, default=.8)
    p.add_argument("--merge-gap", type=float, default=1.5)
    p.add_argument("--layout", choices=("auto", "pip", "split"), default="auto")
    p.add_argument("--pip-x", type=float, default=.655, help="PiP top-left X, normalized [0,1]; default matches examples")
    p.add_argument("--pip-y", type=float, default=.338, help="PiP top-left Y, normalized [0,1]; default matches examples")
    p.add_argument("--pip-vertical", type=float, default=4)
    p.add_argument("--pip-horizontal", type=float, default=3)
    p.add_argument("--split-threshold", type=float, default=26)
    p.add_argument("--ollama-model", default=None, help="EXPERIMENTAL: local vision model e.g. qwen3-vl:4b")
    p.add_argument("--evidence", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = Settings(**{k: v for k, v in vars(args).items() if k not in {"input", "output"}})
        if not (.1 <= settings.sample_fps <= 30 and settings.min_duration > 0 and settings.merge_gap >= 0):
            raise ValueError("Require 0.1 <= sample-fps <= 30, min-duration > 0, merge-gap >= 0")
        source = args.input.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("--input must be a file")
        target = args.output.expanduser().resolve()
        if target == source or target in source.parents:
            raise ValueError("Invalid output path")
        if target == source.parent:
            raise ValueError("Choose a dedicated output directory, not the input's parent")
        report = process_file(source, target, settings)
        n = sum(c["status"] in {"exported", "dry_run"} for c in report["clips"])
        print(json.dumps({"input": str(source), "output": str(target), "candidates": len(report["clips"]),
                          "accepted": n, "report": str(target / "report.json")}, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
