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
from functools import partial

from video_content import semantic_intervals
from video_temporal_cv import scan_temporal
from video_active import load_model

import cv2
import numpy as np

# Robustness policy (see README "退出码与失败记录").
MAX_CONSECUTIVE_READ_FAILURES = 5
INCOMPLETE_TAIL_SECONDS = 2.0
MAX_CONSECUTIVE_VERIFY_ERRORS = 3
ERROR_STATUSES = frozenset({"frame_error", "verification_error", "verification_skipped", "export_error"})

def is_qwen3_vl_thinking_only(model: str) -> bool:
    """Ollama official qwen3-vl base tags are thinking-only, NOT hybrid switches."""
    tag = model.lower().strip()
    return tag in {"qwen3-vl", "qwen3-vl:latest", "qwen3-vl:2b",
                   "qwen3-vl:4b", "qwen3-vl:8b"} or (
        tag.startswith("qwen3-vl:") and "-thinking" in tag
    )


@dataclass(frozen=True)
class Settings:
    sample_fps: float = 2.0
    min_duration: float = 0.8
    merge_gap: float = 2.6
    layout: str = "auto"
    pip_x: float = 0.655
    pip_y: float = 0.338
    pip_vertical: float = 4.0
    pip_horizontal: float = 3.0
    split_threshold: float = 26.0
    evidence: bool = False
    ollama_model: str | None = None
    dry_run: bool = False
    content_mode: str = "layout"
    semantic_step: float = 1.0
    boundary_tolerance: float = 0.125
    motion_delta: float = 0.3
    motion_min: float = 0.02
    temporal_step: float = 1.0
    temporal_fps: float = 4.0
    learned_model: Path | None = None
    learned_threshold: float = 0.60
    ollama_timeout: float = 300.0
    ollama_num_predict: int = 192
    ollama_think: bool = False


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
    # A true rectangular inset produces a strong, spatially localized seam.
    # Ordinary scenery creates gradients of comparable strength on nearby lines.
    vertical_bg = float(np.median([
        np.abs(g[5:y-3, xx-2] - g[5:y-3, xx+2]).mean()
        for xx in (x-14, x-10, x-6, x+6, x+10, x+14)
    ]))
    horizontal_bg = float(np.median([
        np.abs(g[yy-2, x+3:w-5] - g[yy+2, x+3:w-5]).mean()
        for yy in (y-14, y-10, y-6, y+6, y+10, y+14)
    ]))
    vertical_ratio = vertical / (vertical_bg + 1)
    horizontal_ratio = horizontal / (horizontal_bg + 1)
    pip_score = min(vertical / max(settings.pip_vertical, .01), horizontal / max(settings.pip_horizontal, .01))
    pip_ok = (pip_score >= 1.0 and vertical_ratio >= 2.0 and horizontal_ratio >= 1.5
              and inset_std >= 6 and main_std >= 8)

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
               "right_std": round(right_std, 2), "vertical_ratio": round(vertical_ratio, 2),
               "horizontal_ratio": round(horizontal_ratio, 2)}
    if settings.layout in ("auto", "pip") and pip_ok:
        return "pip", round(float(pip_score), 3), metrics
    if settings.layout in ("auto", "split") and split_ok:
        return "split", round(float(split_score), 3), metrics
    return "none", round(float(max(pip_score, split_score)), 3), metrics


def _weak_comparison(o: Observation) -> bool:
    """Low-confidence seam evidence to bridge brief low-contrast PiP intervals.

    A blank or unrelated frame is not allowed to bridge merely because the
    configured maximum time gap has not elapsed.
    """
    m = o.metrics
    return (m.get("vertical_ratio", 0) >= 1.4
            and m.get("horizontal_ratio", 0) >= .85
            and m.get("pip_vertical", 0) >= 2
            and m.get("pip_horizontal", 0) >= 2)


def time_intervals(observations: list[Observation], duration: float, settings: Settings) -> list[tuple[float, float]]:
    """Group positive sample cells, bridging longer gaps only with weak evidence."""
    dt = 1 / settings.sample_fps
    spans: list[list[float]] = []
    previous_positive_idx = None
    for i, o in enumerate(observations):
        if not o.accepted:
            continue
        begin = max(0.0, o.t - dt / 2)
        finish = min(duration, o.t + dt / 2)
        if spans:
            gap = begin - spans[-1][1]
            interior = observations[previous_positive_idx + 1:i]
            bridge = (gap <= min(settings.merge_gap, dt + 1e-8)
                      or (gap <= settings.merge_gap and len(interior) > 0
                          and all(_weak_comparison(x) for x in interior)))
        else:
            bridge = False
        if bridge:
            spans[-1][1] = max(spans[-1][1], finish)
        else:
            spans.append([begin, finish])
        previous_positive_idx = i
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


def sample_video_with_stats(path: Path, duration: float, settings: Settings) -> tuple[list[Observation], dict]:
    """Sample the video; record unreadable samples, not silently terminate the scan."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {path}")
    observations: list[Observation] = []
    failed: list[float] = []
    step = 1 / settings.sample_fps
    expected = max(0, math.ceil(duration / step - 0.5 - 1e-9))
    consecutive = 0
    last_ok: float | None = None
    stopped_early = False
    try:
        for index in range(expected):
            t = (index + 0.5) * step
            frame = read_frame(cap, t)
            if frame is None:
                failed.append(round(t, 4))
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_READ_FAILURES:
                    stopped_early = True
                    break
                continue
            consecutive = 0
            layout, score, metrics = classify(frame, settings)
            observations.append(Observation(round(t, 4), layout, score, metrics))
            last_ok = t
    finally:
        cap.release()
    covered_to = 0.0 if last_ok is None else min(duration, last_ok + step / 2)
    tail = round(max(0.0, duration - covered_to), 3)
    stats = {"expected": expected, "read": len(observations), "failed_times": failed,
             "stopped_early": stopped_early, "unread_tail_seconds": tail,
             "incomplete": tail > max(INCOMPLETE_TAIL_SECONDS, step)}
    return observations, stats


def sample_video(path: Path, duration: float, settings: Settings) -> list[Observation]:
    return sample_video_with_stats(path, duration, settings)[0]


def _sampling_warnings(stats: dict, duration: float) -> list[str]:
    warnings = []
    failed = stats["failed_times"]
    if failed:
        shown = ", ".join(f"{t:.2f}s" for t in failed[:10]) + (" ..." if len(failed) > 10 else "")
        warnings.append(f"{len(failed)} sample(s) could not be decoded and were skipped (t = {shown}).")
    if stats["incomplete"]:
        warnings.append(f"Analysis incomplete: the last {stats['unread_tail_seconds']:.1f}s of {duration:.1f}s "
                        "were not analysed (frames unreadable); results cover only the earlier part.")
    return warnings

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
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Local Ollama unavailable or returned invalid data ({model}): {exc}") from exc
    try:
        content = json.loads(result["message"]["content"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Ollama returned an unparseable response ({model}): {exc}") from exc
    if not isinstance(content, dict) or type(content.get("match")) is not bool:
        raise RuntimeError(f"Ollama did not return boolean match: {content}")
    return content["match"], str(content.get("reason", ""))


def verify_ollama_frames(
    frames: list[np.ndarray], model: str, *,
    timeout: float = 300.0, num_predict: int = 192, think: bool = False,
) -> tuple[bool, str]:
    """Multi-frame reference/generated-content check. Fail closed on malformed VLM responses.

    The three images are chronological frames from the SAME video and window.
    """
    if not model:
        raise RuntimeError("Semantic mode requires --ollama-model (e.g. qwen3-vl:4b)")
    encoded = []
    for frame in frames:
        ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (768, 432)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("Cannot encode semantic verification frame")
        encoded.append(base64.b64encode(jpg.tobytes()).decode("ascii"))
    prompt = (
        "Three images show chronological frames from one continuous video interval. "
        "Evaluate whether there are TWO SIMULTANEOUS VIDEO regions: (1) a 3D reference "
        "model, motion capture, or reference character/action ANIMATION and (2) an AI-"
        "GENERATED output video depicting the CORRESPONDING character or action. "
        "Both regions must show temporal movement across the images. "
        "A website, browser, AI generator UI, Blender editor/tutorial, static character "
        "reference image, still collage, gameplay with webcam, text, or two unrelated "
        "videos DO NOT QUALIFY, even if a PiP border or vertical divider exists. "
        "When unsure, reject. Return ONLY JSON: match (boolean), "
        "reference_video (boolean), generated_video (boolean), same_subject (boolean), "
        "is_software_ui (boolean), reason (brief string)."
    )
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({"model": model, "stream": False, "format": "json",
                         "think": think,
                         "keep_alive": "20m",
                         "options": {"temperature": 0, "num_predict": num_predict},
                         "messages": [{"role": "user", "content": prompt, "images": encoded}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = json.load(resp)
        verdict = json.loads(raw["message"]["content"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Semantic VLM request/response failure ({model}): {exc}") from exc
    if not isinstance(verdict, dict) or any(type(verdict.get(k)) is not bool for k in (
        "match", "reference_video", "generated_video", "same_subject", "is_software_ui")):
        raise RuntimeError(f"Malformed semantic decision from {model}: {verdict}")
    valid = all(verdict[k] for k in ("match", "reference_video", "generated_video", "same_subject"))
    valid = valid and not verdict["is_software_ui"]
    return valid, str(verdict.get("reason", ""))


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


def _save_evidence(entry: dict, target: Path, source: Path, idx: int, frame: np.ndarray | None,
                   enabled: bool, suffix: str = "") -> None:
    if not enabled or frame is None:
        return
    evidence = target / "evidence" / f"{source.stem}_{idx:04d}{suffix}.jpg"
    try:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(evidence), frame):
            raise RuntimeError(f"Could not write evidence: {evidence}")
        entry["evidence"] = str(evidence)
    except (OSError, RuntimeError) as exc:
        entry["evidence_error"] = str(exc)


def process_file(source: Path, target: Path, settings: Settings) -> dict:
    """Per-clip failures are recorded in report.json, not raised."""
    info = probe(source)
    observations, sampling = sample_video_with_stats(source, info["duration"], settings)
    warnings = _sampling_warnings(sampling, info["duration"])
    candidates = time_intervals(observations, info["duration"], settings)
    prepared: list[dict] = []
    verification_trace: list[dict] = []
    temporal_windows: list[dict] = []
    temporal_stats: dict = {}
    if settings.content_mode == "temporal-cv":
        model = load_model(settings.learned_model)
        refined_spans, temporal_windows, temporal_stats = scan_temporal(
            source, candidates, settings, classify, model)
        by_candidate: dict[int, list[tuple[float, float]]] = {}
        for a, b, candidate_id in refined_spans:
            by_candidate.setdefault(candidate_id, []).append((a, b))
        for candidate_id, (a, b) in enumerate(candidates, 1):
            refined = by_candidate.get(candidate_id, [])
            if not refined:
                prepared.append({"start": a, "end": b,
                                 "source_candidate": candidate_id, "status": "rejected_temporal"})
            else:
                for left, right in refined:
                    prepared.append({"start": left, "end": right,
                                     "source_candidate": candidate_id, "status": "candidate"})
    semantic_failures = 0
    for candidate_id, (start, end) in enumerate(candidates, 1):
        if settings.content_mode == "temporal-cv":
            continue
        if settings.content_mode == "layout":
            prepared.append({"start": start, "end": end, "source_candidate": candidate_id,
                             "status": "candidate"})
            continue
        if semantic_failures >= MAX_CONSECUTIVE_VERIFY_ERRORS:
            prepared.append({"start": start, "end": end, "source_candidate": candidate_id,
                             "status": "verification_skipped",
                             "error": "Repeated semantic verification failures"})
            continue
        try:
            verified = partial(verify_ollama_frames, timeout=settings.ollama_timeout,
                               num_predict=settings.ollama_num_predict,
                               think=settings.ollama_think)
            refined, trace = semantic_intervals(
                source, start, end, settings, classify, verified)
        except (RuntimeError, ValueError) as exc:
            prepared.append({"start": start, "end": end, "source_candidate": candidate_id,
                             "status": "verification_error", "error": str(exc)})
            semantic_failures += 1
            continue
        semantic_failures = 0
        verification_trace.append({"source_candidate": candidate_id,
                                   "start": start, "end": end, "probes": trace})
        if not refined:
            prepared.append({"start": start, "end": end, "source_candidate": candidate_id,
                             "status": "rejected_semantic"})
            continue
        for a, b in refined:
            prepared.append({"start": a, "end": b, "source_candidate": candidate_id,
                             "status": "candidate"})
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open for verification: {source}")
    clips: list[dict] = []
    verify_errors = 0
    verify_aborted = False
    try:
        for idx, prepared_entry in enumerate(prepared, 1):
            start, end = prepared_entry["start"], prepared_entry["end"]
            entry: dict = {**prepared_entry, "duration": round(end-start, 3)}
            clips.append(entry)
            if entry["status"] != "candidate":
                if entry["status"] == "rejected_semantic" and settings.evidence:
                    frame = read_frame(cap, (start + end) / 2)
                    _save_evidence(entry, target, source, idx, frame, True, "_rejected")
                continue
            if verify_aborted:
                entry["status"] = "verification_skipped"
                continue
            frame = None
            if (settings.content_mode == "layout" and settings.ollama_model) or settings.evidence:
                frame = read_frame(cap, (start + end) / 2)
                if frame is None:
                    entry["status"] = "frame_error"
                    continue
            if settings.content_mode == "layout" and settings.ollama_model:
                try:
                    match, reason = verify_ollama(frame, settings.ollama_model)
                except RuntimeError as exc:
                    entry.update(status="verification_error", error=str(exc))
                    verify_errors += 1
                    verify_aborted = verify_errors >= MAX_CONSECUTIVE_VERIFY_ERRORS
                    _save_evidence(entry, target, source, idx, frame, settings.evidence, "_unverified")
                    continue
                verify_errors = 0
                entry["verification"] = {"match": match, "reason": reason, "model": settings.ollama_model}
                if not match:
                    entry["status"] = "rejected_semantic"
                    _save_evidence(entry, target, source, idx, frame, settings.evidence, "_rejected")
                    continue
            _save_evidence(entry, target, source, idx, frame, settings.evidence)
            if settings.dry_run:
                entry["status"] = "dry_run"
                continue
            out = target / "clips" / f"{source.stem}_{idx:04d}_{start:.3f}-{end:.3f}.mp4"
            try:
                export_clip(source, out, start, end)
            except RuntimeError as exc:
                entry.update(status="export_error", error=str(exc))
                continue
            entry["status"] = "exported"
            entry["output"] = str(out)
    finally:
        cap.release()
    report = {"input": str(source.resolve()), "video": info, "settings": asdict(settings),
              "sampling": sampling, "warnings": warnings,
              "samples": [asdict(o) for o in observations], "clips": clips,
              "source_candidates": len(candidates), "verification_trace": verification_trace,
              "temporal": temporal_stats, "temporal_windows": temporal_windows,
              "note": ("Temporal-CV mode uses CPU-only two-region motion and optical flow; "
                       "white-model/render correspondence is NOT semantically verified."
                       if settings.content_mode == "temporal-cv" else
                       "Semantic mode uses sampled images and VLM decisions."
                       if settings.content_mode == "semantic" else
                       "Layout-only mode can include unrelated picture-in-picture content.")}
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract simultaneous model/reference and generated-video comparisons (macOS-friendly)")
    p.add_argument("--input", required=True, type=Path, help="MP4 input file")
    p.add_argument("--output", required=True, type=Path, help="Output directory; source is untouched")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--min-duration", type=float, default=.8)
    p.add_argument("--merge-gap", type=float, default=2.6)
    p.add_argument("--layout", choices=("auto", "pip", "split"), default="auto")
    p.add_argument("--pip-x", type=float, default=.655, help="PiP top-left X, normalized [0,1]; default matches examples")
    p.add_argument("--pip-y", type=float, default=.338, help="PiP top-left Y, normalized [0,1]; default matches examples")
    p.add_argument("--pip-vertical", type=float, default=4)
    p.add_argument("--pip-horizontal", type=float, default=3)
    p.add_argument("--split-threshold", type=float, default=26)
    p.add_argument("--ollama-model", default=None, help="EXPERIMENTAL: local vision model e.g. qwen3-vl:4b")
    p.add_argument("--content-mode", choices=("temporal-cv", "layout", "semantic"), default="temporal-cv",
                   help="temporal-cv (default): CPU-only two-region motion; "
                        "layout: geometry baseline; semantic: slow Ollama VLM")
    p.add_argument("--temporal-step", type=float, default=1.0,
                   help="Length of each CPU temporal window, seconds (default 1)")
    p.add_argument("--temporal-fps", type=float, default=4.0,
                   help="CPU sample frames per second within each temporal window")
    p.add_argument("--learned-model", type=Path, default=None,
                   help="Optional JSON model trained from human review labels")
    p.add_argument("--learned-threshold", type=float, default=.60,
                   help="Minimum learned probability to keep a temporal window")
    p.add_argument("--semantic-step", type=float, default=1.0,
                   help="Seconds between VLM scan probes in semantic mode")
    p.add_argument("--boundary-tolerance", type=float, default=0.125,
                   help="Max duration of semantic boundary bisection bracket")
    p.add_argument("--motion-delta", type=float, default=0.3,
                   help="Half-width in seconds of three-frame motion observation")
    p.add_argument("--motion-min", type=float, default=0.02,
                   help="Minimum changed pixel ratio in BOTH video regions")
    p.add_argument("--ollama-timeout", type=float, default=300.0,
                   help="Seconds to wait for one local vision request (default: 300)")
    p.add_argument("--ollama-num-predict", type=int, default=192,
                   help="Max generated tokens for JSON classification (default: 192)")
    p.add_argument("--ollama-think", action="store_true",
                   help="Enable thinking for supported models; slower, default disabled")
    p.add_argument("--evidence", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = Settings(**{k: v for k, v in vars(args).items() if k not in {"input", "output"}})
        if not (.1 <= settings.sample_fps <= 30 and settings.min_duration > 0 and settings.merge_gap >= 0):
            raise ValueError("Require 0.1 <= sample-fps <= 30, min-duration > 0, merge-gap >= 0")
        if not (.2 <= settings.semantic_step <= 10
                and .025 <= settings.boundary_tolerance <= .5
                and .1 <= settings.motion_delta <= 1
                and 0 <= settings.motion_min <= .5):
            raise ValueError("Invalid semantic step, boundary tolerance, motion delta or motion minimum")
        if not (5 <= settings.ollama_timeout <= 3600 and 64 <= settings.ollama_num_predict <= 2048):
            raise ValueError("Require 5 <= ollama-timeout <= 3600 and 64 <= ollama-num-predict <= 2048")
        if not (.25 <= settings.temporal_step <= 5.0
                and 2 <= settings.temporal_fps <= 16
                and 0.5 <= settings.learned_threshold <= 0.95):
            raise ValueError("Require .25 <= temporal-step <= 5, 2 <= temporal-fps <= 16, "
                             "and .5 <= learned-threshold <= .95")
        if settings.learned_model and settings.content_mode != "temporal-cv":
            raise ValueError("--learned-model requires --content-mode temporal-cv")
        if settings.content_mode == "temporal-cv" and settings.ollama_model:
            raise ValueError("temporal-cv does not call Ollama; remove --ollama-model")
        if settings.content_mode == "semantic" and not settings.ollama_model:
            raise ValueError("Semantic mode requires --ollama-model (e.g. qwen3-vl:4b). "
                             "Use --content-mode layout only for geometry-only diagnostics.")
        if (settings.content_mode == "semantic" and settings.ollama_model
                and is_qwen3_vl_thinking_only(settings.ollama_model)):
            if not settings.ollama_think:
                raise ValueError(
                    f"{settings.ollama_model} is a Thinking-only Qwen3-VL tag. "
                    "Ollama cannot disable its thinking using think=false. "
                    "Use qwen3-vl:4b-instruct or qwen3-vl:8b-instruct. "
                    "Advanced: --ollama-think --ollama-num-predict 2048 "
                    "(expect substantially slower inference).")
            if settings.ollama_num_predict < 1024:
                raise ValueError("Thinking-only Qwen3-VL needs --ollama-num-predict >= 1024 "
                                 "to leave room for a JSON answer. Prefer an -instruct model.")
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
        errors = [(i, c) for i, c in enumerate(report["clips"], 1) if c["status"] in ERROR_STATUSES]
        for w in report["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
        for i, c in errors:
            print(f"error: clip {i} ({c['start']:.3f}-{c['end']:.3f}s) {c['status']}: {c.get('error', '')}", file=sys.stderr)
        incomplete = report["sampling"]["incomplete"]
        print(json.dumps({"input": str(source), "output": str(target), "candidates": len(report["clips"]),
                          "accepted": n, "errors": len(errors), "warnings": len(report["warnings"]),
                          "incomplete": incomplete, "report": str(target / "report.json")}, ensure_ascii=False))
        if errors:
            return 1
        return 2 if incomplete else 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
