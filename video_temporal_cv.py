"""CPU-only temporal evidence for simultaneous reference and rendered video.

It establishes TWO continuously changing visual regions. It does not prove the
3D white model and the rendered video depict the same action; that still
requires annotated data or a separate semantic model.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from video_content import _regions

FEATURE_VERSION = "temporal-cv-1"
FEATURE_NAMES = (
    "reference_motion", "rendered_motion",
    "reference_flow", "rendered_flow",
    "reference_coherence", "rendered_coherence",
    "reference_edges", "rendered_edges",
    "dual_motion_fraction", "scene_cut_fraction",
    "layout_strength",
)


def _gray_region(region: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (96, 64))


def _pair_stats(a: np.ndarray, b: np.ndarray, use_flow: bool) -> tuple[float, float, float]:
    """Local motion, flow coverage, and dominant global translation."""
    aa, bb = _gray_region(a), _gray_region(b)
    changed = float(np.mean(cv2.absdiff(aa, bb) > 12))
    if not use_flow:
        return changed, 0.0, 0.0
    flow = cv2.calcOpticalFlowFarneback(aa, bb, None, .5, 2, 11, 2, 5, 1.1, 0)
    mag = np.linalg.norm(flow, axis=2)
    moving = mag > .55
    fraction = float(np.mean(moving))
    if not np.any(moving):
        coherence = 0.0
    else:
        vectors = flow[moving]
        coherence = float(np.linalg.norm(np.median(vectors, axis=0))
                          / max(.001, float(np.mean(mag[moving]))))
    return changed, fraction, float(np.clip(coherence, 0.0, 1.0))


def _edges(region: np.ndarray) -> float:
    return float(np.mean(cv2.Canny(_gray_region(region), 80, 170) > 0))


def calculate_features(frames: list[np.ndarray], layout: str, settings: object,
                       layout_strength: float = 1.0) -> dict[str, float]:
    if len(frames) < 3:
        raise ValueError("At least three frames are required for motion evidence")
    references, rendereds = zip(*[_regions(f, layout, settings.pip_x, settings.pip_y) for f in frames])
    ref_d, ren_d, ref_f, ren_f, ref_co, ren_co = [], [], [], [], [], []
    scene_cuts = []
    for i in range(len(frames) - 1):
        # Flow every second pair: frame differences cover all samples, optical flow
        # supplements rather than replacing the inexpensive temporal signal.
        use_flow = (i % 2 == 0)
        x = _pair_stats(references[i], references[i + 1], use_flow)
        y = _pair_stats(rendereds[i], rendereds[i + 1], use_flow)
        ref_d.append(x[0])
        ren_d.append(y[0])
        if use_flow:
            ref_f.append(x[1])
            ren_f.append(y[1])
            ref_co.append(x[2])
            ren_co.append(y[2])
        g1 = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (96, 64))
        g2 = cv2.resize(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY), (96, 64))
        scene_cuts.append(float(np.mean(cv2.absdiff(g1, g2) > 55)) > .65)
    metrics = {
        "reference_motion": float(np.median(ref_d)),
        "rendered_motion": float(np.median(ren_d)),
        "reference_flow": float(np.median(ref_f)),
        "rendered_flow": float(np.median(ren_f)),
        "reference_coherence": float(np.median(ref_co)),
        "rendered_coherence": float(np.median(ren_co)),
        "reference_edges": _edges(references[len(references) // 2]),
        "rendered_edges": _edges(rendereds[len(rendereds) // 2]),
        "dual_motion_fraction": float(np.mean([(x >= .012 and y >= .012)
                                               for x, y in zip(ref_d, ren_d)])),
        "scene_cut_fraction": float(np.mean(scene_cuts)),
        "layout_strength": float(np.clip(layout_strength / 4, 0.0, 1.0)),
    }
    return {name: round(metrics[name], 5) for name in FEATURE_NAMES}


def rule_score(features: dict[str, float]) -> tuple[float, bool]:
    """Conservative temporal gate. Content semantics cannot be deduced from it."""
    r, g = features["reference_motion"], features["rendered_motion"]
    dual = features["dual_motion_fraction"]
    cut = features["scene_cut_fraction"]
    flow = min(features["reference_flow"], features["rendered_flow"])
    score = float(np.clip(
        .45 * dual + .35 * min(1.0, min(r, g) / .06)
        + .20 * min(1.0, flow / .07) - .4 * cut,
        0.0, 1.0,
    ))
    keep = (r >= .012 and g >= .012 and dual >= .45 and
            cut <= .25 and score >= .48)
    return round(score, 4), bool(keep)


def _frame(cap: cv2.VideoCapture, t: float, *, near_file_end: bool = False) -> tuple[np.ndarray, bool]:
    """Read exact sample, retry only at the true container tail.

    MP4 format.duration can be slightly longer than the timestamp of the last
    decodable frame. Never hide a decode error in the middle of the video.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame, False
    if near_file_end:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if math.isfinite(fps) and 1 <= fps <= 240:
            for frames_back in (1, 2, 4, 6):
                retry_t = max(0.0, t - frames_back / fps)
                cap.set(cv2.CAP_PROP_POS_MSEC, retry_t * 1000)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame, True
    raise RuntimeError(f"Undecodable temporal sample at {t:.3f}s")


def _spans(windows: list[dict], min_duration: float) -> list[tuple[float, float]]:
    """Do not bridge an explicitly rejected temporal window."""
    spans: list[list[float]] = []
    for w in windows:
        if not w["accepted"]:
            continue
        a, b = float(w["start"]), float(w["end"])
        if spans and a <= spans[-1][1] + .0001:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in spans
            if b - a >= min_duration - 1e-8]


def _context_supported(row: dict, previous: dict | None, following: dict | None,
                       model: dict | None, threshold: float) -> bool:
    """Conservative evidence for short uncertain windows, never hard negatives."""
    if row["reason"] not in {"insufficient_temporal_evidence", "no_comparison_layout"}:
        return False
    if row["reason"] == "no_comparison_layout":
        choices = row.get("_alternative_features", {})
        if not choices:
            return False
        nearby = {v["layout"] for v in (previous, following)
                  if v is not None and v["layout"] in {"pip", "split"}}
        candidates = []
        for mode in nearby:
            if mode in choices:
                f = choices[mode]
                score, _ = rule_score(f)
                candidates.append((score, mode, f))
        if not candidates:
            return False
        _, inferred, features = max(candidates, key=lambda v: v[0])
    else:
        inferred = row["layout"]
        features = row["features"]
    if not (features["rendered_motion"] >= .012
            and features["reference_edges"] >= .015
            and features["rendered_edges"] >= .012
            and features["scene_cut_fraction"] <= .25):
        return False
    if model is not None:
        from video_active import predict_proba
        probability = predict_proba(model, features)
        if probability < threshold:
            return False
        row["learned_probability"] = round(probability, 4)
    row["layout"] = inferred
    row["features"] = features
    row["accepted"] = True
    row["reason"] = "context_layout_recovered" if row["raw_reason"] == "no_comparison_layout" else "context_motion_pause"
    row["decision_source"] = "short_gap_context"
    return True


def _restore_context(windows: list[dict], max_gap: float, model: dict | None,
                     threshold: float) -> None:
    """Rescue only weak/uncertain windows with nearby strong valid evidence.

    A zero-motion reference inside a known comparison can be a brief animation
    pause, but there is no reliable way to identify long static screenshots
    from optical flow alone. Hard errors, scene cuts and learned rejections are
    never bridged. Every rescued window is tagged for manual review.
    """
    if max_gap <= 0:
        return
    i = 0
    while i < len(windows):
        if windows[i]["accepted"]:
            i += 1
            continue
        j = i
        while j < len(windows) and not windows[j]["accepted"]:
            j += 1
        group = windows[i:j]
        seconds = sum(w["end"] - w["start"] for w in group)
        left = windows[i - 1] if i > 0 else None
        right = windows[j] if j < len(windows) else None
        interior = left is not None and right is not None and seconds <= max_gap + 1e-6
        terminal = (left is not None and right is None and j == len(windows)
                    and i >= 2 and windows[i - 2]["accepted"]
                    and seconds <= min(max_gap, 1.5) + 1e-6)
        if interior or terminal:
            if all(w["reason"] in {"insufficient_temporal_evidence", "no_comparison_layout"}
                   for w in group):
                # Precheck every window: either ALL are rescued or NONE,
                # avoiding partial bridges that would hide uncertainty.
                tentative = [dict(w) for w in group]
                if all(_context_supported(w, left, right, model, threshold) for w in tentative):
                    for original, replacement in zip(group, tentative):
                        original.update(replacement)
        i = j


def scan_temporal(path: Path, candidates: list[tuple[float, float]], settings: object,
                  classify: object, model: dict | None = None) -> tuple[list[tuple[float, float, int]], list[dict], dict]:
    """Scan candidate times, returning accepted windows, all evidence, and timing."""
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open temporal video: {path}")
    accepted: list[tuple[float, float, int]] = []
    windows: list[dict] = []
    failed = []
    interval = float(settings.temporal_step)
    recovered_tail_samples: list[float] = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    approximate_last_frame = ((nframes - 1) / fps if fps > 0 and nframes > 0 else None)
    try:
        for candidate_id, (start, end) in enumerate(candidates, 1):
            count = math.ceil((end - start) / interval)
            own: list[dict] = []
            for i in range(count):
                a = round(start + i * interval, 4)
                b = round(min(end, a + interval), 4)
                if b - a < .25:
                    continue
                times = np.linspace(a + min(.03, (b-a)/8), b - min(.03, (b-a)/8),
                                    max(3, int(math.ceil((b-a)*settings.temporal_fps))+1))
                try:
                    reads = [
                        _frame(cap, float(t), near_file_end=(
                            approximate_last_frame is not None and
                            float(t) >= approximate_last_frame - .05 and
                            float(t) <= approximate_last_frame + .45))
                        for t in times
                    ]
                    frames = [frame for frame, _ in reads]
                    recovered_tail_samples.extend(
                        round(float(t), 3) for t, (_, recovered) in zip(times, reads) if recovered
                    )
                    label, strength, _ = classify(frames[len(frames)//2], settings)
                    alternates = {}
                    if label == "none":
                        feats = {k: 0.0 for k in FEATURE_NAMES}
                        score, gate = 0.0, False
                        reason = "no_comparison_layout"
                        # Geometry alone is not proof of absence; retain
                        # alternative region evidence for nearby anchors.
                        alternates = {mode: calculate_features(frames, mode, settings)
                                      for mode in ("pip", "split")}
                    else:
                        feats = calculate_features(frames, label, settings, strength)
                        score, gate = rule_score(feats)
                        reason = "dual_motion" if gate else "insufficient_temporal_evidence"
                    prediction = None
                    if model is not None and label != "none":
                        from video_active import predict_proba
                        prediction = predict_proba(model, feats)
                        if prediction < settings.learned_threshold:
                            gate = False
                            reason = "rejected_learned"
                        elif not gate:
                            # Learned feedback may RESTORE a weak-motion
                            # window, unlike v0.3.1's veto-only rule.
                            plausible = (feats["rendered_motion"] >= .012
                                         and feats["reference_edges"] >= .015
                                         and feats["reference_flow"] >= .02
                                         and feats["scene_cut_fraction"] <= .25)
                            if plausible:
                                gate = True
                                reason = "accepted_learned"
                    row = {
                        "window_id": len(windows) + 1, "source_candidate": candidate_id,
                        "start": a, "end": b, "layout": label,
                        "features": feats, "heuristic_score": score,
                        "learned_probability": (round(prediction, 4) if prediction is not None else None),
                        "accepted": bool(gate), "reason": reason,
                        "raw_accepted": bool(gate), "raw_reason": reason,
                        "decision_source": ("learned" if reason in {"accepted_learned", "rejected_learned"}
                                            else "temporal_rules"),
                        "_alternative_features": alternates,
                    }
                except (RuntimeError, cv2.error, ValueError) as exc:
                    failed.append({"start": a, "end": b, "error": str(exc)})
                    row = {
                        "window_id": len(windows) + 1, "source_candidate": candidate_id,
                        "start": a, "end": b, "layout": "none",
                        "features": {k: 0.0 for k in FEATURE_NAMES},
                        "heuristic_score": 0.0, "learned_probability": None,
                        "accepted": False, "reason": "frame_error",
                        "raw_accepted": False, "raw_reason": "frame_error",
                        "decision_source": "decoder_error", "_alternative_features": {},
                    }
                windows.append(row)
                own.append(row)
            _restore_context(own, float(settings.temporal_context_gap), model,
                             float(settings.learned_threshold))
            for row in own:
                row.pop("_alternative_features", None)
            for a, b in _spans(own, settings.min_duration):
                accepted.append((a, b, candidate_id))
    finally:
        cap.release()
    stats = {"mode": FEATURE_VERSION, "windows": len(windows),
             "accepted_windows": sum(bool(w["accepted"]) for w in windows),
             "elapsed_seconds": round(time.perf_counter() - started, 3),
             "failed_windows": failed,
             "recovered_tail_samples": recovered_tail_samples,
             "context_recovered_windows": sum(w["decision_source"] == "short_gap_context" for w in windows),
             "learned_recovered_windows": sum(w["reason"] == "accepted_learned" for w in windows),
             "boundary_uncertainty_seconds": interval}
    return accepted, windows, stats
