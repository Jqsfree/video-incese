"""Independent CPU scene-boundary detection for already accepted temporal spans.

This stage does NOT decide if a frame is an acceptable white-model comparison.
It proposes cuts only when reference AND rendered regions change abruptly,
relative to their adjacent motion. Ordinary layout changes by themselves are
not evidence of a new case. Decisions and uncertainty are exposed for review.

A scene detector is not a semantic case/episode detector: similar-looking
scenes or gradual dissolves may remain merged.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_content import _regions

BOUNDARY_VERSION = "dual-region-scene-v1"


@dataclass(frozen=True)
class Signature:
    t: float
    layout: str
    reference_gray: np.ndarray
    rendered_gray: np.ndarray
    reference_hist: np.ndarray
    rendered_hist: np.ndarray


def _region_descriptor(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(region, (96, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return gray, histogram


def _signature(frame: np.ndarray, t: float, settings: object, classify: object) -> Signature | None:
    layout, _, _ = classify(frame, settings)
    if layout not in {"pip", "split"}:
        return None
    ref, rendered = _regions(frame, layout, settings.pip_x, settings.pip_y)
    rg, rh = _region_descriptor(ref)
    gg, gh = _region_descriptor(rendered)
    return Signature(t, layout, rg, gg, rh, gh)


def _region_change(gray_a: np.ndarray, hist_a: np.ndarray,
                   gray_b: np.ndarray, hist_b: np.ndarray) -> tuple[float, float, float]:
    changed_fraction = float(np.mean(cv2.absdiff(gray_a, gray_b) >= 32))
    histogram_distance = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
    score = 0.6 * changed_fraction + 0.4 * histogram_distance
    return score, changed_fraction, histogram_distance


def _distance(a: Signature | None, b: Signature | None) -> dict | None:
    # Treat a PiP->split switch as uncertain, not as an automatic scene break.
    if a is None or b is None or a.layout != b.layout:
        return None
    r, rp, rh = _region_change(a.reference_gray, a.reference_hist,
                               b.reference_gray, b.reference_hist)
    g, gp, gh = _region_change(a.rendered_gray, a.rendered_hist,
                               b.rendered_gray, b.rendered_hist)
    return {
        "score": round(min(r, g), 5),
        "reference_change": round(r, 5),
        "rendered_change": round(g, 5),
        "reference_pixel_change": round(rp, 5),
        "rendered_pixel_change": round(gp, 5),
        "reference_hist_change": round(rh, 5),
        "rendered_hist_change": round(gh, 5),
    }


def _abrupt_cut(distance: dict | None, before: dict | None, after: dict | None,
                threshold: float) -> bool:
    if distance is None or distance["score"] < threshold:
        return False
    # Color/composition change in both panels, or a near-complete grayscale
    # replacement. Excludes many fast pan/camera-motion false positives.
    min_hist = min(distance["reference_hist_change"], distance["rendered_hist_change"])
    min_pixel = min(distance["reference_pixel_change"], distance["rendered_pixel_change"])
    if min_hist < .10 and min_pixel < .75:
        return False
    neighbors = [x["score"] for x in (before, after) if x is not None]
    # No local baseline => insufficient evidence to claim a hard cut.
    if not neighbors:
        return False
    return distance["score"] >= max(.14, max(neighbors) * 1.35 + .025)


def _decode(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def _refine_cut(cap: cv2.VideoCapture, left: Signature, right: Signature,
                settings: object, classify: object) -> tuple[float, float]:
    """Bisect a bracket, based on similarity to either side of the cut."""
    a, b = left.t, right.t
    for _ in range(5):
        if b - a <= .09:
            break
        mid = (a + b) / 2
        frame = _decode(cap, mid)
        if frame is None:
            break
        current = _signature(frame, mid, settings, classify)
        ld = _distance(left, current)
        rd = _distance(current, right)
        if ld is None or rd is None:
            break
        if ld["score"] <= rd["score"]:
            a = mid
        else:
            b = mid
    return round((a + b) / 2, 3), round((b - a) / 2, 3)


def scan_boundaries(path: Path, spans: list[tuple[float, float, int]],
                    settings: object, classify: object) -> tuple[list[dict], dict]:
    """Return auditable events only. No detection labels or spans are mutated."""
    started = time.perf_counter()
    fps = float(settings.boundary_fps)
    min_length = max(float(settings.min_duration), float(settings.boundary_min_separation))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open scene boundary source: {path}")
    events: list[dict] = []
    failures: list[dict] = []
    pairs = 0
    proposals = 0
    try:
        for source_candidate, a, b in sorted(
            ((int(cid), float(start), float(end)) for start, end, cid in spans),
            key=lambda x: (x[0], x[1])):
            if b - a < 2 * min_length:
                continue
            count = int(math.floor((b - a) * fps))
            if count < 4:
                continue
            signatures: list[Signature | None] = []
            for i in range(count):
                t = a + (i + .5) / fps
                frame = _decode(cap, t)
                if frame is None:
                    failures.append({"source_candidate": source_candidate, "time": round(t, 3)})
                    signatures.append(None)
                    continue
                try:
                    signatures.append(_signature(frame, t, settings, classify))
                except (ValueError, cv2.error):
                    failures.append({"source_candidate": source_candidate, "time": round(t, 3)})
                    signatures.append(None)
            distance = [_distance(signatures[i-1], signatures[i])
                        for i in range(1, len(signatures))]
            pairs += len(distance)
            tentative = []
            for i in range(1, len(signatures) - 2):
                current = distance[i - 1]
                before = distance[i - 2] if i >= 2 else None
                after = distance[i] if i < len(distance) else None
                if not _abrupt_cut(current, before, after, float(settings.boundary_threshold)):
                    continue
                left, right = signatures[i - 1], signatures[i]
                if left is None or right is None:
                    continue
                timestamp, uncertainty = _refine_cut(cap, left, right, settings, classify)
                if timestamp - a < min_length or b - timestamp < min_length:
                    continue
                proposals += 1
                tentative.append({
                    "time": timestamp, "uncertainty_seconds": uncertainty,
                    "source_candidate": source_candidate,
                    "span_start": round(a, 3), "span_end": round(b, 3),
                    "layout": left.layout,
                    "score": current["score"],
                    "reference_change": current["reference_change"],
                    "rendered_change": current["rendered_change"],
                    "reference_hist_change": current["reference_hist_change"],
                    "rendered_hist_change": current["rendered_hist_change"],
                    "reason": "simultaneous_abrupt_scene_change",
                    "status": "candidate",
                })
            # Non-maximum suppression, retaining strong cuts while preventing
            # sub-minimum length clips. Never drop accepted video duration.
            chosen: list[dict] = []
            for e in sorted(tentative, key=lambda item: (-item["score"], item["time"])):
                if all(abs(e["time"] - saved["time"]) >= min_length - 1e-8
                       for saved in chosen):
                    e["status"] = "accepted"
                    chosen.append(e)
                else:
                    e["status"] = "suppressed_nearby"
            events.extend(sorted(tentative, key=lambda item: item["time"]))
    finally:
        cap.release()
    for i, event in enumerate(events, 1):
        event["event_id"] = i
    stats = {
        "mode": BOUNDARY_VERSION,
        "sample_fps": fps,
        "threshold": float(settings.boundary_threshold),
        "min_separation_seconds": float(settings.boundary_min_separation),
        "pairs_evaluated": pairs,
        "proposed_events": proposals,
        "accepted_events": sum(e["status"] == "accepted" for e in events),
        "failed_samples": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": ("Only abrupt changes in both stable-layout panels count. "
                        "Slow transitions, near-identical scenes, and semantic case changes "
                        "without obvious visual cuts may remain merged."),
    }
    return events, stats


def split_spans(spans: list[tuple[float, float, int]], events: list[dict]) -> list[dict]:
    """Partition accepted intervals on accepted scene events, preserving coverage.

    Unlike the original group-by-motion operation, this cannot merge distinct
    intervals, cross candidate IDs, or promote rejected windows.
    """
    out: list[dict] = []
    for start, end, candidate_id in spans:
        cuts = sorted(
            (e for e in events if e["status"] == "accepted"
             and e["source_candidate"] == candidate_id
             and e["span_start"] == round(float(start), 3)
             and e["span_end"] == round(float(end), 3)
             and float(start) < e["time"] < float(end)),
            key=lambda x: x["time"],
        )
        cursor = float(start)
        previous_event_id = None
        for e in cuts:
            out.append({
                "start": round(cursor, 3), "end": e["time"],
                "source_candidate": candidate_id, "status": "candidate",
                "boundary_before": previous_event_id,
                "boundary_after": e["event_id"],
            })
            cursor = e["time"]
            previous_event_id = e["event_id"]
        out.append({
            "start": round(cursor, 3), "end": round(float(end), 3),
            "source_candidate": candidate_id, "status": "candidate",
            "boundary_before": previous_event_id, "boundary_after": None,
        })
    return out
