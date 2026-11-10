"""Multi-frame content verification and subsecond boundary localization.

A positive requires detectable motion in BOTH reference and generated regions and
a separate VLM semantic approval. Motion is only a prefilter, never proof of
semantic correspondence. Every candidate is independently evaluated.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ContentProbe:
    time: float
    match: bool
    reason: str
    layout: str
    reference_motion: float
    generated_motion: float
    phase: str = "scan"


def _regions(frame: np.ndarray, layout: str, pip_x: float, pip_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop reference and generated video interiors, avoiding captions/borders."""
    image = cv2.resize(frame, (384, 216))
    h, w = image.shape[:2]

    def crop(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        return image[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]

    if layout == "pip":
        # Examples: reference overlay is top right; generated video is underlying.
        # Bounds are derived from configurable overlay origin.
        reference = crop(min(.98, pip_x + .055), .055, .975, max(.13, pip_y - .055))
        generated = crop(.075, .26, max(.38, pip_x - .07), .82)
    elif layout == "split":
        # Examples: reference in the right half, generated in the left.
        reference = crop(.57, .20, .94, .80)
        generated = crop(.06, .20, .43, .80)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    if min(reference.shape[:2]) < 8 or min(generated.shape[:2]) < 8:
        raise ValueError("Reference/generated video region is too small")
    return reference, generated


def _motion(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of significantly changed grayscale pixels in a region."""
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(a_gray, b_gray) > 14))


def motion_pair(before: np.ndarray, after: np.ndarray, layout: str,
                pip_x: float, pip_y: float) -> tuple[float, float]:
    """Return (reference, generated) temporal motion fractions."""
    ref_a, gen_a = _regions(before, layout, pip_x, pip_y)
    ref_b, gen_b = _regions(after, layout, pip_x, pip_y)
    return _motion(ref_a, ref_b), _motion(gen_a, gen_b)


def _capture(cap: cv2.VideoCapture, t: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, image = cap.read()
    if not ok or image is None:
        raise RuntimeError(f"Cannot decode semantic verification frame at {t:.3f}s")
    return image


def semantic_intervals(
    path: Path,
    start: float,
    end: float,
    settings: object,
    classify: Callable,
    verify: Callable[[list[np.ndarray], str], tuple[bool, str]],
) -> tuple[list[tuple[float, float]], list[dict]]:
    """Return contiguous verified spans and trace.

    Scan at semantic_step; bisect transitions with both motion and VLM probes.
    Do not bridge rejected time cells. Detection inside a cell has an intrinsic
    uncertainty bounded by the probe spacing; the bisection localizes observable
    transitions but is not a ground-truth frame-perfect annotation.
    """
    step = float(settings.semantic_step)
    tol = float(settings.boundary_tolerance)
    count = max(1, int(math.ceil((end - start) / step)))
    centers = [start + (i + .5) * (end - start) / count for i in range(count)]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for semantic analysis: {path}")
    probes: dict[float, ContentProbe] = {}

    def inspect(t: float, phase: str) -> ContentProbe:
        key = round(t, 6)
        if key in probes:
            return probes[key]
        half = min(float(settings.motion_delta), (end - start) / 4)
        left = max(start, t - half)
        right = min(end, t + half)
        if right - left < .09:
            result = ContentProbe(round(t, 4), False, "insufficient_temporal_context", "none", 0, 0, phase)
        else:
            first = _capture(cap, left)
            middle = _capture(cap, t)
            last = _capture(cap, right)
            layout, _, _ = classify(middle, settings)
            if layout == "none":
                result = ContentProbe(round(t, 4), False, "no_comparison_layout", "none", 0, 0, phase)
            else:
                rm, gm = motion_pair(first, last, layout, settings.pip_x, settings.pip_y)
                if min(rm, gm) < settings.motion_min:
                    result = ContentProbe(round(t, 4), False, "missing_dual_video_motion",
                                          layout, round(rm, 4), round(gm, 4), phase)
                else:
                    approved, reason = verify([first, middle, last], settings.ollama_model)
                    result = ContentProbe(round(t, 4), bool(approved),
                                          reason if reason else ("matched" if approved else "semantic_rejected"),
                                          layout, round(rm, 4), round(gm, 4), phase)
        probes[key] = result
        return result

    def edge(a: ContentProbe, b: ContentProbe) -> float:
        """Find the acceptance transition bracketed by unlike samples."""
        lo, hi = a, b
        while hi.time - lo.time > tol:
            mid = inspect((lo.time + hi.time) / 2, "boundary")
            if mid.match == lo.match:
                lo = mid
            else:
                hi = mid
        return round((lo.time + hi.time) / 2, 3)

    try:
        coarse = [inspect(t, "scan") for t in centers]
        spans: list[tuple[float, float]] = []
        i = 0
        while i < len(coarse):
            if not coarse[i].match:
                i += 1
                continue
            j = i
            while j + 1 < len(coarse) and coarse[j+1].match:
                j += 1
            begin = start if i == 0 else edge(coarse[i-1], coarse[i])
            finish = end if j == len(coarse) - 1 else edge(coarse[j], coarse[j+1])
            if finish - begin >= settings.min_duration:
                spans.append((round(begin, 3), round(finish, 3)))
            i = j+1
        return spans, [asdict(p) for _, p in sorted(probes.items())]
    finally:
        cap.release()
