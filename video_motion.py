"""CPU optical-flow evidence that both video panels move over multiple frames.

Inspired by RAFT/VideoMAE/CoTracker research; no pretrained weights are used.
Residual optical flow rejects uniform page scroll/panning, but cannot prove
semantic correspondence. The VLM still makes the semantic decision.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PanelMotion:
    active_pairs: int
    total_pairs: int
    changed_fraction: float
    residual_fraction: float
    global_flow_fraction: float
    accepted: bool


def _panels(frame: np.ndarray, layout: str, pip_x: float, pip_y: float
            ) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]

    def crop(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        part = image[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
        if min(part.shape[:2]) < 8:
            raise ValueError(f"Invalid comparison region for {layout}: {part.shape}")
        return cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)

    if layout == "pip":
        return (crop(min(.93, pip_x + .05), .05, .973, max(.12, pip_y - .04)),
                crop(.08, max(.34, pip_y + .04), max(.35, pip_x - .07), .83))
    if layout == "split":
        return crop(.58, .19, .94, .79), crop(.06, .19, .42, .79)
    raise ValueError(f"Unsupported comparison layout: {layout}")


def _measure(images: list[np.ndarray], min_changed: float,
             min_residual: float, consistency: float) -> PanelMotion:
    changed: list[float] = []
    residual: list[float] = []
    global_flow: list[float] = []
    active = 0
    for prev, current in zip(images, images[1:]):
        diff_ratio = float(np.mean(cv2.absdiff(prev, current) > 14))
        flow = cv2.calcOpticalFlowFarneback(
            prev, current, None, .5, 2, 11, 2, 5, 1.1, 0)
        # Subtract median 2D motion to reject globally translated page/UI.
        translation = np.median(flow.reshape(-1, 2), axis=0)
        global_ratio = float(np.mean(np.linalg.norm(flow, axis=2) > .8))
        residual_ratio = float(np.mean(np.linalg.norm(flow - translation, axis=2) > .7))
        changed.append(diff_ratio)
        residual.append(residual_ratio)
        global_flow.append(global_ratio)
        if diff_ratio >= min_changed and residual_ratio >= min_residual:
            active += 1
    total = len(images) - 1
    return PanelMotion(
        active_pairs=active, total_pairs=total,
        changed_fraction=round(float(np.mean(changed)), 4),
        residual_fraction=round(float(np.mean(residual)), 4),
        global_flow_fraction=round(float(np.mean(global_flow)), 4),
        accepted=active >= max(2, math.ceil(total * consistency)),
    )


def temporal_motion(frames: list[np.ndarray], layout: str, pip_x: float,
                    pip_y: float, min_changed: float = .02,
                    min_residual: float = .012,
                    consistency: float = .5) -> tuple[float, float, bool, dict]:
    """Check sustained local motion in both panels across five frames.

    Returns (reference_changed_fraction, rendered_changed_fraction,
    both_pass, details). Fails closed for static reference and global scroll.
    Very slow model motion can be missed: tune with independently labeled data.
    """
    if len(frames) != 5:
        raise ValueError("Expected exactly five consecutive frames")
    if not (0 <= min_changed <= .5 and 0 <= min_residual <= .5
            and .25 <= consistency <= 1):
        raise ValueError("Invalid temporal motion thresholds")
    panels = [_panels(frame, layout, pip_x, pip_y) for frame in frames]
    reference = _measure([p[0] for p in panels], min_changed, min_residual, consistency)
    rendered = _measure([p[1] for p in panels], min_changed, min_residual, consistency)
    return (reference.changed_fraction, rendered.changed_fraction,
            reference.accepted and rendered.accepted,
            {"reference": asdict(reference), "rendered": asdict(rendered)})
