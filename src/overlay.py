"""Fullscreen overlay: class-colored masks, dimensions, stacked warning, HUD."""

from __future__ import annotations

import cv2
import numpy as np

from .types import CLASS_COLORS, CLASS_DISPLAY, Package
from .track import SessionTracker


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw(
    bgr: np.ndarray,
    packages: list[Package],
    tracker: SessionTracker,
    fps: float,
    stacked_warning: bool,
) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()
    for pkg in packages:
        color = CLASS_COLORS.get(pkg.class_id, (200, 200, 200))
        overlay[pkg.mask] = (0.45 * overlay[pkg.mask] + 0.55 * np.array(color, dtype=np.float32)).astype(
            np.uint8
        )
        x1, y1, x2, y2 = pkg.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    for pkg in packages:
        color = CLASS_COLORS.get(pkg.class_id, (200, 200, 200))
        x1, y1, x2, y2 = pkg.bbox.astype(int)
        name = CLASS_DISPLAY.get(pkg.class_id, pkg.class_id.upper())
        tid = f"#{pkg.track_id}" if pkg.track_id is not None else ""
        lines = [f"{name} {tid}".strip()]
        if pkg.length_cm is not None:
            lines.append(
                f"{pkg.length_cm:.0f}x{pkg.width_cm:.0f}x{pkg.height_cm:.0f} cm  {pkg.volume_l:.2f} L"
            )
        if pkg.stacked:
            lines.append("STACKED")
        y = max(18, y1 - 8 - 18 * (len(lines) - 1))
        for line in lines:
            col = (40, 40, 255) if line == "STACKED" else (255, 255, 255)
            _draw_text(vis, line, (x1, y), scale=0.5, color=col, thickness=1)
            y += 18

    h, w = vis.shape[:2]
    hud_h = 78 if stacked_warning else 52
    cv2.rectangle(vis, (0, 0), (w, hud_h), (20, 20, 20), -1)
    counts = tracker.counts
    hud = (
        f"Boxes: {counts['box']}    Padded bags: {counts['padded_bag']}    "
        f"Jiffy envelopes: {counts['jiffy_envelope']}    Total: {tracker.total}"
    )
    _draw_text(vis, hud, (12, 28), scale=0.65, color=(240, 240, 240), thickness=2)
    live = sum(1 for p in packages)
    stacked_n = sum(1 for p in packages if p.stacked)
    status = f"Live: {live}    Stacked now: {stacked_n}    {fps:.1f} FPS    [q] quit  [r] reset  [p] pause"
    _draw_text(vis, status, (12, 50), scale=0.48, color=(200, 200, 200), thickness=1)
    if stacked_warning:
        _draw_text(
            vis,
            "WARNING: stacked packages detected - dimensions may be approximate",
            (12, 72),
            scale=0.5,
            color=(40, 40, 255),
            thickness=2,
        )
    return vis
