"""Overlay for the red-bin empty check: outline, item masks, remaining count."""

from __future__ import annotations

import cv2
import numpy as np

from .bin_roi import BinView
from .types import CLASS_BOX, CLASS_COLORS, CLASS_DISPLAY, CLASS_JIFFY, CLASS_PADDED_BAG, Package

UNKNOWN = "unknown"

_COLORS = {
    **CLASS_COLORS,
    UNKNOWN: (170, 170, 170),
}
_LABELS = {
    **CLASS_DISPLAY,
    UNKNOWN: "ITEM",
}


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
        cv2.LINE_8,
    )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_8)


def _counts(items: list[Package]) -> dict[str, int]:
    counts = {CLASS_BOX: 0, CLASS_PADDED_BAG: 0, CLASS_JIFFY: 0, UNKNOWN: 0}
    for item in items:
        key = item.class_id if item.class_id in counts else UNKNOWN
        counts[key] += 1
    return counts


def draw(
    bgr: np.ndarray,
    view: BinView,
    items: list[Package],
    fps: float,
    extra: str = "",
) -> np.ndarray:
    vis = bgr.copy()
    if view.found and view.outline.size:
        cv2.polylines(vis, [view.outline.reshape(-1, 1, 2)], True, (255, 220, 40), 2)

    for item in items:
        color = _COLORS.get(item.class_id, (200, 200, 200))
        if item.mask is not None and item.mask.any():
            roi = vis[item.mask]
            vis[item.mask] = (
                roi.astype(np.float32) * 0.55 + np.array(color, dtype=np.float32) * 0.45
            ).astype(np.uint8)
        x1, y1, x2, y2 = item.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = _LABELS.get(item.class_id, item.class_id.upper())
        _draw_text(vis, name, (x1, max(18, y1 - 8)), scale=0.5, color=(255, 255, 255), thickness=1)

    if view.found and view.outline.size:
        cv2.polylines(vis, [view.outline.reshape(-1, 1, 2)], True, (255, 220, 40), 2)

    h, w = vis.shape[:2]
    hud_h = 96
    cv2.rectangle(vis, (0, 0), (w, hud_h), (20, 20, 20), -1)
    n = len(items)
    if not view.found:
        title = "BIN NOT FOUND"
        title_color = (0, 200, 255)
        detail = "Aim the camera into the red bin"
    elif n == 0:
        title = "EMPTY"
        title_color = (80, 220, 90)
        detail = "Boxes: 0    Bags: 0    Jiffy: 0"
    else:
        title = "1 ITEM REMAINING" if n == 1 else f"{n} ITEMS REMAINING"
        title_color = (50, 60, 255)
        counts = _counts(items)
        detail = (
            f"Boxes: {counts[CLASS_BOX]}    Bags: {counts[CLASS_PADDED_BAG]}    "
            f"Jiffy: {counts[CLASS_JIFFY]}"
        )
        if counts[UNKNOWN]:
            detail += f"    Unclassified: {counts[UNKNOWN]}"
    _draw_text(vis, title, (12, 36), scale=1.0, color=title_color, thickness=2)
    _draw_text(vis, detail, (12, 64), scale=0.55, color=(230, 230, 230), thickness=1)
    status = f"{fps:.1f} FPS{('    ' + extra) if extra else ''}    [q] quit  [p] pause"
    _draw_text(vis, status, (12, 86), scale=0.48, color=(190, 190, 190), thickness=1)
    return vis
