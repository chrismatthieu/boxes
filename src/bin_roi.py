"""Red plastic bin opening, floor distance, and leftover-content mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV hue is 0-180. Red wraps around both ends.
_HUE_LOW = 10
_HUE_HIGH = 170
_SAT_MIN = 80
_VAL_MIN = 40

# Largest red region must cover this much of the frame or the bin is missing.
_MIN_BIN_FRAC = 0.08
# Pull the interior off the near rim so the wall is not counted as contents.
_RIM_ERODE_FRAC = 0.03
_MIN_DEPTH_M = 0.15
_MAX_DEPTH_M = 3.50
# Specks smaller than this fraction of the interior are ignored.
_MIN_BLOB_FRAC = 0.02
_MIN_BLOB_PX = 2000
# Hue this far from red, with real color, is a package rather than the bin or a scuff.
_PACKAGE_HUE_DIST = 10
_PACKAGE_SAT_MIN = 90
# Shadowed red plastic: same hue as the bin, but darker and duller.
_PLASTIC_SAT_MIN = 30
_PLASTIC_VAL_MIN = 15
# Deep shadow is too dark to be a package the color check can trust.
_MIN_BLOB_VALUE = 45


@dataclass
class BinView:
    found: bool
    opening: np.ndarray
    interior: np.ndarray
    red: np.ndarray
    content: np.ndarray
    blobs: list[np.ndarray]
    outline: np.ndarray
    floor_z_m: float | None
    height: np.ndarray
    plastic: np.ndarray


def _empty_view(h: int, w: int) -> BinView:
    def blank() -> np.ndarray:
        return np.zeros((h, w), dtype=bool)

    return BinView(
        found=False,
        opening=blank(),
        interior=blank(),
        red=blank(),
        content=blank(),
        blobs=[],
        outline=np.zeros((0, 2), dtype=np.int32),
        floor_z_m=None,
        height=np.full((h, w), np.nan, dtype=np.float32),
        plastic=blank(),
    )


def _hue_mask(hsv: np.ndarray, sat_min: int, val_min: int) -> np.ndarray:
    low = cv2.inRange(hsv, (0, sat_min, val_min), (_HUE_LOW, 255, 255))
    high = cv2.inRange(hsv, (_HUE_HIGH, sat_min, val_min), (180, 255, 255))
    return cv2.bitwise_or(low, high) > 0


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels that look like red plastic, before gap-filling."""
    return _hue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), _SAT_MIN, _VAL_MIN)


def plastic_mask(bgr: np.ndarray) -> np.ndarray:
    """Red plastic including shadowed and dull areas of the same bin."""
    return _hue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), _PLASTIC_SAT_MIN, _PLASTIC_VAL_MIN)


def _clean_red(raw: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    u8 = raw.astype(np.uint8) * 255
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened > 0


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    # Label 0 is background. Pick the biggest remaining region.
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return labels == idx


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    blobs: list[tuple[int, np.ndarray]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blobs.append((area, labels == i))
    blobs.sort(key=lambda item: item[0], reverse=True)
    return [blob for _, blob in blobs]


def _quadratic_height(
    z: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Meters above a curved floor. The fit uses the bin, not objects on it."""
    h, w = z.shape
    height = np.full((h, w), np.nan, dtype=np.float32)
    ys, xs = np.nonzero(fit_mask)
    if ys.size < 30:
        return height, None
    step = max(1, int(ys.size // 2500))
    ys_s = ys[::step]
    xs_s = xs[::step]
    zz = z[ys_s, xs_s].astype(np.float64)
    x = xs_s.astype(np.float64) / w
    y = ys_s.astype(np.float64) / h
    design = np.column_stack((np.ones_like(x), x, y, x * x, y * y, x * y))
    coef, *_ = np.linalg.lstsq(design, zz, rcond=None)
    for _ in range(3):
        # Positive residual: this pixel is closer than the floor (an item).
        resid = (design @ coef) - zz
        keep = resid < 0.02
        if int(keep.sum()) < 30 or int(keep.sum()) == len(zz):
            break
        design = design[keep]
        zz = zz[keep]
        coef, *_ = np.linalg.lstsq(design, zz, rcond=None)

    yy, xx = np.indices((h, w))
    xn = xx.astype(np.float64) / w
    yn = yy.astype(np.float64) / h
    floor = coef[0] + coef[1] * xn + coef[2] * yn + coef[3] * xn * xn + coef[4] * yn * yn + coef[5] * xn * yn
    height[eval_mask] = (floor[eval_mask] - z[eval_mask]).astype(np.float32)
    return height, float(np.median(z[fit_mask]))


def mask_lift(view: BinView, mask: np.ndarray) -> float | None:
    """Median meters above the local floor, or None when depth is missing."""
    if view.height.shape != mask.shape:
        return None
    vals = view.height[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 30:
        return None
    return float(np.median(vals))


def _package_colors(hsv: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Cardboard, kraft, and pale bags. Dull scuffs and red plastic stay out."""
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue_dist = np.minimum(hue, 180 - hue)
    chromatic = (hue_dist >= _PACKAGE_HUE_DIST) & (sat >= _PACKAGE_SAT_MIN) & (val >= 50)
    bright = (sat <= 30) & (val >= 180)
    return interior & (chromatic | bright)


def find_bin(color: np.ndarray, depth: np.ndarray, depth_scale: float) -> BinView:
    """Locate the bin opening and pixels that are packages on its floor.

    A bowed bottom, shadows, and dull scuffs stay empty. A box or jiffy counts
    even when it lies flat and fills most of the opening.
    """
    h, w = color.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    raw = red_mask(color)
    cleaned = _clean_red(raw)
    component = _largest_component(cleaned)
    frame_area = h * w
    if component is None or int(component.sum()) < _MIN_BIN_FRAC * frame_area:
        view = _empty_view(h, w)
        view.red = raw
        return view

    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        view = _empty_view(h, w)
        view.red = raw
        return view
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    outline = hull.reshape(-1, 2).astype(np.int32)
    opening_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(opening_u8, outline, 1)
    if int(opening_u8.sum()) < _MIN_BIN_FRAC * frame_area:
        view = _empty_view(h, w)
        view.red = raw
        return view

    k = max(3, int(round(_RIM_ERODE_FRAC * min(h, w))))
    if k % 2 == 0:
        k += 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    interior_u8 = cv2.erode(opening_u8, erode_kernel)
    if int(interior_u8.sum()) < 0.5 * int(opening_u8.sum()):
        # Rim erosion ate a thin or partial bin. Keep a lighter inset.
        k = max(3, k // 2)
        if k % 2 == 0:
            k += 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        interior_u8 = cv2.erode(opening_u8, erode_kernel)
    if int(interior_u8.sum()) < 500:
        view = _empty_view(h, w)
        view.red = raw
        return view

    opening = opening_u8 > 0
    interior = interior_u8 > 0
    z = depth.astype(np.float32) * float(depth_scale)
    valid_z = (z >= _MIN_DEPTH_M) & (z <= _MAX_DEPTH_M)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    plastic = _hue_mask(hsv, _PLASTIC_SAT_MIN, _PLASTIC_VAL_MIN)
    # Fit the bowl to the plastic only, so a box or mailer is not absorbed into the floor.
    floor_support = interior & valid_z & plastic
    if int(floor_support.sum()) < 30:
        floor_support = interior & valid_z
    height, floor_z = _quadratic_height(z, floor_support, interior & valid_z)

    # Color only. A bowed or dirty floor stays near red and must not become an item.
    # Raised non-plastic patches were that false alarm; packages are named later.
    colors = _package_colors(hsv, interior)
    candidate = colors
    speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    content_u8 = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, speck)
    min_blob = max(_MIN_BLOB_PX, int(_MIN_BLOB_FRAC * float(interior.sum())))
    blobs = []
    for blob in _components(content_u8 > 0, min_blob):
        if float(np.median(hsv[:, :, 2][blob])) < _MIN_BLOB_VALUE:
            continue
        blobs.append(blob)
    if blobs:
        content = np.zeros((h, w), dtype=bool)
        for blob in blobs:
            content |= blob
    else:
        content = np.zeros((h, w), dtype=bool)

    return BinView(
        found=True,
        opening=opening,
        interior=interior,
        red=raw,
        content=content,
        blobs=blobs,
        outline=outline,
        floor_z_m=floor_z,
        height=height,
        plastic=plastic,
    )
