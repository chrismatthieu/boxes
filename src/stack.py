"""Split stacked layers inside a mask and flag stacked packages."""

from __future__ import annotations

import numpy as np

from .types import Package


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the smaller box covered by intersection."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, float(a[2] - a[0]) * float(a[3] - a[1]))
    area_b = max(1.0, float(b[2] - b[0]) * float(b[3] - b[1]))
    return inter / min(area_a, area_b)


def _z_gap(a: Package, b: Package) -> float:
    if a.z_max <= 0 or b.z_max <= 0:
        return 1e9
    if a.z_max < b.z_min:
        return b.z_min - a.z_max
    if b.z_max < a.z_min:
        return a.z_min - b.z_max
    return 0.0


def split_layers(pkg: Package, depth: np.ndarray, depth_scale: float) -> list[Package]:
    """If one mask covers two depth layers, split into two instances."""
    if pkg.points_xyz is None or len(pkg.points_xyz) < 250:
        return [pkg]
    z = pkg.points_xyz[:, 2]
    z_span = float(z.max() - z.min())
    if z_span < 0.12:
        return [pkg]
    hist, edges = np.histogram(z, bins=20)
    if hist.max() < 15:
        return [pkg]
    peak_idxs = np.argsort(hist)[::-1]
    peaks = []
    for idx in peak_idxs:
        if hist[idx] < max(12, hist.max() * 0.18):
            continue
        if all(abs(idx - p) >= 3 for p in peaks):
            peaks.append(int(idx))
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        return [pkg]
    p0, p1 = sorted(peaks)
    valley = int(p0 + np.argmin(hist[p0 : p1 + 1]))
    split_z = float(0.5 * (edges[valley] + edges[valley + 1]))
    if min(hist[p0], hist[p1]) < 12:
        return [pkg]

    h, w = pkg.mask.shape
    ys, xs = np.nonzero(pkg.mask)
    z_img = depth[ys, xs].astype(np.float32) * depth_scale
    near = np.zeros((h, w), dtype=bool)
    far = np.zeros((h, w), dtype=bool)
    near[ys, xs] = z_img <= split_z
    far[ys, xs] = z_img > split_z
    if near.sum() < 250 or far.sum() < 250:
        return [pkg]
    smaller, larger = sorted((int(near.sum()), int(far.sum())))
    if smaller / max(larger, 1) < 0.35:
        return [pkg]

    split: list[Package] = []
    for mask in (near, far):
        ys_m, xs_m = np.nonzero(mask)
        bbox = np.array(
            [xs_m.min(), ys_m.min(), xs_m.max() + 1, ys_m.max() + 1], dtype=np.float32
        )
        child = Package(
            class_id=pkg.class_id,
            prompt=pkg.prompt,
            conf=pkg.conf,
            bbox=bbox,
            mask=mask,
        )
        split.append(child)
    return split


def mark_stacked(packages: list[Package], iou_thresh: float = 0.18, max_gap_m: float = 0.06) -> bool:
    any_stacked = False
    n = len(packages)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = packages[i], packages[j]
            contained = _containment(a.bbox, b.bbox)
            # A shipping label (or class double) sits inside the parent, not on top of it.
            if contained >= 0.60:
                continue
            area_a = max(1.0, float(a.bbox[2] - a.bbox[0]) * float(a.bbox[3] - a.bbox[1]))
            area_b = max(1.0, float(b.bbox[2] - b.bbox[0]) * float(b.bbox[3] - b.bbox[1]))
            if min(area_a, area_b) / max(area_a, area_b) < 0.40:
                continue
            iou = _bbox_iou(a.bbox, b.bbox)
            gap = _z_gap(a, b)
            close_in_z = gap <= max_gap_m
            overlap = iou >= iou_thresh
            ax1, ay1, ax2, ay2 = a.bbox
            bx1, by1, bx2, by2 = b.bbox
            x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) / max(1.0, min(ax2 - ax1, bx2 - bx1))
            y_gap = min(abs(ay2 - by1), abs(by2 - ay1))
            height = max(ay2 - ay1, by2 - by1, 1.0)
            side_stack = x_overlap > 0.55 and y_gap < 0.15 * height and abs(a.z_med - b.z_med) < 0.18
            if (overlap and close_in_z) or side_stack:
                a.stacked = True
                b.stacked = True
                any_stacked = True
    return any_stacked
