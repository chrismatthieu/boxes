"""Deproject instance masks and fit metric oriented bounding boxes."""

from __future__ import annotations

import numpy as np

from .types import Package

_MIN_POINTS = 40
_MIN_DEPTH_M = 0.20
_MAX_DEPTH_M = 3.50


def _deproject(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    depth_scale: float,
    stride: int = 3,
) -> np.ndarray:
    ys, xs = np.nonzero(mask[::stride, ::stride])
    ys = ys * stride
    xs = xs * stride
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32) * depth_scale
    valid = (z >= _MIN_DEPTH_M) & (z <= _MAX_DEPTH_M)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = (xs.astype(np.float32) - ppx) / fx * z
    y = (ys.astype(np.float32) - ppy) / fy * z
    return np.stack((x, y, z), axis=1)


def _trim_outliers(points: np.ndarray) -> np.ndarray:
    if len(points) < _MIN_POINTS:
        return points
    z = points[:, 2]
    lo, hi = np.percentile(z, [8.0, 92.0])
    keep = (z >= lo) & (z <= hi)
    trimmed = points[keep]
    return trimmed if len(trimmed) >= _MIN_POINTS else points


def pca_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (L,W,H) in meters sorted L>=W>=H, and centroid."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    projected = centered @ eigvecs
    extents = projected.max(axis=0) - projected.min(axis=0)
    extents = np.sort(np.abs(extents))[::-1]
    extents = np.maximum(extents, 1e-4)
    return extents.astype(np.float32), centroid.astype(np.float32)


def fit_support_plane(
    depth: np.ndarray,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    depth_scale: float,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """RANSAC plane (table/floor) from background depth. Returns (normal, d) with n·x + d = 0."""
    h, w = depth.shape
    stride = 10
    grid = np.zeros((h, w), dtype=bool)
    grid[::stride, ::stride] = True
    if ignore_mask is not None:
        grid &= ~ignore_mask
    ys, xs = np.nonzero(grid)
    z = depth[ys, xs].astype(np.float32) * depth_scale
    valid = (z >= _MIN_DEPTH_M) & (z <= _MAX_DEPTH_M)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if xs.size < 80:
        return None
    x = (xs.astype(np.float32) - ppx) / fx * z
    y = (ys.astype(np.float32) - ppy) / fy * z
    points = np.stack((x, y, z), axis=1)
    if len(points) > 1200:
        rng = np.random.default_rng(1)
        points = points[rng.choice(len(points), 1200, replace=False)]
    npts = len(points)
    rng = np.random.default_rng(2)
    best_count = 0
    best_n: np.ndarray | None = None
    best_d = 0.0
    thresh = 0.015
    for _ in range(28):
        idx = rng.choice(npts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = float(-normal.dot(p0))
        dist = np.abs(points @ normal + d)
        count = int((dist < thresh).sum())
        if count > best_count:
            best_count = count
            best_n, best_d = normal, d
    if best_n is None or best_count < npts * 0.25:
        return None
    return best_n.astype(np.float32), best_d


def _plane_aligned_extents(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """L, W in the plane and H along the plane normal, meters, sorted L>=W then H last."""
    n = normal / max(float(np.linalg.norm(normal)), 1e-8)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(n.dot(tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_u = np.cross(n, tmp)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-8)
    axis_v = np.cross(n, axis_u)
    u = points @ axis_u
    v = points @ axis_v
    h = points @ n
    length = float(u.max() - u.min())
    width = float(v.max() - v.min())
    height = float(h.max() - h.min())
    lw = sorted((abs(length), abs(width)), reverse=True)
    return np.array([lw[0], lw[1], max(abs(height), 1e-4)], dtype=np.float32)


def measure_package(
    pkg: Package,
    depth: np.ndarray,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    depth_scale: float,
    plane: tuple[np.ndarray, float] | None = None,
) -> Package:
    points = _deproject(depth, pkg.mask, fx, fy, ppx, ppy, depth_scale)
    points = _trim_outliers(points)
    if len(points) < _MIN_POINTS:
        return pkg
    pkg.points_xyz = points
    pkg.z_min = float(points[:, 2].min())
    pkg.z_max = float(points[:, 2].max())
    pkg.z_med = float(np.median(points[:, 2]))
    extents_m, centroid = pca_obb(points)
    pkg.centroid_xyz = centroid

    if plane is not None:
        normal, d = plane
        dist = np.abs(points @ normal + d)
        height_to_plane = float(np.median(dist))
        plane_ext = _plane_aligned_extents(points, normal)
        # Visible-face PCA often collapses height; use table distance for isolated objects.
        if extents_m[2] < 0.015 and height_to_plane > 0.015:
            extents_m = np.array(
                [plane_ext[0], plane_ext[1], height_to_plane], dtype=np.float32
            )
        else:
            extents_m = np.array(
                [max(extents_m[0], plane_ext[0]), max(extents_m[1], plane_ext[1]), max(extents_m[2], plane_ext[2])],
                dtype=np.float32,
            )
            extents_m = np.sort(extents_m)[::-1]

    pkg.length_cm = float(extents_m[0] * 100.0)
    pkg.width_cm = float(extents_m[1] * 100.0)
    pkg.height_cm = float(extents_m[2] * 100.0)
    pkg.volume_l = float(float(extents_m[0]) * float(extents_m[1]) * float(extents_m[2]) * 1000.0)
    return pkg


def measure_all(
    packages: list[Package],
    depth: np.ndarray,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    depth_scale: float,
) -> list[Package]:
    ignore = None
    if packages:
        ignore = np.zeros(depth.shape, dtype=bool)
        for pkg in packages:
            if pkg.mask.shape == depth.shape:
                ignore |= pkg.mask
    plane = fit_support_plane(depth, fx, fy, ppx, ppy, depth_scale, ignore)
    return [measure_package(p, depth, fx, fy, ppx, ppy, depth_scale, plane) for p in packages]
