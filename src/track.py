"""Per-class unique session counting via IoU tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import CLASS_BOX, CLASS_IDS, CLASS_JIFFY, CLASS_PADDED_BAG, Package


def _iou(a: np.ndarray, b: np.ndarray) -> float:
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


@dataclass
class Track:
    track_id: int
    class_id: str
    bbox: np.ndarray
    hits: int = 0
    age: int = 0
    missed: int = 0
    counted: bool = False
    last_pkg: Package | None = field(default=None, repr=False)


class SessionTracker:
    def __init__(self, confirm_hits: int = 8, max_missed: int = 20, iou_thresh: float = 0.30) -> None:
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self._next_id = 1
        self.tracks: list[Track] = []
        self.counts = {
            CLASS_BOX: 0,
            CLASS_PADDED_BAG: 0,
            CLASS_JIFFY: 0,
        }

    @property
    def total(self) -> int:
        return sum(self.counts[c] for c in CLASS_IDS)

    def reset(self) -> None:
        self._next_id = 1
        self.tracks = []
        for key in self.counts:
            self.counts[key] = 0

    def update(self, packages: list[Package]) -> list[Package]:
        for tr in self.tracks:
            tr.age += 1
            tr.missed += 1

        unmatched = list(range(len(packages)))
        used_tracks: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for pi in unmatched:
                pkg = packages[pi]
                if pkg.class_id != tr.class_id:
                    continue
                pairs.append((_iou(tr.bbox, pkg.bbox), ti, pi))
        pairs.sort(reverse=True)

        assigned_pkg: set[int] = set()
        for iou, ti, pi in pairs:
            if iou < self.iou_thresh:
                break
            if ti in used_tracks or pi in assigned_pkg:
                continue
            pkg = packages[pi]
            tr = self.tracks[ti]
            tr.bbox = pkg.bbox.copy()
            tr.hits += 1
            tr.missed = 0
            tr.last_pkg = pkg
            pkg.track_id = tr.track_id
            if not tr.counted and tr.hits >= self.confirm_hits:
                self.counts[tr.class_id] += 1
                tr.counted = True
            used_tracks.add(ti)
            assigned_pkg.add(pi)

        for pi, pkg in enumerate(packages):
            if pi in assigned_pkg:
                continue
            tr = Track(
                track_id=self._next_id,
                class_id=pkg.class_id,
                bbox=pkg.bbox.copy(),
                hits=1,
                last_pkg=pkg,
            )
            self._next_id += 1
            pkg.track_id = tr.track_id
            self.tracks.append(tr)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return packages
