"""Shared package types and class IDs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASS_BOX = "box"
CLASS_PADDED_BAG = "padded_bag"
CLASS_JIFFY = "jiffy_envelope"

CLASS_IDS = (CLASS_BOX, CLASS_PADDED_BAG, CLASS_JIFFY)

CLASS_DISPLAY = {
    CLASS_BOX: "BOX",
    CLASS_PADDED_BAG: "BAG",
    CLASS_JIFFY: "JIFFY",
}

# OpenCV BGR: blue boxes, green bags, orange envelopes
CLASS_COLORS = {
    CLASS_BOX: (220, 120, 40),
    CLASS_PADDED_BAG: (60, 190, 70),
    CLASS_JIFFY: (40, 145, 255),
}

PROMPT_TO_CLASS = {
    "cardboard box": CLASS_BOX,
    "shipping box": CLASS_BOX,
    "corrugated box": CLASS_BOX,
    "cardboard package": CLASS_BOX,
    "crate": CLASS_BOX,
    "shopping bag": CLASS_PADDED_BAG,
    "paper shopping bag": CLASS_PADDED_BAG,
    "grocery bag": CLASS_PADDED_BAG,
    "gift bag": CLASS_PADDED_BAG,
    "plastic tote": CLASS_PADDED_BAG,
    "padded bag": CLASS_PADDED_BAG,
    "bubble mailer": CLASS_PADDED_BAG,
    "poly mailer": CLASS_PADDED_BAG,
    "jiffy envelope": CLASS_JIFFY,
    "jiffy bag": CLASS_JIFFY,
    "jiffy mailer": CLASS_JIFFY,
    "kraft mailer": CLASS_JIFFY,
    "kraft padded envelope": CLASS_JIFFY,
    "padded envelope": CLASS_JIFFY,
    "bubble envelope": CLASS_JIFFY,
    "paper envelope": CLASS_JIFFY,
    "manila envelope": CLASS_JIFFY,
    "shipping envelope": CLASS_JIFFY,
}

# Plastic mailers that also fire on cardboard; shopping bags and envelopes use --conf.
MAILER_PROMPTS = frozenset(
    {
        "padded bag",
        "bubble mailer",
        "poly mailer",
    }
)

PROMPTS = list(PROMPT_TO_CLASS.keys())


@dataclass
class Package:
    class_id: str
    prompt: str
    conf: float
    bbox: np.ndarray
    mask: np.ndarray
    track_id: int | None = None
    length_cm: float | None = None
    width_cm: float | None = None
    height_cm: float | None = None
    volume_l: float | None = None
    stacked: bool = False
    z_min: float = 0.0
    z_max: float = 0.0
    z_med: float = 0.0
    centroid_xyz: np.ndarray | None = None
    points_xyz: np.ndarray | None = field(default=None, repr=False)
