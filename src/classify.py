"""Geometry-based class cleanup after depth measurement."""

from __future__ import annotations

from .types import CLASS_BOX, CLASS_JIFFY, CLASS_PADDED_BAG, Package

# Kraft / generic bag prompts that YOLOE often assigns to brown jiffy mailers.
_BAG_PROMPTS_THAT_LOOK_LIKE_ENVELOPES = frozenset(
    {
        "padded bag",
        "gift bag",
        "bubble mailer",
    }
)


def _sorted_dims(pkg: Package) -> tuple[float, float, float] | None:
    if pkg.length_cm is None or pkg.width_cm is None or pkg.height_cm is None:
        return None
    shortest, mid, longest = sorted((pkg.length_cm, pkg.width_cm, pkg.height_cm))
    return shortest, mid, longest


def _is_cubic(dims: tuple[float, float, float]) -> bool:
    """Small shipping carton: all three sides substantial and similar."""
    shortest, _mid, longest = dims
    return shortest >= 5.0 and (shortest / max(longest, 1.0)) >= 0.40


def _is_mailer(dims: tuple[float, float, float]) -> bool:
    """Envelope / jiffy: thin relative to a letter-sized face."""
    shortest, mid, longest = dims
    return (
        shortest <= 3.2
        and (shortest / max(longest, 1.0)) <= 0.18
        and 12.0 <= longest <= 42.0
        and mid >= 8.0
    )


def refine_package(pkg: Package) -> Package | None:
    """Keep envelopes as jiffy unless the shape is a real carton."""
    dims = _sorted_dims(pkg)

    if pkg.class_id == CLASS_JIFFY:
        if dims is not None and _is_cubic(dims):
            pkg.class_id = CLASS_BOX
        return pkg

    if pkg.class_id == CLASS_PADDED_BAG:
        if (
            dims is not None
            and _is_mailer(dims)
            and pkg.prompt in _BAG_PROMPTS_THAT_LOOK_LIKE_ENVELOPES
        ):
            pkg.class_id = CLASS_JIFFY
        return pkg

    return pkg
