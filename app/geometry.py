"""Segment geometry primitives.

All coordinates are integer-valued in the plane.  Distances are computed as
floating point values; ``segment_distance`` returns ``(distance, point_a,
point_b)`` where ``point_a`` lies on the first segment and ``point_b`` on the
second.
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]
Seg = Tuple[Point, Point]


def segment_distance(s1: Seg, s2: Seg) -> Tuple[float, Point, Point]:
    """Minimum Euclidean distance between two closed segments.

    Shared endpoints (or crossing/overlapping segments) yield ``0``.
    Returns the distance together with the pair of closest points.
    Uses the standard clamped six-region projection.
    """
    a1, a2 = s1
    b1, b2 = s2

    ux, uy = a2[0] - a1[0], a2[1] - a1[1]
    vx, vy = b2[0] - b1[0], b2[1] - b1[1]
    wx, wy = a1[0] - b1[0], a1[1] - b1[1]

    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    e = ux * wx + uy * wy
    f = vx * wx + vy * wy

    if a == 0.0 and c == 0.0:
        return math.dist(a1, b1), a1, b1
    if a == 0.0:
        return _point_segment_dist(a1, s2)
    if c == 0.0:
        d, q, p = _point_segment_dist(b1, s1)
        return d, p, q

    det = a * c - b * b
    if det != 0.0:
        s = (b * f - c * e) / det
        t = (a * f - b * e) / det
    else:  # parallel segments: pin s=0 and let the t-clamp move s
        s, t = 0.0, f / c

    if s < 0.0:
        s, t = 0.0, f / c
    elif s > 1.0:
        s, t = 1.0, (f + b) / c

    if t < 0.0:
        t = 0.0
        if -e < 0.0:
            s = 0.0
        elif -e > a:
            s = 1.0
        else:
            s = -e / a
    elif t > 1.0:
        t = 1.0
        if -e + b < 0.0:
            s = 0.0
        elif -e + b > a:
            s = 1.0
        else:
            s = (-e + b) / a

    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))

    p = (a1[0] + s * ux, a1[1] + s * uy)
    q = (b1[0] + t * vx, b1[1] + t * vy)
    return math.dist(p, q), p, q


def _point_segment_dist(p: Point, seg: Seg) -> Tuple[float, Point, Point]:
    a, b = seg
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.dist(p, a), p, a
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * dx, a[1] + t * dy)
    return math.dist(p, q), p, q
