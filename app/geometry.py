"""Segment geometry primitives.

All production coordinates are integers, but the closest-point parameters
of two segments are rational.  The computation therefore uses exact
``fractions.Fraction`` arithmetic throughout.  Converting the inputs to
``float`` beforehand is not safe: at a coordinate offset of ``1e17`` a
float already carries a ULP of about 16, so genuinely intersecting
segments can be reported as a double-digit gap apart.

``segment_distance`` returns ``(distance, point_a, point_b)`` where
``point_a`` lies on the first segment and ``point_b`` on the second.  The
distance and evidence points are floats (the distance is a square root
anyway), but a true zero-distance result is always exactly ``0.0``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Tuple

Point = Tuple[float, float]
Seg = Tuple[Point, Point]

F = Fraction
FPoint = Tuple[F, F]

_ZERO = F(0)
_ONE = F(1)


def _fpoint(p: Point) -> FPoint:
    return (F(p[0]), F(p[1]))


def _finish(p: FPoint, q: FPoint) -> Tuple[float, Point, Point]:
    dx, dy = p[0] - q[0], p[1] - q[1]
    d2 = dx * dx + dy * dy
    dist = 0.0 if d2 == 0 else math.sqrt(float(d2))
    return dist, (float(p[0]), float(p[1])), (float(q[0]), float(q[1]))


def _clamp01(x: F) -> F:
    if x < _ZERO:
        return _ZERO
    if x > _ONE:
        return _ONE
    return x


def _point_segment(
    p: FPoint, a: FPoint, b: FPoint
) -> Tuple[float, Point, Point]:
    """Distance from point ``p`` to the closed segment ``a -> b``.

    Returns ``(distance, p, q)`` with ``q`` the closest point on the
    segment.  Degenerate (zero-length) segments are handled.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return _finish(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    t = _clamp01(t)
    q = (a[0] + t * dx, a[1] + t * dy)
    return _finish(p, q)


def segment_distance(s1: Seg, s2: Seg) -> Tuple[float, Point, Point]:
    """Minimum Euclidean distance between two closed segments.

    Shared endpoints (or crossing/overlapping segments) yield ``0``.
    Returns the distance together with the pair of closest points.
    Uses the standard clamped six-region projection with exact rational
    intermediate values.
    """
    a1, a2 = _fpoint(s1[0]), _fpoint(s1[1])
    b1, b2 = _fpoint(s2[0]), _fpoint(s2[1])

    ux, uy = a2[0] - a1[0], a2[1] - a1[1]
    vx, vy = b2[0] - b1[0], b2[1] - b1[1]
    wx, wy = a1[0] - b1[0], a1[1] - b1[1]

    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    e = ux * wx + uy * wy
    f = vx * wx + vy * wy

    if a == 0 and c == 0:
        return _finish(a1, b1)
    if a == 0:
        return _point_segment(a1, b1, b2)
    if c == 0:
        d, q, p = _point_segment(b1, a1, a2)
        return d, p, q

    det = a * c - b * b
    if det != 0:
        s = (b * f - c * e) / det
        t = (a * f - b * e) / det
    else:  # parallel segments: pin s=0 and let the t-clamp move s
        s, t = _ZERO, f / c

    if s < _ZERO:
        s, t = _ZERO, f / c
    elif s > _ONE:
        s, t = _ONE, (f + b) / c

    if t < _ZERO:
        t = _ZERO
        if -e < 0:
            s = _ZERO
        elif -e > a:
            s = _ONE
        else:
            s = -e / a
    elif t > _ONE:
        t = _ONE
        if -e + b < 0:
            s = _ZERO
        elif -e + b > a:
            s = _ONE
        else:
            s = (-e + b) / a

    s = _clamp01(s)
    t = _clamp01(t)

    p = (a1[0] + s * ux, a1[1] + s * uy)
    q = (b1[0] + t * vx, b1[1] + t * vy)
    return _finish(p, q)
