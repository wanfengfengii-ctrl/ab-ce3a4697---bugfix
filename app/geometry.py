"""Segment geometry primitives.

All request coordinates are integers, and the clearance comparisons are
performed with *exact* integer/rational arithmetic: converting coordinates
near or above ``2**53`` to ``float`` would round endpoints by several units
(the float64 spacing at ``10**17`` is 16), which can turn two strictly
crossing segments into an apparent 16-unit gap.  The exact entry points are:

- :func:`segments_clearance_ok` -- integral clearance predicate;
- :func:`segment_distance_sq_exact` -- squared distance as a reduced-shape
  rational ``num / den`` together with the rational closest points;
- :func:`segment_distance` -- same result rendered as floats (distance and
  closest points), for evidence reporting.

A float-input fallback keeps the historical six-region projection for
callers that pass non-integral coordinates.
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]
Seg = Tuple[Point, Point]

# Exact result: squared distance num/den, then closest points on s1 and s2,
# each expressed as (num_x, num_y, den) with a shared, positive denominator.
ExactResult = Tuple[int, int, Tuple[int, int, int], Tuple[int, int, int]]


def _point_segment_exact(
    px: int, py: int,
    ax: int, ay: int,
    bx: int, by: int,
) -> Tuple[int, int, Tuple[int, int, int]]:
    """Squared distance from point P to segment AB and the closest point Q.

    Returns ``(d2_num, d2_den, (qx_num, qy_num, q_den))`` where
    ``d2 == d2_num / d2_den`` and ``Q == (qx_num, qy_num) / q_den`` exactly.
    """
    ux, uy = bx - ax, by - ay
    length_sq = ux * ux + uy * uy
    if length_sq == 0:
        dx, dy = px - ax, py - ay
        return dx * dx + dy * dy, 1, (ax, ay, 1)

    s_num = (px - ax) * ux + (py - ay) * uy
    if s_num <= 0:
        dx, dy = px - ax, py - ay
        return dx * dx + dy * dy, 1, (ax, ay, 1)
    if s_num >= length_sq:
        dx, dy = px - bx, py - by
        return dx * dx + dy * dy, 1, (bx, by, 1)

    # Q = A + (s_num / length_sq) * u, an interior projection.
    qx_num = ax * length_sq + s_num * ux
    qy_num = ay * length_sq + s_num * uy
    dx_num = px * length_sq - qx_num
    dy_num = py * length_sq - qy_num
    d2_num = dx_num * dx_num + dy_num * dy_num
    return d2_num, length_sq * length_sq, (qx_num, qy_num, length_sq)


def _cross(
    ax: int, ay: int, bx: int, by: int, cx: int, cy: int
) -> int:
    """Cross product (B - A) x (C - A)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(
    ax: int, ay: int, bx: int, by: int, px: int, py: int
) -> bool:
    """True when collinear point P lies on the closed segment AB."""
    return (
        min(ax, bx) <= px <= max(ax, bx)
        and min(ay, by) <= py <= max(ay, by)
    )


def _segments_intersect_exact(
    ax: int, ay: int, bx: int, by: int,
    cx: int, cy: int, dx: int, dy: int,
) -> bool:
    """Whether the two closed segments share any point (integer endpoints)."""
    o1 = _cross(ax, ay, bx, by, cx, cy)
    o2 = _cross(ax, ay, bx, by, dx, dy)
    o3 = _cross(cx, cy, dx, dy, ax, ay)
    o4 = _cross(cx, cy, dx, dy, bx, by)

    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and (
        (o3 > 0 > o4) or (o3 < 0 < o4)
    ):
        return True

    # Collinear endpoint containment (also covers shared endpoints).
    if o1 == 0 and _on_segment(ax, ay, bx, by, cx, cy):
        return True
    if o2 == 0 and _on_segment(ax, ay, bx, by, dx, dy):
        return True
    if o3 == 0 and _on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if o4 == 0 and _on_segment(cx, cy, dx, dy, bx, by):
        return True
    return False


def _intersection_point(
    ax: int, ay: int, bx: int, by: int,
    cx: int, cy: int, dx: int, dy: int,
) -> Tuple[int, int, int]:
    """Rational intersection ``(num_x, num_y, den)`` of the two non-parallel
    segment lines (the intersection is known to lie on both segments)."""
    ux, uy = bx - ax, by - ay
    vx, vy = dx - cx, dy - cy
    wx, wy = ax - cx, ay - cy
    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    e = ux * wx + uy * wy
    f = vx * wx + vy * wy
    det = a * c - b * b
    s_num = b * f - c * e
    nx = ax * det + s_num * ux
    ny = ay * det + s_num * uy
    if det < 0:
        return -nx, -ny, -det
    return nx, ny, det


def segment_distance_sq_exact(
    p1: Tuple[int, int], p2: Tuple[int, int],
    q1: Tuple[int, int], q2: Tuple[int, int],
) -> ExactResult:
    """Exact minimum squared Euclidean distance between closed segments.

    Inputs have integer coordinates.  Returns ``(num, den, point_a, point_b)``
    with ``distance**2 == num / den`` (``den > 0``) and each closest point a
    rational ``(num_x, num_y, den)`` lying on the respective segment.  The
    key observation is that a convex quadratic minimiser over the parameter
    square is either the line intersection (distance zero) or a boundary
    point, so four point-to-segment projections exhaust the non-crossing
    cases -- with no floating point clamping involved.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = q1
    dx, dy = q2

    deg_a = ax == bx and ay == by
    deg_c = cx == dx and cy == dy
    if deg_a and deg_c:
        ex, ey = ax - cx, ay - cy
        pt = (ax, ay, 1)
        return ex * ex + ey * ey, 1, pt, (cx, cy, 1)
    if deg_a:
        n, d, q = _point_segment_exact(ax, ay, cx, cy, dx, dy)
        return n, d, (ax, ay, 1), q
    if deg_c:
        n, d, p = _point_segment_exact(cx, cy, ax, ay, bx, by)
        return n, d, p, (cx, cy, 1)

    if _segments_intersect_exact(
        ax, ay, bx, by, cx, cy, dx, dy
    ):
        ux, uy = bx - ax, by - ay
        vx, vy = dx - cx, dy - cy
        det = (ux * ux + uy * uy) * (vx * vx + vy * vy) - (
            ux * vx + uy * vy
        ) * (ux * vx + uy * vy)
        if det == 0:
            # Collinear overlap: a shared endpoint (known to exist from the
            # containment test) is itself a zero-distance witness.
            shared = (
                (cx, cy) if _on_segment(ax, ay, bx, by, cx, cy)
                else (dx, dy) if _on_segment(ax, ay, bx, by, dx, dy)
                else (ax, ay) if _on_segment(cx, cy, dx, dy, ax, ay)
                else (bx, by)
            )
            pt = (shared[0], shared[1], 1)
            return 0, 1, pt, pt
        x = _intersection_point(
            ax, ay, bx, by, cx, cy, dx, dy
        )
        return 0, 1, x, x

    candidates = []

    # Endpoints of s1 projected onto s2.
    n, d, q = _point_segment_exact(ax, ay, cx, cy, dx, dy)
    candidates.append((n, d, (ax, ay, 1), q))
    n, d, q = _point_segment_exact(bx, by, cx, cy, dx, dy)
    candidates.append((n, d, (bx, by, 1), q))

    # Endpoints of s2 projected onto s1.
    n, d, p = _point_segment_exact(cx, cy, ax, ay, bx, by)
    candidates.append((n, d, p, (cx, cy, 1)))
    n, d, p = _point_segment_exact(dx, dy, ax, ay, bx, by)
    candidates.append((n, d, p, (dx, dy, 1)))

    best = candidates[0]
    for cand in candidates[1:]:
        # Compare n1/d1 < n2/d2 by cross multiplication (denominators > 0).
        if cand[0] * best[1] < best[0] * cand[1]:
            best = cand
    return best


def segments_clearance_ok(
    p1: Tuple[int, int], p2: Tuple[int, int],
    q1: Tuple[int, int], q2: Tuple[int, int],
    clearance: int,
) -> bool:
    """Exact predicate: closed-segment distance ``>= clearance``.

    Equality is admissible (closed intervals); touching or crossing
    segments have distance zero and never pass for positive clearance.
    """
    num, den, _, _ = segment_distance_sq_exact(p1, p2, q1, q2)
    return num >= clearance * clearance * den


def _safe_div(num: int, den: int) -> float:
    try:
        return num / den
    except OverflowError:
        return math.copysign(math.inf, num)


def _exact_as_float(result: ExactResult) -> Tuple[float, Point, Point]:
    num, den, pa, pb = result
    if num == 0:
        dist = 0.0
    else:
        # sqrt(num/den) without rounding the ratio before taking the root.
        try:
            dist = math.sqrt(num / den)
        except OverflowError:
            dist = math.inf
    point_a = (_safe_div(pa[0], pa[2]), _safe_div(pa[1], pa[2]))
    point_b = (_safe_div(pb[0], pb[2]), _safe_div(pb[1], pb[2]))
    return dist, point_a, point_b


def _all_ints(*values) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def segment_distance(s1: Seg, s2: Seg) -> Tuple[float, Point, Point]:
    """Minimum Euclidean distance between two closed segments.

    Shared endpoints (or crossing/overlapping segments) yield ``0``.
    Returns the distance together with the pair of closest points.

    Integer-coordinate segments take the exact arithmetic path (vital when
    coordinates exceed the float64 exact-integer range); non-integral inputs
    use the standard clamped six-region projection.
    """
    a1, a2 = s1
    b1, b2 = s2
    if _all_ints(a1[0], a1[1], a2[0], a2[1], b1[0], b1[1], b2[0], b2[1]):
        result = segment_distance_sq_exact(
            (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])),
            (int(b1[0]), int(b1[1])), (int(b2[0]), int(b2[1])),
        )
        return _exact_as_float(result)
    return _segment_distance_float(s1, s2)


def _segment_distance_float(s1: Seg, s2: Seg) -> Tuple[float, Point, Point]:
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
