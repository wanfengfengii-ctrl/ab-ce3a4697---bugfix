
from app.geometry import (
    segment_distance,
    segment_distance_sq_exact,
    segments_clearance_ok,
)


def test_intersecting_segments_zero():
    s1 = ((0.0, 0.0), (10.0, 10.0))
    s2 = ((0.0, 10.0), (10.0, 0.0))
    d, p, q = segment_distance(s1, s2)
    assert d < 1e-12


def test_shared_endpoint_zero():
    s1 = ((0.0, 0.0), (3.0, 4.0))
    s2 = ((3.0, 4.0), (9.0, 9.0))
    d, _, _ = segment_distance(s1, s2)
    assert d < 1e-12


def test_parallel_segments_perpendicular_gap():
    s1 = ((0.0, 0.0), (5.0, 0.0))
    s2 = ((1.0, 3.0), (4.0, 3.0))
    d, p, q = segment_distance(s1, s2)
    assert abs(d - 3.0) < 1e-12


def test_parallel_offset_segments_endpoint_gap():
    s1 = ((0.0, 0.0), (2.0, 0.0))
    s2 = ((5.0, 0.0), (8.0, 0.0))
    d, _, _ = segment_distance(s1, s2)
    assert abs(d - 3.0) < 1e-12


def test_endpoint_to_interior_distance():
    s1 = ((0.0, 0.0), (0.0, 5.0))
    s2 = ((3.0, 2.0), (3.0, 9.0))
    d, p, q = segment_distance(s1, s2)
    assert abs(d - 3.0) < 1e-12
    assert abs(p[1] - 2.0) < 1e-12


def test_skew_segments():
    s1 = ((0.0, 0.0), (10.0, 0.0))
    s2 = ((4.0, 3.0), (6.0, 3.0))
    d, p, q = segment_distance(s1, s2)
    assert abs(d - 3.0) < 1e-12


def test_points_on_segments():
    s1 = ((0.0, 0.0), (10.0, 0.0))
    s2 = ((3.0, 4.0), (7.0, 4.0))
    d, p, q = segment_distance(s1, s2)
    assert 0.0 - 1e-9 <= p[0] <= 10.0 + 1e-9 and abs(p[1]) < 1e-9
    assert 3.0 - 1e-9 <= q[0] <= 7.0 + 1e-9 and abs(q[1] - 4.0) < 1e-9


def test_degenerate_point_segment():
    d, p, q = segment_distance(((2.0, 2.0), (2.0, 2.0)), ((0.0, 0.0), (4.0, 0.0)))
    assert abs(d - 2.0) < 1e-12


def test_crossing_segments_large_coordinates_zero():
    # Coordinates beyond 2**53: float64 spacing at 10**17 is 16, so a naive
    # float projection rounds the endpoints and reports a phantom ~16 gap for
    # two strictly intersecting segments.  Exact arithmetic must return 0.
    n = 10**17
    s1 = ((n + 559, n - 119), (n + 778, n + 667))
    s2 = ((n + 615, n - 143), (n - 923, n + 545))
    num, den, _, _ = segment_distance_sq_exact(s1[0], s1[1], s2[0], s2[1])
    assert num == 0
    d, p, q = segment_distance(s1, s2)
    assert d == 0.0
    assert p == q


def test_clearance_predicate_exact_at_large_coordinates():
    n = 10**17
    s1 = ((n, 0), (n, 500))
    s2_ok = ((n + 10, 0), (n + 10, 500))
    s2_near = ((n + 9, 0), (n + 9, 500))
    # Distance exactly equal to clearance (closed segments) is admissible.
    assert segments_clearance_ok(s1[0], s1[1], s2_ok[0], s2_ok[1], 10)
    assert not segments_clearance_ok(s1[0], s1[1], s2_near[0], s2_near[1], 10)


def test_clearance_predicate_crossing_with_offset_origin():
    # The crossing pair from the adjudication regression, translated by an
    # arbitrary integer vector: result must be translation invariant.
    n = 10**17
    base = (10**17 + 123456789, -10**17 - 987654321)
    pts = [
        (n + 559, n - 119), (n + 778, n + 667),
        (n + 615, n - 143), (n - 923, n + 545),
    ]
    pts = [(x + base[0], y + base[1]) for x, y in pts]
    assert not segments_clearance_ok(pts[0], pts[1], pts[2], pts[3], 1)


def test_collinear_overlapping_segments_zero():
    s1 = ((10**17, 10**17), (10**17 + 100, 10**17))
    s2 = ((10**17 + 50, 10**17), (10**17 + 150, 10**17))
    num, den, pa, pb = segment_distance_sq_exact(s1[0], s1[1], s2[0], s2[1])
    assert num == 0 and den == 1
    # closest points coincide at a shared (integer) point
    assert pa == pb and pa[2] == 1


def test_shared_endpoint_large_coordinates_zero():
    a = (10**17 + 7, 10**17 - 3)
    s1 = (a, (10**17 + 999, 10**17 + 1001))
    s2 = (a, (10**17 - 888, 10**17 + 333))
    num, den, _, _ = segment_distance_sq_exact(s1[0], s1[1], s2[0], s2[1])
    assert num == 0
