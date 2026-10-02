
from app.geometry import segment_distance


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


def test_large_coordinate_offset_intersection_is_exact_zero():
    # At an offset of 1e17 a float ULP is ~16, so a float-based projection
    # falsely reports intersecting segments as a double-digit gap.  The
    # geometry layer must use exact arithmetic.
    n = 10**17
    s1 = ((n + 559, n - 119), (n + 778, n + 667))
    s2 = ((n + 615, n - 143), (n - 923, n + 545))
    d, p, q = segment_distance(s1, s2)
    assert d == 0.0
    assert p == q


def test_large_coordinate_offset_exact_clearance_gap():
    # Two genuinely separated segments near 1e17 must keep their true gap
    # (parallel vertical runs 10 units apart).
    n = 10**17
    s1 = ((n, n), (n, n + 500))
    s2 = ((n + 10, n), (n + 10, n + 500))
    d, _, _ = segment_distance(s1, s2)
    assert abs(d - 10.0) < 1e-9

