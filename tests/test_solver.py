"""Solver tests.

The optimality checks use an independent brute-force enumerator over all
partial injective assignments with a self-contained segment-distance
routine, so they validate the branch-and-bound solver rather than share its
code.
"""

from __future__ import annotations

import math
import random

from app.solver import solve


def brute_segment_distance(p1, p2, q1, q2):
    def point_seg(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / l2
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    if p1 == p2 and q1 == q2:
        return math.hypot(p1[0] - q1[0], p1[1] - q1[1])
    if p1 == p2:
        return point_seg(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1])
    if q1 == q2:
        return point_seg(q1[0], q1[1], p1[0], p1[1], p2[0], p2[1])

    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    vx, vy = q2[0] - q1[0], q2[1] - q1[1]
    wx, wy = p1[0] - q1[0], p1[1] - q1[1]
    a, b, c = ux * ux + uy * uy, ux * vx + uy * vy, vx * vx + vy * vy
    e, f = ux * wx + uy * wy, vx * wx + vy * wy
    det = a * c - b * b
    if det != 0:
        s, t = (b * f - c * e) / det, (a * f - b * e) / det
    else:
        s, t = 0.0, f / c
    if s < 0.0:
        s, t = 0.0, f / c
    elif s > 1.0:
        s, t = 1.0, (f + b) / c
    if t < 0.0:
        t = 0.0
        s = 0.0 if -e < 0.0 else 1.0 if -e > a else -e / a
    elif t > 1.0:
        t = 1.0
        s = 0.0 if b - e < 0.0 else 1.0 if b - e > a else (b - e) / a
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    return math.hypot(
        p1[0] + s * ux - (q1[0] + t * vx),
        p1[1] + s * uy - (q1[1] + t * vy),
    )


def brute_optimal(arms, targets, clearance):
    n, m = len(arms), len(targets)

    def reach(i, j):
        a, t = arms[i], targets[j]
        return (t["x"] - a["x"]) ** 2 + (t["y"] - a["y"]) ** 2 <= a["reach"] ** 2

    options = [[j for j in range(m) if reach(i, j)] for i in range(n)]

    def seg_ok(placed, i, j):
        for i2, j2 in placed:
            a1, t1 = arms[i], targets[j]
            a2, t2 = arms[i2], targets[j2]
            d = brute_segment_distance(
                (a1["x"], a1["y"]), (t1["x"], t1["y"]),
                (a2["x"], a2["y"]), (t2["x"], t2["y"]),
            )
            if d + 1e-9 < clearance:
                return False
        return True

    best = None

    def rec(i, used, placed, vec, cnt, prio, ln):
        nonlocal best
        if i == n:
            key = (-cnt, -prio, ln, tuple(
                (j + 1) if j >= 0 else m + 1 for j in vec
            ))
            if best is None or key < best[0]:
                best = (key, tuple(vec))
            return
        # idle branch
        vec.append(-1)
        rec(i + 1, used, placed, vec, cnt, prio, ln)
        vec.pop()
        for j in options[i]:
            bit = 1 << j
            if used & bit:
                continue
            if not seg_ok(placed, i, j):
                continue
            d2 = (targets[j]["x"] - arms[i]["x"]) ** 2 + (
                targets[j]["y"] - arms[i]["y"]
            ) ** 2
            vec.append(j)
            rec(i + 1, used | bit, placed + [(i, j)], vec,
                cnt + 1, prio + targets[j]["priority"], ln + d2)
            vec.pop()

    rec(0, 0, [], [], 0, 0, 0)
    return best[1]


def make_instance(seed, n=7, m=9):
    rng = random.Random(seed)
    arms = [
        {
            "id": f"a{i}",
            "x": rng.randint(-20, 20),
            "y": rng.randint(-20, 20),
            "reach": rng.randint(4, 14),
        }
        for i in range(n)
    ]
    # unique coordinates
    coords = set()
    targets = []
    while len(targets) < m:
        x, y = rng.randint(-25, 25), rng.randint(-25, 25)
        if (x, y) in coords:
            continue
        coords.add((x, y))
        targets.append(
            {"id": f"t{len(targets)}", "x": x, "y": y,
             "priority": rng.randint(1, 9)}
        )
    return arms, targets


def test_matches_brute_force_across_seeds():
    for seed in range(40):
        arms, targets = make_instance(seed)
        clearance = seed % 6 + 1
        res = solve(arms, targets, clearance)
        expected = brute_optimal(arms, targets, clearance)
        assert tuple(res["vec"]) == expected, (
            f"seed {seed}: solver {res['vec']} != brute {list(expected)}"
        )


def test_count_priority_extension_objectives():
    # Two targets reachable from one arm: higher priority must win.
    arms = [{"id": "a", "x": 0, "y": 0, "reach": 5}]
    # pad to 6/6 is solver-agnostic; add far-away independent arms/targets
    arms += [{"id": f"a{i}", "x": 100 * i, "y": 100, "reach": 5}
             for i in range(1, 6)]
    targets = [
        {"id": "low", "x": 3, "y": 0, "priority": 1},
        {"id": "high", "x": 4, "y": 0, "priority": 9},
    ]
    targets += [
        {"id": f"t{i}", "x": 100 * i, "y": 103, "priority": i}
        for i in range(1, 5)
    ]
    res = solve(arms, targets, 1)
    assert res["vec"][0] == 1  # high priority target, index 1
    assert res["count"] == 5
    assert res["priority_sum"] == 9 + 1 + 2 + 3 + 4


def test_extension_tiebreak_prefers_closer_target():
    arms = [{"id": "a", "x": 0, "y": 0, "reach": 5}]
    arms += [{"id": f"a{i}", "x": 100 * i, "y": 100, "reach": 5}
             for i in range(1, 6)]
    targets = [
        {"id": "far", "x": 4, "y": 0, "priority": 3},
        {"id": "near", "x": 3, "y": 0, "priority": 3},
    ]
    targets += [
        {"id": f"t{i}", "x": 100 * i, "y": 103, "priority": i}
        for i in range(1, 5)
    ]
    res = solve(arms, targets, 1)
    assert res["vec"][0] == 1  # nearer target, equal priority
    assert res["length_sq_sum"] == 9 + 4 * 9


def test_clearance_forces_collision_avoidance():
    # Both bases on x-axis 1 unit apart, both targets 5 units up; clearance 2
    # makes the parallel segments collide, so only one pair may be placed.
    arms = [
        {"id": "a1", "x": 0, "y": 0, "reach": 10},
        {"id": "a2", "x": 1, "y": 0, "reach": 10},
    ]
    arms += [{"id": f"a{i}", "x": 100 * i, "y": 100, "reach": 5}
             for i in range(2, 6)]
    targets = [
        {"id": "g1", "x": 0, "y": 5, "priority": 1},
        {"id": "g2", "x": 1, "y": 5, "priority": 2},
    ]
    targets += [
        {"id": f"t{i}", "x": 100 * i, "y": 103, "priority": i}
        for i in range(2, 6)
    ]
    res = solve(arms, targets, 2)
    assert res["count"] == 5  # not 6
    # Higher priority target g2 survives.
    assert res["vec"][0] == -1 and res["vec"][1] == 1


def test_clearance_boundary_is_inclusive():
    # Segments exactly `clearance` apart are admissible (closed segments).
    arms = [
        {"id": "a1", "x": 0, "y": 0, "reach": 5},
        {"id": "a2", "x": 2, "y": 0, "reach": 5},
    ]
    arms += [{"id": f"a{i}", "x": 100 * i, "y": 100, "reach": 5}
             for i in range(2, 6)]
    targets = [
        {"id": "g1", "x": 0, "y": 5, "priority": 1},
        {"id": "g2", "x": 2, "y": 5, "priority": 2},
    ]
    targets += [
        {"id": f"t{i}", "x": 100 * i, "y": 103, "priority": i}
        for i in range(2, 6)
    ]
    res = solve(arms, targets, 2)
    assert res["count"] == 6


def test_stable_sequence_prefers_earlier_target_index():
    # Equal everything: identical-length options to t0/t1 from a0.
    arms = [{"id": "a0", "x": 0, "y": 0, "reach": 5}]
    arms += [{"id": f"a{i}", "x": 100 * i, "y": 100, "reach": 5}
             for i in range(1, 6)]
    targets = [
        {"id": "x1", "x": 3, "y": 0, "priority": 4},
        {"id": "x2", "x": -3, "y": 0, "priority": 4},
    ]
    targets += [
        {"id": f"t{i}", "x": 100 * i, "y": 103, "priority": i}
        for i in range(1, 5)
    ]
    res = solve(arms, targets, 1)
    assert res["vec"][0] == 0  # earlier listed target wins tie


def test_full_size_instance_performance():
    arms, targets = make_instance(1234, n=12, m=16)
    import time

    start = time.perf_counter()
    res = solve(arms, targets, 3)
    elapsed = time.perf_counter() - start
    assert elapsed < 15.0, f"solver too slow: {elapsed:.1f}s"

    # Independently verify every hard constraint of the returned solution.
    vec = res["vec"]
    used = set()
    placed = []
    for i, j in enumerate(vec):
        if j < 0:
            continue
        assert j not in used
        used.add(j)
        a, t = arms[i], targets[j]
        assert (t["x"] - a["x"]) ** 2 + (t["y"] - a["y"]) ** 2 <= a["reach"] ** 2
        for i2, j2 in placed:
            a2, t2 = arms[i2], targets[j2]
            d = brute_segment_distance(
                (a["x"], a["y"]), (t["x"], t["y"]),
                (a2["x"], a2["y"]), (t2["x"], t2["y"]),
            )
            assert d + 1e-9 >= 3
        placed.append((i, j))
    assert res["count"] == len(placed)
