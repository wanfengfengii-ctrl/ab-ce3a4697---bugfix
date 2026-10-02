#!/usr/bin/env python3
"""Collision-constrained API smoke test.

Runs against the service named by ``$API_BASE_URL`` (set by docker-compose
to ``http://api:${API_PORT}``).  Exercises:

1. a feasible adjudication and independently re-checks reach, target
   uniqueness and the pairwise closed-segment clearance from the response;
2. an instance that cannot reach ``minimum_assignments`` because the
   clearance forces pairs apart, and checks the reported maximum and reason;
3. malformed inputs that must be rejected with HTTP 422 without solving.

Exits non-zero on the first failed expectation.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import httpx

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PATH = "/api/v1/assignment/adjudicate"
CLEARANCE = int(os.environ.get("CLEARANCE", "2"))


def fail(msg: str) -> None:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def seg_dist(p1, p2, q1, q2) -> float:
    # Independent reference implementation for the smoke assertions.
    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    vx, vy = q2[0] - q1[0], q2[1] - q1[1]
    wx, wy = p1[0] - q1[0], p1[1] - q1[1]
    a, b, c = ux * ux + uy * uy, ux * vx + uy * vy, vx * vx + vy * vy
    e, f = ux * wx + uy * wy, vx * wx + vy * wy
    det = a * c - b * b
    if det != 0:
        s = (b * f - c * e) / det
        t = (a * f - b * e) / det
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
    px, py = p1[0] + s * ux, p1[1] + s * uy
    qx, qy = q1[0] + t * vx, q1[1] + t * vy
    return math.hypot(px - qx, py - qy)


def seg_dist_exact(p1, p2, q1, q2):
    """Exact squared distance (Fraction) between integer-coordinate
    closed segments -- independent of the service's geometry code."""

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, p):
        return (
            cross(a, b, p) == 0
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

    o = [cross(p1, p2, q1), cross(p1, p2, q2),
         cross(q1, q2, p1), cross(q1, q2, p2)]
    if ((o[0] > 0 > o[1] or o[0] < 0 < o[1])
            and (o[2] > 0 > o[3] or o[2] < 0 < o[3])):
        return Fraction(0)
    if on(p1, p2, q1) or on(p1, p2, q2) or on(q1, q2, p1) or on(q1, q2, p2):
        return Fraction(0)

    def point_seg(p, a, b):
        ux, uy = b[0] - a[0], b[1] - a[1]
        l2 = ux * ux + uy * uy
        if l2 == 0:
            return Fraction((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
        t = Fraction((p[0] - a[0]) * ux + (p[1] - a[1]) * uy, l2)
        t = max(Fraction(0), min(Fraction(1), t))
        qx, qy = a[0] + t * ux, a[1] + t * uy
        return (Fraction(p[0]) - qx) ** 2 + (Fraction(p[1]) - qy) ** 2

    return min(
        point_seg(p1, q1, q2), point_seg(p2, q1, q2),
        point_seg(q1, p1, p2), point_seg(q2, p1, p2),
    )


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=30)

    # ---- readiness -------------------------------------------------------
    r = client.get("/healthz/ready")
    if r.status_code != 200 or r.json().get("status") != "ready":
        fail(f"service not ready: {r.status_code} {r.text}")
    print("readiness ok")

    # ---- 1. feasible instance -------------------------------------------
    # Two parallel vertical fibre runs are exactly CLEARANCE apart; the
    # remaining four pairs sit well outside the collision corridor.
    c = CLEARANCE
    payload = {
        "arms": [
            {"id": "a1", "x": 0, "y": 0, "max_extension": 10},
            {"id": "a2", "x": c, "y": 0, "max_extension": 10},
            {"id": "a3", "x": 30, "y": 0, "max_extension": 10},
            {"id": "a4", "x": 60, "y": 0, "max_extension": 10},
            {"id": "a5", "x": 90, "y": 0, "max_extension": 10},
            {"id": "a6", "x": 120, "y": 0, "max_extension": 10},
        ],
        "targets": [
            {"id": "t1", "x": 0, "y": 5, "priority": 1},
            {"id": "t2", "x": c, "y": 5, "priority": 2},
            {"id": "t3", "x": 30, "y": 6, "priority": 3},
            {"id": "t4", "x": 60, "y": 4, "priority": 4},
            {"id": "t5", "x": 90, "y": 7, "priority": 5},
            {"id": "t6", "x": 120, "y": 3, "priority": 6},
        ],
        "clearance": c,
        "minimum_assignments": 5,
    }
    r = client.post(PATH, json=payload)
    if r.status_code != 200:
        fail(f"feasible case rejected: {r.status_code} {r.text}")
    data = r.json()

    if data["status"] != "satisfied":
        fail(f"expected satisfied, got {data['status']}: {data.get('reason')}")
    if data["objectives"]["assigned_count"] != 6:
        fail(f"expected 6 assignments, got {data['objectives']['assigned_count']}")

    arms = {a["id"]: a for a in payload["arms"]}
    targets = {t["id"]: t for t in payload["targets"]}

    used_targets = set()
    segs = []
    for a in data["assignments"]:
        if a["arm_id"] in used_targets:
            fail("duplicate arm in assignments")
        if a["target_id"] in used_targets:
            fail("target reused by two arms")
        used_targets.add(a["target_id"])
        base = arms[a["arm_id"]]
        tgt = targets[a["target_id"]]
        d = math.hypot(tgt["x"] - base["x"], tgt["y"] - base["y"])
        if d > base["max_extension"] + 1e-9:
            fail(f"arm {a['arm_id']} reaches beyond max_extension")
        if abs(d - a["extension_length"]) > 1e-9:
            fail("reported extension length inconsistent with geometry")
        segs.append(
            ((base["x"], base["y"]), (tgt["x"], tgt["y"]))
        )

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            d = seg_dist(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            if d + 1e-9 < c:
                fail(
                    f"segments {i} and {j} violate clearance: {d:.6f} < {c}"
                )
    if len(data["assignments"]) >= 2:
        md = data["clearance_evidence"]["minimum_pair_distance"]
        if abs(md - c) > 1e-6:
            fail(f"expected tightest distance {c}, got {md}")
        if not data["clearance_evidence"]["satisfied"]:
            fail("clearance evidence unexpectedly unsatisfied")
    if len(data["arm_lengths"]) != 6:
        fail("expected per-arm length evidence for all 6 arms")
    print("feasible case ok: 6 pairs, clearance re-verified")

    # ---- 2. clearance forces below the minimum --------------------------
    # Every arm/target pair lives in a small cluster with a huge clearance:
    # only one segment can be placed at a time.
    tight = {
        "arms": [
            {"id": "b1", "x": 0, "y": 0, "max_extension": 10},
            {"id": "b2", "x": 1, "y": 0, "max_extension": 10},
            {"id": "b3", "x": 2, "y": 0, "max_extension": 10},
            {"id": "b4", "x": 3, "y": 0, "max_extension": 10},
            {"id": "b5", "x": 4, "y": 0, "max_extension": 10},
            {"id": "b6", "x": 5, "y": 0, "max_extension": 10},
        ],
        "targets": [
            {"id": "g1", "x": 0, "y": 5, "priority": 1},
            {"id": "g2", "x": 1, "y": 5, "priority": 2},
            {"id": "g3", "x": 2, "y": 5, "priority": 3},
            {"id": "g4", "x": 3, "y": 5, "priority": 4},
            {"id": "g5", "x": 4, "y": 5, "priority": 5},
            {"id": "g6", "x": 5, "y": 5, "priority": 6},
        ],
        "clearance": 50,
        "minimum_assignments": 2,
    }
    r = client.post(PATH, json=tight)
    if r.status_code != 200:
        fail(f"tight case rejected: {r.status_code} {r.text}")
    data = r.json()
    if data["status"] != "below_minimum":
        fail(f"expected below_minimum, got {data['status']}")
    if data["maximum_attainable"] >= 2:
        fail(
            "clearance 50 over a 5-wide cluster must limit placements to 1, "
            f"got {data['maximum_attainable']}"
        )
    if not data.get("reason"):
        fail("below_minimum response must explain the shortfall")
    if data["objectives"]["assigned_count"] != data["maximum_attainable"]:
        fail("objective count inconsistent with maximum_attainable")
    # The returned partial assignment must still satisfy every constraint.
    placed = data["assignments"]
    if len(placed) > 1:
        fail("at most one pair should be placed in the tight cluster")
    print(
        f"below-minimum case ok: max {data['maximum_attainable']} < 2, reason given"
    )

    # ---- 2b. large coordinate translation must not hide crossings --------
    # Endpoints near 10**17 exceed 2**53 (float64 ULP = 16); the a0->t0 and
    # a1->t1 closed segments strictly intersect.  Six assignments require
    # both, so the true maximum is five.
    n = 10**17
    translated = {
        "arms": [
            {"id": "c0", "x": n + 559, "y": n - 119, "max_extension": 816},
            {"id": "c1", "x": n + 615, "y": n - 143, "max_extension": 1685},
            {"id": "c2", "x": 2000, "y": 0, "max_extension": 20},
            {"id": "c3", "x": 3000, "y": 0, "max_extension": 20},
            {"id": "c4", "x": 4000, "y": 0, "max_extension": 20},
            {"id": "c5", "x": 5000, "y": 0, "max_extension": 20},
        ],
        "targets": [
            {"id": "u0", "x": n + 778, "y": n + 667, "priority": 1},
            {"id": "u1", "x": n - 923, "y": n + 545, "priority": 1},
            {"id": "u2", "x": 2000, "y": 20, "priority": 1},
            {"id": "u3", "x": 3000, "y": 20, "priority": 1},
            {"id": "u4", "x": 4000, "y": 20, "priority": 1},
            {"id": "u5", "x": 5000, "y": 20, "priority": 1},
        ],
        "clearance": 10,
        "minimum_assignments": 6,
    }
    r = client.post(PATH, json=translated)
    if r.status_code != 200:
        fail(f"translated case rejected: {r.status_code} {r.text}")
    data = r.json()
    if data["status"] != "below_minimum":
        fail(f"translated crossing must be below_minimum, got {data['status']}")
    if data["maximum_attainable"] != 5:
        fail(f"translated crossing caps attainable pairs at 5, got "
             f"{data['maximum_attainable']}")
    chosen = {(a["arm_id"], a["target_id"]) for a in data["assignments"]}
    if {("c0", "u0"), ("c1", "u1")} <= chosen:
        fail("both intersecting pairs were admitted into the solution")

    arms = {a["id"]: a for a in translated["arms"]}
    tgts = {t["id"]: t for t in translated["targets"]}
    placed = list(chosen)
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ba, ta = placed[i]
            bb, tb = placed[j]
            a1, a2 = arms[ba], arms[bb]
            g1, g2 = tgts[ta], tgts[tb]
            d2 = seg_dist_exact(
                (a1["x"], a1["y"]), (g1["x"], g1["y"]),
                (a2["x"], a2["y"]), (g2["x"], g2["y"]),
            )
            if d2 < 100:
                fail(f"placed pair {placed[i]} vs {placed[j]} violates "
                     f"clearance 10: d^2 = {float(d2):.6f}")
    if not data["clearance_evidence"]["satisfied"]:
        fail("clearance evidence must agree with the collision-free solution")

    # Distance exactly equal to the clearance remains legal after translation.
    equal = {
        "arms": [
            {"id": "e0", "x": n, "y": n, "max_extension": 20},
            {"id": "e1", "x": n + 10, "y": n, "max_extension": 20},
            *[
                {"id": f"e{i}", "x": n + 1000 * i, "y": n, "max_extension": 20}
                for i in range(2, 6)
            ],
        ],
        "targets": [
            {"id": "v0", "x": n, "y": n + 10, "priority": 1},
            {"id": "v1", "x": n + 10, "y": n + 10, "priority": 1},
            *[
                {"id": f"v{i}", "x": n + 1000 * i, "y": n + 10, "priority": 1}
                for i in range(2, 6)
            ],
        ],
        "clearance": 10,
        "minimum_assignments": 6,
    }
    data = client.post(PATH, json=equal).json()
    if data["status"] != "satisfied" or data["maximum_attainable"] != 6:
        fail(f"distance == clearance must be admissible: {data['status']} "
             f"max={data['maximum_attainable']}")
    if data["clearance_evidence"]["minimum_pair_distance"] + 1e-9 < 10:
        fail("tightest reported pair drifted below the exact 10-unit clearance")
    print("translated-coordinate cases ok: crossing rejected (max 5), "
          "exact-equality gap admitted")

    # ---- 3. invalid inputs never reach the solver -----------------------
    base_good = {
        "arms": [
            {"id": f"a{i}", "x": 30 * i, "y": 0, "max_extension": 10}
            for i in range(1, 7)
        ],
        "targets": [
            {"id": f"t{i}", "x": 30 * i, "y": 5, "priority": i}
            for i in range(1, 7)
        ],
        "clearance": 1,
        "minimum_assignments": 1,
    }

    too_few = {**base_good, "arms": base_good["arms"][:5]}
    dup = {
        **base_good,
        "targets": [
            *base_good["targets"][:5],
            {"id": "t1", "x": 999, "y": 999, "priority": 1},
        ],
    }
    neg_clear = {**base_good, "clearance": -2}
    min_high = {**base_good, "minimum_assignments": 7}
    neg_reach = {
        **base_good,
        "arms": [
            {**a, "max_extension": -3} if a["id"] == "a1" else a
            for a in base_good["arms"]
        ],
    }
    non_int = {**base_good, "clearance": 1.5}

    for label, bad in [
        ("too few arms", too_few),
        ("duplicate target ids", dup),
        ("non-positive clearance", neg_clear),
        ("minimum above arm count", min_high),
        ("arm reaches negative", neg_reach),
        ("non-integer clearance", non_int),
    ]:
        r = client.post(PATH, json=bad)
        if r.status_code != 422:
            fail(f"invalid case '{label}' expected 422, got {r.status_code}: {r.text}")
    print("invalid-input cases ok: all rejected with 422")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
