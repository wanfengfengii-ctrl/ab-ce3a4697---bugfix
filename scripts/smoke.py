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
