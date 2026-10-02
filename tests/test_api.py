
import pytest
from fastapi.testclient import TestClient

from app.main import app


def six_arms(offset=30):
    return [
        {"id": f"a{i}", "x": offset * i, "y": 0, "max_extension": 10}
        for i in range(1, 7)
    ]


def six_targets(offset=30):
    return [
        {"id": f"t{i}", "x": offset * i, "y": 5, "priority": i}
        for i in range(1, 7)
    ]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_readiness(client):
    r = client.get("/healthz/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_satisfied_response_shape(client):
    payload = {
        "arms": six_arms(),
        "targets": six_targets(),
        "clearance": 2,
        "minimum_assignments": 6,
    }
    r = client.post("/api/v1/assignment/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "satisfied"
    assert data["maximum_attainable"] == 6
    assert data["objectives"]["assigned_count"] == 6
    assert data["objectives"]["priority_sum"] == 21
    assert data["objectives"]["extension_squared_sum"] == 6 * 25
    assert data["objectives"]["stable_sequence"] == [1, 2, 3, 4, 5, 6]
    assert len(data["assignments"]) == 6
    assert len(data["arm_lengths"]) == 6
    assert data["clearance_evidence"]["checked_pairs"] == 15
    assert data["clearance_evidence"]["satisfied"] is True
    assert len(data["unassigned"]) == 0


def test_per_arm_lengths_and_pair_evidence(client):
    payload = {
        "arms": six_arms(),
        "targets": six_targets(),
        "clearance": 2,
        "minimum_assignments": 1,
    }
    data = client.post("/api/v1/assignment/adjudicate", json=payload).json()
    for al in data["arm_lengths"]:
        assert al["assigned"] is True
        assert abs(al["extension_length"] - 5.0) < 1e-9
        assert al["extension_length_squared"] == 25
    tightest = data["clearance_evidence"]["tightest_pairs"][0]
    # Closest adjacent pair runs from x=30 to x=60: distance 30 horizontally.
    assert tightest["distance"] > 29.9


def test_below_minimum_due_to_clearance(client):
    payload = {
        "arms": [
            {"id": f"b{i}", "x": i, "y": 0, "max_extension": 10}
            for i in range(1, 7)
        ],
        "targets": [
            {"id": f"g{i}", "x": i, "y": 5, "priority": i}
            for i in range(1, 7)
        ],
        "clearance": 50,
        "minimum_assignments": 3,
    }
    data = client.post("/api/v1/assignment/adjudicate", json=payload).json()
    assert data["status"] == "below_minimum"
    assert data["maximum_attainable"] == 1
    assert data["objectives"]["assigned_count"] == 1
    assert "3" in data["reason"] and "clearance" in data["reason"].lower()
    # Unassigned objects explain every idle arm and unused target.
    idle_arms = [u for u in data["unassigned"] if u["kind"] == "arm"]
    unused = [u for u in data["unassigned"] if u["kind"] == "target"]
    assert len(idle_arms) == 5 and len(unused) == 5
    assert data["clearance_evidence"]["checked_pairs"] == 0


def test_below_minimum_due_to_reachability(client):
    # All six arms clustered at the origin with tiny reach; targets far away.
    payload = {
        "arms": [
            {"id": f"b{i}", "x": i, "y": 0, "max_extension": 1}
            for i in range(1, 7)
        ],
        "targets": [
            {"id": f"g{i}", "x": 100 + i, "y": 100 + i, "priority": i}
            for i in range(1, 7)
        ],
        "clearance": 1,
        "minimum_assignments": 2,
    }
    data = client.post("/api/v1/assignment/adjudicate", json=payload).json()
    assert data["status"] == "below_minimum"
    assert data["maximum_attainable"] == 0
    assert "unreachable" in data["reason"].lower()
    assert all(u["kind"] == "arm" for u in data["unassigned"][:6])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(arms=p["arms"][:5]),
        lambda p: p.update(targets=p["targets"] + [p["targets"][0]]),
        lambda p: p.update(clearance=0),
        lambda p: p.update(minimum_assignments=99),
        lambda p: p["arms"].__setitem__(0, {**p["arms"][0], "max_extension": -1}),
        lambda p: p["targets"].__setitem__(0, {**p["targets"][0], "priority": 0}),
        lambda p: p.update(clearance="wide"),
        lambda p: p.pop("arms"),
    ],
)
def test_invalid_inputs_rejected(client, mutation):
    payload = {
        "arms": six_arms(),
        "targets": six_targets(),
        "clearance": 2,
        "minimum_assignments": 1,
    }
    mutation(payload)
    r = client.post("/api/v1/assignment/adjudicate", json=payload)
    assert r.status_code == 422


def test_arm_and_target_id_overlap_rejected(client):
    payload = {
        "arms": six_arms(),
        "targets": [
            {"id": "a1", "x": 30, "y": 5, "priority": 1},
            *six_targets()[1:],
        ],
        "clearance": 2,
        "minimum_assignments": 1,
    }
    r = client.post("/api/v1/assignment/adjudicate", json=payload)
    assert r.status_code == 422


def test_large_coordinate_offset_intersecting_arms_excluded(client):
    # Shifted by N = 1e17 the two only ways to reach six pairs are
    # a0->t0 and a1->t1, but those closed segments strictly intersect
    # (true gap 0).  A float pre-conversion of the integer coordinates
    # hides the collision (ULP ~ 16 at this offset); the adjudicator must
    # still cap the plan at 5 and report below_minimum.
    n = 10**17
    arms = [
        {"id": "a0", "x": n + 559, "y": n - 119, "max_extension": 816},
        {"id": "a1", "x": n + 615, "y": n - 143, "max_extension": 1685},
    ] + [
        {"id": f"a{i}", "x": i * 1000, "y": 0, "max_extension": 20}
        for i in range(2, 6)
    ]
    targets = [
        {"id": "t0", "x": n + 778, "y": n + 667, "priority": 1},
        {"id": "t1", "x": n - 923, "y": n + 545, "priority": 1},
    ] + [
        {"id": f"t{i}", "x": i * 1000, "y": 20, "priority": 1}
        for i in range(2, 6)
    ]
    payload = {
        "arms": arms,
        "targets": targets,
        "clearance": 10,
        "minimum_assignments": 6,
    }
    r = client.post("/api/v1/assignment/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["status"] == "below_minimum"
    assert data["maximum_attainable"] == 5
    assert data["objectives"]["assigned_count"] == 5

    pairs = {(a["arm_id"], a["target_id"]) for a in data["assignments"]}
    assert ("a0", "t0") not in pairs or ("a1", "t1") not in pairs

    # Every returned plan must honour the 10-unit clearance; the
    # intersecting pair must not be reported as a 16-unit gap.
    evidence = data["clearance_evidence"]
    assert evidence["satisfied"] is True
    assert evidence["minimum_pair_distance"] is None or (
        evidence["minimum_pair_distance"] + 1e-9 >= 10
    )
    for rec in evidence["tightest_pairs"]:
        assert rec["satisfies_clearance"] is True
        assert rec["distance"] + 1e-9 >= 10

    # Independently re-verify the clearance of every selected segment
    # using exact geometry.
    from app.geometry import segment_distance

    by_arm = {a["id"]: a for a in arms}
    by_target = {t["id"]: t for t in targets}
    segs = []
    for a in data["assignments"]:
        base, tgt = by_arm[a["arm_id"]], by_target[a["target_id"]]
        segs.append(
            ((base["x"], base["y"]), (tgt["x"], tgt["y"]))
        )
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            d, _, _ = segment_distance(segs[i], segs[j])
            assert d + 1e-9 >= 10

