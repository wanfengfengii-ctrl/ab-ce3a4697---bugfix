
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
