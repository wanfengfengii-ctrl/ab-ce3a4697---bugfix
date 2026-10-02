"""FastAPI application exposing the fibre-arm adjudication service."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .models import (
    ArmLength,
    Assignment,
    AssignmentRequest,
    AssignmentResponse,
    ClearanceEvidence,
    Objectives,
    PairEvidence,
    UnassignedObject,
)
from .geometry import segments_clearance_ok
from .matching import max_cardinality
from .solver import solve

READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tiny capability self-check before the service announces readiness:
    # the adjudication endpoint only reports healthy once the solver runs.
    result = solve(
        arms=[
            {"id": "s1", "x": 0, "y": 0, "reach": 5},
            {"id": "s2", "x": 10, "y": 0, "reach": 5},
        ],
        targets=[
            {"id": "t1", "x": 3, "y": 0, "priority": 1},
            {"id": "t2", "x": 7, "y": 0, "priority": 2},
        ],
        clearance=1,
    )
    assert result["count"] == 2
    global READY
    READY = True
    yield
    READY = False


app = FastAPI(
    title="Fibre-Arm Assignment Adjudicator",
    version="1.0.0",
    description=(
        "Allocates retractable fibre arms to spectroscopic targets under "
        "reach, uniqueness and inter-segment clearance constraints, "
        "optimising count, priority, extension and assignment stability."
    ),
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    # Cross-field validation failures must never reach the solver.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz/ready")
async def readyz() -> Dict[str, Any]:
    if not READY:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "solver self-check pending"},
        )
    return {"status": "ready", "adjudication": "open"}


def _build_response(req: AssignmentRequest) -> AssignmentResponse:
    arm_dicts = [a.model_dump() for a in req.arms]
    target_dicts = [t.model_dump() for t in req.targets]
    for d, a in zip(arm_dicts, req.arms):
        d["reach"] = a.max_extension
    result = solve(arm_dicts, target_dicts, req.clearance)
    vec: Tuple[int, ...] = result["vec"]
    count: int = result["count"]
    arms = req.arms
    targets = req.targets
    m = len(targets)

    # Reachability diagnostics (before clearance is applied).
    reachable_targets = set()
    arm_reach_count = [0] * len(arms)
    for i, a in enumerate(arms):
        for j, t in enumerate(targets):
            d2 = (t.x - a.x) ** 2 + (t.y - a.y) ** 2
            if d2 <= a.max_extension ** 2:
                reachable_targets.add(j)
                arm_reach_count[i] += 1
    # Pairwise clearance evidence over connected segments.
    placed: List[Tuple[int, int]] = [
        (i, j) for i, j in enumerate(vec) if j >= 0
    ]
    pair_records: List[Tuple[float, PairEvidence]] = []
    pair_distance = result["pair_distance"]
    for idx in range(len(placed)):
        i, j = placed[idx]
        for k in range(idx + 1, len(placed)):
            i2, j2 = placed[k]
            d, p, q = pair_distance(i, j, i2, j2)
            # Exact integer predicate: the float distance is evidence only.
            ok = segments_clearance_ok(
                (arms[i].x, arms[i].y), (targets[j].x, targets[j].y),
                (arms[i2].x, arms[i2].y), (targets[j2].x, targets[j2].y),
                req.clearance,
            )
            rec = PairEvidence(
                arm_a=arms[i].id,
                target_a=targets[j].id,
                arm_b=arms[i2].id,
                target_b=targets[j2].id,
                distance=d,
                closest_point_on_a=[p[0], p[1]],
                closest_point_on_b=[q[0], q[1]],
                satisfies_clearance=ok,
            )
            pair_records.append((d, rec))
    pair_records.sort(key=lambda r: r[0])
    min_pair: Optional[float] = pair_records[0][0] if pair_records else None
    clearance_ok = all(rec.satisfies_clearance for _, rec in pair_records)

    assignments: List[Assignment] = []
    arm_lengths: List[ArmLength] = []
    assigned_targets = {j for j in vec if j >= 0}
    for i, a in enumerate(arms):
        j = vec[i]
        if j >= 0:
            d2 = (targets[j].x - a.x) ** 2 + (targets[j].y - a.y) ** 2
            length = math.sqrt(d2)
            assignments.append(
                Assignment(
                    arm_id=a.id,
                    target_id=targets[j].id,
                    extension_length=length,
                    extension_length_squared=d2,
                )
            )
            arm_lengths.append(
                ArmLength(
                    arm_id=a.id,
                    assigned=True,
                    extension_length=length,
                    extension_length_squared=d2,
                )
            )
        else:
            arm_lengths.append(
                ArmLength(
                    arm_id=a.id,
                    assigned=False,
                    extension_length=None,
                    extension_length_squared=None,
                )
            )

    unassigned: List[UnassignedObject] = []
    for i, a in enumerate(arms):
        if vec[i] < 0:
            if arm_reach_count[i] == 0:
                detail = "no target lies within max_extension of this base"
            else:
                detail = (
                    "idle in the optimal assignment: every reachable target is "
                    "taken by another arm or blocked by the safety clearance"
                )
            unassigned.append(
                UnassignedObject(kind="arm", id=a.id, x=a.x, y=a.y, detail=detail)
            )
    for j, t in enumerate(targets):
        if j not in assigned_targets:
            if j not in reachable_targets:
                detail = "no arm base can reach this target within max_extension"
            else:
                detail = (
                    "reachable but unused: target uniqueness, clearance and the "
                    "lexicographic objective leave it out of the optimum"
                )
            unassigned.append(
                UnassignedObject(kind="target", id=t.id, x=t.x, y=t.y, detail=detail)
            )

    stable_sequence = [(j + 1) if j >= 0 else 0 for j in vec]

    satisfied = count >= req.min_assignments
    if satisfied:
        status = "satisfied"
        reason = None
    else:
        unreachable = m - len(reachable_targets)
        # Cardinality reachable without the clearance constraint.
        adj = []
        for a in arms:
            neigh = []
            for j, t in enumerate(targets):
                if (t.x - a.x) ** 2 + (t.y - a.y) ** 2 <= a.max_extension ** 2:
                    neigh.append(j)
            adj.append(neigh)
        mu_reach = max_cardinality(adj, m)
        status = "below_minimum"
        reason = (
            f"Only {count} arm-target pair(s) can be placed, fewer than the "
            f"required minimum {req.min_assignments}. Reachability alone "
            f"(max_extension, target uniqueness) supports at most "
            f"{mu_reach} pair(s); {unreachable} target(s) are unreachable by "
            f"any arm. The binding shortfall is forced by the "
            f"{req.clearance}-unit closed-segment clearance once pairs are placed."
            if mu_reach > count
            else f"Only {count} arm-target pair(s) can be placed, fewer than "
            f"the required minimum {req.min_assignments}. At most {mu_reach} "
            f"pair(s) are possible from reachability and target uniqueness "
            f"alone; {unreachable} target(s) are unreachable by any arm."
        )

    evidence = ClearanceEvidence(
        required_clearance=req.clearance,
        minimum_pair_distance=min_pair,
        satisfied=clearance_ok,
        checked_pairs=len(pair_records),
        tightest_pairs=[rec for _, rec in pair_records[:5]],
    )

    objectives = Objectives(
        assigned_count=count,
        priority_sum=int(result["priority_sum"]),
        extension_squared_sum=int(result["length_sq_sum"]),
        stable_sequence=stable_sequence,
    )

    return AssignmentResponse(
        status=status,
        reason=reason,
        minimum_assignments=req.min_assignments,
        maximum_attainable=count,
        assignments=assignments,
        unassigned=unassigned,
        arm_lengths=arm_lengths,
        clearance_evidence=evidence,
        objectives=objectives,
    )


@app.post(
    "/api/v1/assignment/adjudicate",
    response_model=AssignmentResponse,
    summary="Adjudicate an optimal fibre-arm assignment",
)
async def adjudicate(payload: Dict[str, Any]) -> AssignmentResponse:
    # Parse strictly: invalid input is rejected before any solving starts.
    try:
        req = AssignmentRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    try:
        req.cross_validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _build_response(req)


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    return {
        "service": "fibre-arm-assignment-adjudicator",
        "adjudicate": "/api/v1/assignment/adjudicate",
        "health": "/healthz/ready",
    }
