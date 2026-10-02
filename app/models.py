"""Request/response schemas and input validation."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Arm(BaseModel):
    id: str = Field(min_length=1, description="Unique fibre-arm identifier")
    x: int = Field(strict=True, description="Integer base x coordinate")
    y: int = Field(strict=True, description="Integer base y coordinate")
    max_extension: int = Field(
        strict=True, alias="reach", gt=0,
        description="Maximum reach (positive integer)",
    )

    model_config = {"populate_by_name": True}


class Target(BaseModel):
    id: str = Field(min_length=1, description="Unique target identifier")
    x: int = Field(strict=True, description="Integer x coordinate")
    y: int = Field(strict=True, description="Integer y coordinate")
    priority: int = Field(strict=True, gt=0, description="Positive integer priority")


class AssignmentRequest(BaseModel):
    arms: List[Arm] = Field(min_length=6, max_length=12)
    targets: List[Target] = Field(min_length=6, max_length=16)
    clearance: int = Field(
        strict=True, gt=0,
        description="Positive integer safety clearance",
    )
    min_assignments: int = Field(
        strict=True, gt=0, alias="minimum_assignments",
        description="Required minimum number of assignments (positive integer)",
    )

    model_config = {"populate_by_name": True}

    @field_validator("arms")
    @classmethod
    def _unique_arms(cls, v: List[Arm]) -> List[Arm]:
        ids = [a.id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError("arm ids must be unique")
        coords = [(a.x, a.y) for a in v]
        if len(set(coords)) != len(coords):
            raise ValueError("arm base coordinates must be unique")
        return v

    @field_validator("targets")
    @classmethod
    def _unique_targets(cls, v: List[Target]) -> List[Target]:
        ids = [t.id for t in v]
        if len(set(ids)) != len(ids):
            raise ValueError("target ids must be unique")
        coords = [(t.x, t.y) for t in v]
        if len(set(coords)) != len(coords):
            raise ValueError("target coordinates must be unique")
        return v

    def cross_validate(self) -> None:
        """Extra checks spanning both collections."""
        arm_ids = {a.id for a in self.arms}
        overlap = arm_ids.intersection(t.id for t in self.targets)
        if overlap:
            raise ValueError(f"ids must not be shared between arms and targets: {sorted(overlap)}")
        if self.min_assignments > len(self.arms):
            raise ValueError("minimum_assignments cannot exceed the number of arms")


class PairEvidence(BaseModel):
    arm_a: str
    target_a: str
    arm_b: str
    target_b: str
    distance: float
    closest_point_on_a: List[float]
    closest_point_on_b: List[float]
    satisfies_clearance: bool


class Assignment(BaseModel):
    arm_id: str
    target_id: str
    extension_length: float
    extension_length_squared: int


class ArmLength(BaseModel):
    arm_id: str
    assigned: bool
    extension_length: Optional[float]
    extension_length_squared: Optional[int]


class UnassignedObject(BaseModel):
    kind: Literal["arm", "target"]
    id: str
    x: int
    y: int
    detail: str


class Objectives(BaseModel):
    assigned_count: int
    priority_sum: int
    extension_squared_sum: int
    stable_sequence: List[int] = Field(
        description="Connected target index (1-based) per arm in request order; 0 = idle"
    )


class ClearanceEvidence(BaseModel):
    required_clearance: int
    minimum_pair_distance: Optional[float] = Field(
        default=None,
        description="Smallest distance between any two connected segments; "
        "null when fewer than two arms are assigned",
    )
    satisfied: bool
    checked_pairs: int
    tightest_pairs: List[PairEvidence]


class AssignmentResponse(BaseModel):
    status: Literal["satisfied", "below_minimum"]
    reason: Optional[str]
    minimum_assignments: int
    maximum_attainable: int
    assignments: List[Assignment]
    unassigned: List[UnassignedObject]
    arm_lengths: List[ArmLength]
    clearance_evidence: ClearanceEvidence
    objectives: Objectives
