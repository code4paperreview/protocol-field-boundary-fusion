from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class ConstraintStrength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintKind(str, Enum):
    LENGTH = "length"
    TLV = "tlv"
    REPETITION = "repetition"
    SEQUENCE = "sequence"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    PAYLOAD = "payload"
    CONSTANT = "constant"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class Span:
    message_index: int
    start: int
    stop: int
    type_code: str | None = None

    @property
    def width(self) -> int:
        return self.stop - self.start

    def overlaps(self, other: "Span") -> bool:
        if self.message_index != other.message_index:
            return False
        return self.start < other.stop and other.start < self.stop

    def contains(self, other: "Span") -> bool:
        if self.message_index != other.message_index:
            return False
        return self.start <= other.start and self.stop >= other.stop


@dataclass(slots=True)
class Constraint:
    name: str
    kind: ConstraintKind
    strength: ConstraintStrength
    source: str
    spans: list[Span]
    annotation: str = ""
    type_code: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    constraint_id: str = field(default_factory=lambda: str(uuid4()))

    def average_start(self) -> float:
        if not self.spans:
            return 0.0
        return sum(span.start for span in self.spans) / len(self.spans)

    def average_stop(self) -> float:
        if not self.spans:
            return 0.0
        return sum(span.stop for span in self.spans) / len(self.spans)

    def span_for_message(self, message_index: int) -> Span | None:
        for span in self.spans:
            if span.message_index == message_index:
                return span
        return None

    def overlaps(self, other: "Constraint") -> bool:
        for span in self.spans:
            other_span = other.span_for_message(span.message_index)
            if other_span is not None and span.overlaps(other_span):
                return True
        return False

    def contains(self, other: "Constraint") -> bool:
        matched = False
        for span in self.spans:
            other_span = other.span_for_message(span.message_index)
            if other_span is None:
                continue
            matched = True
            if not span.contains(other_span):
                return False
        return matched


@dataclass(slots=True)
class HeuristicEvidence:
    constraints: list[Constraint]
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BoundaryPrior:
    source: str
    scores: list[float]
    binary_boundaries: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = [min(1.0, max(0.0, float(score))) for score in self.scores]
        if not self.binary_boundaries:
            self.binary_boundaries = [index for index, score in enumerate(self.scores) if score >= 0.5]

    def score_at(self, position: int) -> float:
        if 0 <= position < len(self.scores):
            return self.scores[position]
        return 0.0

    def candidate_positions(self, threshold: float = 0.5) -> list[int]:
        return [index for index, score in enumerate(self.scores) if score >= threshold]


@dataclass(slots=True)
class StructureNode:
    name: str
    kind: str
    spans: list[Span]
    source: str
    confidence: float = 0.0
    children: list["StructureNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructureHypothesis:
    root: StructureNode
    selected_constraints: list[Constraint]
    unresolved_regions: list[Constraint] = field(default_factory=list)
    boundary_prior: BoundaryPrior | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
