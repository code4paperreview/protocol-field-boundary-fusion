from __future__ import annotations

from collections import defaultdict

from ..common.messages import MessageBatch
from ..common.models import BoundaryPrior, Constraint, ConstraintKind


class ConstrainedBoundaryDecoder:
    def __init__(
        self,
        neural_weight: float = 1.15,
        heuristic_boundary_weight: float = 0.85,
        endpoint_bonus_cap: float = 2.4,
        exact_span_weight: float = 0.25,
        candidate_threshold: float = 0.18,
        length_guard_min_confidence: float = 0.75,
        length_guard_max_width: int = 2,
        length_guard_prior_gate: float = 0.50,
        length_guard_penalty: float = 2.0,
        length_guard_region: str = "header",
        long_span_penalty: float = 0.15,
    ) -> None:
        self.neural_weight = neural_weight
        self.heuristic_boundary_weight = heuristic_boundary_weight
        self.endpoint_bonus_cap = endpoint_bonus_cap
        self.exact_span_weight = exact_span_weight
        self.candidate_threshold = candidate_threshold
        self.length_guard_min_confidence = length_guard_min_confidence
        self.length_guard_max_width = length_guard_max_width
        self.length_guard_prior_gate = length_guard_prior_gate
        self.length_guard_penalty = length_guard_penalty
        self.length_guard_region = length_guard_region
        self.long_span_penalty = long_span_penalty

    def decode_boundaries(
        self,
        batch: MessageBatch,
        constraints: list[Constraint],
        boundary_prior: BoundaryPrior,
        heuristic_boundaries: list[list[int]] | None = None,
    ) -> list[list[int]]:
        result: list[list[int]] = []
        for message_index, message_len in enumerate(batch.byte_lengths):
            h_boundaries = set()
            if heuristic_boundaries is not None and message_index < len(heuristic_boundaries):
                h_boundaries = {p for p in heuristic_boundaries[message_index] if 0 <= p <= message_len}
            potentials = self._build_potentials(message_index, message_len, constraints, boundary_prior, h_boundaries)
            result.append(self._decode_one(message_len, potentials))
        return result

    def _build_potentials(
        self,
        message_index: int,
        message_len: int,
        constraints: list[Constraint],
        boundary_prior: BoundaryPrior,
        heuristic_boundaries: set[int],
    ) -> dict[str, object]:
        boundary_bonus: dict[int, float] = {}
        split_penalty: dict[int, float] = {}
        exact_span_bonus: dict[tuple[int, int], float] = {}
        candidates = {0, message_len}

        for pos in range(1, message_len):
            score = boundary_prior.score_at(pos)
            if score >= self.candidate_threshold:
                candidates.add(pos)
            if score > 0.0:
                centered_score = 2.0 * score - 1.0
                boundary_bonus[pos] = boundary_bonus.get(pos, 0.0) + self.neural_weight * centered_score

        for pos in heuristic_boundaries:
            if 0 < pos < message_len:
                candidates.add(pos)
                if self.heuristic_boundary_weight > 0.0:
                    boundary_bonus[pos] = boundary_bonus.get(pos, 0.0) + self.heuristic_boundary_weight

        endpoint_by_pos: dict[int, dict[str, float]] = defaultdict(dict)
        interval_by_span: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)

        for constraint in constraints:
            role = self._role_key(constraint)
            detector = self._detector_key(constraint)
            span = constraint.span_for_message(message_index)
            if span is None:
                continue
            start = max(0, min(message_len, span.start))
            stop = max(0, min(message_len, span.stop))
            if start > stop:
                start, stop = stop, start

            role_bonus = self._role_boundary_bonus(constraint, role) * constraint.confidence
            for pos in (start, stop):
                if 0 < pos < message_len:
                    candidates.add(pos)
                    endpoint_by_pos[pos][detector] = max(endpoint_by_pos[pos].get(detector, 0.0), role_bonus)

            if stop > start:
                span_bonus = self._role_span_bonus(constraint, role) * constraint.confidence
                interval_by_span[(start, stop)][detector] = max(
                    interval_by_span[(start, stop)].get(detector, 0.0), span_bonus
                )

        for pos, by_detector in endpoint_by_pos.items():
            boundary_bonus[pos] = boundary_bonus.get(pos, 0.0) + min(
                sum(by_detector.values()), self.endpoint_bonus_cap
            )

        for span, by_detector in interval_by_span.items():
            exact_span_bonus[span] = min(sum(by_detector.values()), self.endpoint_bonus_cap)

        if self.length_guard_penalty > 0.0:
            for constraint in constraints:
                if constraint.confidence < self.length_guard_min_confidence:
                    continue
                role = self._role_key(constraint)
                if constraint.kind != ConstraintKind.LENGTH and role != "length_field":
                    continue
                region = str(constraint.metadata.get("hp_region", "")).strip().lower()
                if self.length_guard_region and self.length_guard_region != "all" and region:
                    if region != self.length_guard_region:
                        continue
                span = constraint.span_for_message(message_index)
                if span is None:
                    continue
                start = max(0, min(message_len, span.start))
                stop = max(0, min(message_len, span.stop))
                if start > stop:
                    start, stop = stop, start
                width = stop - start
                if width <= 1:
                    continue
                if self.length_guard_max_width > 0 and width > self.length_guard_max_width:
                    continue
                for pos in range(start + 1, stop):
                    if 0 < pos < message_len and boundary_prior.score_at(pos) < self.length_guard_prior_gate:
                        split_penalty[pos] = max(
                            split_penalty.get(pos, 0.0), self.length_guard_penalty * constraint.confidence
                        )

        return {
            "candidates": sorted(candidates),
            "boundary_bonus": boundary_bonus,
            "split_penalty": split_penalty,
            "exact_span_bonus": exact_span_bonus,
        }

    def _decode_one(self, message_len: int, potentials: dict[str, object]) -> list[int]:
        candidates = potentials["candidates"]
        boundary_bonus = potentials["boundary_bonus"]
        split_penalty = potentials["split_penalty"]
        exact_span_bonus = potentials["exact_span_bonus"]
        assert isinstance(candidates, list)
        assert isinstance(boundary_bonus, dict)
        assert isinstance(split_penalty, dict)
        assert isinstance(exact_span_bonus, dict)

        dp: dict[int, float] = {0: 0.0}
        prev: dict[int, int | None] = {0: None}
        for j in candidates[1:]:
            best_score = -1e18
            best_prev: int | None = None
            for i in candidates:
                if i >= j or i not in dp:
                    continue
                score = dp[i] + self._interval_score(i, j, message_len, exact_span_bonus)
                if j != message_len:
                    score += boundary_bonus.get(j, 0.0)
                    score -= split_penalty.get(j, 0.0)
                if score > best_score:
                    best_score = score
                    best_prev = i
            if best_prev is not None:
                dp[j] = best_score
                prev[j] = best_prev

        if message_len not in prev:
            return [0, message_len]

        path: list[int] = []
        cursor: int | None = message_len
        while cursor is not None:
            path.append(cursor)
            cursor = prev.get(cursor)
        path.reverse()
        if path[0] != 0:
            path.insert(0, 0)
        if path[-1] != message_len:
            path.append(message_len)
        return sorted(set(path))

    def _interval_score(
        self,
        start: int,
        stop: int,
        message_len: int,
        exact_span_bonus: dict[tuple[int, int], float],
    ) -> float:
        width = stop - start
        if width <= 0:
            return -1e18
        score = exact_span_bonus.get((start, stop), 0.0) * self.exact_span_weight
        if message_len > 0 and width > message_len * 0.75 and start > 0 and stop < message_len:
            score -= self.long_span_penalty
        return score

    @staticmethod
    def _detector_key(constraint: Constraint) -> str:
        source = str(constraint.source or "").strip().lower()
        role = str(constraint.metadata.get("semantic_role", "")).strip().lower()
        if source.startswith("native_heuristic:"):
            source = source.split(":", 1)[1]
        if source:
            return source
        if role:
            return role
        return str(constraint.kind.value)

    @staticmethod
    def _role_key(constraint: Constraint) -> str:
        role = str(constraint.metadata.get("semantic_role", "")).strip().lower()
        return role or str(constraint.kind.value)

    def _role_boundary_bonus(self, constraint: Constraint, role: str) -> float:
        if constraint.kind in {ConstraintKind.LENGTH, ConstraintKind.TLV, ConstraintKind.REPETITION}:
            return 1.10
        if role in {"delimiter", "sequence_counter", "identifier", "timestamp"}:
            return 0.75
        if role in {"compressed_payload", "encrypted_payload"}:
            return 0.55
        if constraint.kind == ConstraintKind.CONSTANT:
            return 0.65
        return 0.35

    def _role_span_bonus(self, constraint: Constraint, role: str) -> float:
        if role in {
            "length_field",
            "tlv_t",
            "tlv_l",
            "tlv_q",
            "tlv_star_t",
            "tlv_star_l",
            "tlv_star_q",
            "tlv_prefix_t",
            "tlv_prefix_l",
            "tlv_prefix_q",
            "rep_q",
            "rep_l",
            "rep_vector_l",
        }:
            return 1.00
        if role in {"sequence_counter", "identifier", "timestamp"}:
            return 0.70
        if role in {"compressed_payload", "encrypted_payload"}:
            return 0.55
        if constraint.kind == ConstraintKind.CONSTANT:
            return 0.60
        return 0.20
