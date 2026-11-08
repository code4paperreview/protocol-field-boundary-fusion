from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from ..common.messages import MessageBatch
from ..common.models import Constraint, ConstraintKind, ConstraintStrength, HeuristicEvidence, Span
from .detectors import (
    deduplicate_constraints,
    detect_constant_runs,
    detect_counted_repetition,
    detect_length_fields,
    detect_tlv_star_structures,
    detect_tlv_structures,
)
from .semantic_detectors import (
    detect_entropy_delimiters,
    detect_identifier_fields,
    detect_payload_regions,
    detect_sequence_fields,
    detect_timestamp_fields,
)


class StandaloneHeuristicEngine:
    def __init__(
        self,
        detector_names: Iterable[str] | None = None,
        header_scan_limit: int = 8,
        recursive_header_scan_limit: int = 80,
        boundary_threshold: float = 0.6,
        max_recursion_depth: int = 2,
        recursive_min_span_width: int = 3,
    ) -> None:
        self.detector_names = list(
            detector_names
            or ("length", "tlv", "repetition", "constant", "identifier", "delimiter", "payload")
        )
        self.header_scan_limit = header_scan_limit
        self.recursive_header_scan_limit = recursive_header_scan_limit
        self.boundary_threshold = boundary_threshold
        self.max_recursion_depth = max_recursion_depth
        self.recursive_min_span_width = recursive_min_span_width

    def infer(self, batch: MessageBatch) -> HeuristicEvidence:
        constraints, raw_outputs, stats = self._infer_recursive(batch, depth=0, scope_name=batch.group_label or "root")
        return HeuristicEvidence(
            constraints=constraints,
            raw_outputs=raw_outputs,
            metadata={
                "engine": "standalone_heuristic_v2",
                "variable_length_supported": True,
                "message_lengths": batch.byte_lengths,
                "detectors": list(self.detector_names),
                "header_scan_limit": self.header_scan_limit,
                "recursive_header_scan_limit": self.recursive_header_scan_limit,
                "max_recursion_depth": self.max_recursion_depth,
                "recursive_min_span_width": self.recursive_min_span_width,
                **stats,
            },
        )

    def infer_flat_boundaries(self, batch: MessageBatch) -> list[list[int]]:
        evidence = self.infer(batch)
        boundaries: list[list[int]] = []
        for message_index, message_length in enumerate(batch.byte_lengths):
            offsets = {0, message_length}
            for constraint in evidence.constraints:
                span = constraint.span_for_message(message_index)
                if span is None:
                    continue
                if constraint.strength == ConstraintStrength.HARD or constraint.confidence >= self.boundary_threshold:
                    offsets.add(span.start)
                    offsets.add(span.stop)
            boundaries.append(sorted(offsets))
        return boundaries

    def _infer_recursive(
        self,
        batch: MessageBatch,
        depth: int,
        scope_name: str,
    ) -> tuple[list[Constraint], dict[str, int], dict[str, Any]]:
        current_level_constraints = self._run_detectors(batch, depth=depth)
        raw_outputs = self._count_by_source(current_level_constraints)
        recursive_constraints: list[Constraint] = []
        recursive_region_count = 0
        scope_count = 1
        max_depth_reached = depth

        if depth < self.max_recursion_depth:
            candidates = self._collect_recursive_candidates(current_level_constraints)
            candidates = [
                candidate
                for candidate in candidates
                if self._candidate_covers_all_messages(candidate, batch.num_messages)
            ]
            recursive_region_count = len(candidates)
            for candidate in candidates:
                child_batch = batch.slice_by_spans(
                    candidate.spans,
                    group_label=f"{scope_name}:{candidate.name}",
                    metadata={
                        "parent_scope": scope_name,
                        "parent_constraint": candidate.name,
                        "recursion_depth": depth + 1,
                    },
                )
                child_constraints, child_raw_outputs, child_stats = self._infer_recursive(
                    child_batch,
                    depth=depth + 1,
                    scope_name=f"{scope_name}/{candidate.name}",
                )
                recursive_constraints.extend(
                    self._shift_constraints(
                        child_constraints,
                        candidate.spans,
                        parent_constraint=candidate,
                        scope_name=scope_name,
                        depth=depth + 1,
                    )
                )
                self._merge_counts(raw_outputs, child_raw_outputs)
                scope_count += int(child_stats.get("scope_count", 0))
                max_depth_reached = max(max_depth_reached, int(child_stats.get("max_depth_reached", depth + 1)))

        combined = deduplicate_constraints(current_level_constraints + recursive_constraints)
        return combined, raw_outputs, {
            "scope_count": scope_count,
            "recursive_region_count": recursive_region_count,
            "max_depth_reached": max_depth_reached,
        }

    def _run_detectors(self, batch: MessageBatch, depth: int = 0) -> list[Constraint]:
        constraints: list[Constraint] = []
        structure_scan_limit = self._scan_limit_for_depth(batch, depth)
        semantic_scan_limit = max(structure_scan_limit * 3, 24)

        if "length" in self.detector_names:
            constraints.extend(detect_length_fields(batch, offset_scan_limit=structure_scan_limit))

        if "tlv" in self.detector_names:
            tlv_constraints = detect_tlv_structures(batch, offset_scan_limit=structure_scan_limit)
            constraints.extend(tlv_constraints)
            preferred_tlv_endian = self._dominant_tlv_endian(tlv_constraints)
            constraints.extend(
                detect_tlv_star_structures(
                    batch,
                    offset_scan_limit=structure_scan_limit,
                    preferred_endian=preferred_tlv_endian,
                )
            )

        if "repetition" in self.detector_names:
            constraints.extend(detect_counted_repetition(batch, offset_scan_limit=structure_scan_limit))

        if "constant" in self.detector_names:
            constraints.extend(detect_constant_runs(batch))

        if "sequence" in self.detector_names:
            constraints.extend(detect_sequence_fields(batch, offset_scan_limit=structure_scan_limit + 2))

        if "identifier" in self.detector_names:
            constraints.extend(detect_identifier_fields(batch, offset_scan_limit=structure_scan_limit + 2))

        if "timestamp" in self.detector_names:
            constraints.extend(detect_timestamp_fields(batch, offset_scan_limit=structure_scan_limit + 4))

        if "delimiter" in self.detector_names:
            constraints.extend(detect_entropy_delimiters(batch, offset_scan_limit=semantic_scan_limit))

        if "payload" in self.detector_names:
            constraints.extend(detect_payload_regions(batch, offset_scan_limit=semantic_scan_limit))

        return deduplicate_constraints(constraints)

    def _dominant_tlv_endian(self, constraints: list[Constraint]) -> str | None:
        scores: dict[str, float] = {}
        for constraint in constraints:
            if constraint.kind != ConstraintKind.TLV:
                continue
            rule_annotation = str(constraint.metadata.get("rule", "")).strip().lower()
            if rule_annotation.endswith("_big") or rule_annotation.endswith("_big*"):
                endian = "big"
            elif rule_annotation.endswith("_little") or rule_annotation.endswith("_little*"):
                endian = "little"
            else:
                continue
            average_width = (
                sum(span.width for span in constraint.spans) / len(constraint.spans)
                if constraint.spans
                else 1.0
            )
            scores[endian] = scores.get(endian, 0.0) + constraint.confidence * max(1.0, average_width)
        if not scores:
            return None
        return max(sorted(scores), key=lambda endian: scores[endian])

    def _collect_recursive_candidates(self, constraints: list[Constraint]) -> list[Constraint]:
        raw_candidates: list[Constraint] = []
        for constraint in constraints:
            if not bool(constraint.metadata.get("recursive_candidate")):
                continue
            if not constraint.spans:
                continue
            if any(span.width < self.recursive_min_span_width for span in constraint.spans):
                continue
            semantic_role = str(constraint.metadata.get("semantic_role", ""))
            if semantic_role not in {"tlv_v", "rep_item", "rep_body"}:
                continue
            raw_candidates.append(constraint)
        if not raw_candidates:
            return []
        raw_candidates.sort(
            key=lambda c: (
                -self._constraint_average_width(c),
                -c.metadata.get("support_ratio", 0),
                -c.confidence,
                c.average_start(),
                c.average_stop(),
            ),
        )
        selected: list[Constraint] = []
        for candidate in raw_candidates:
            if any(self._candidates_overlap(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
        return sorted(selected, key=self._constraint_average_width, reverse=True)

    @staticmethod
    def _candidates_overlap(left: Constraint, right: Constraint) -> bool:
        for left_span in left.spans:
            right_span = right.span_for_message(left_span.message_index)
            if right_span is None:
                continue
            overlap_start = max(left_span.start, right_span.start)
            overlap_end = min(left_span.stop, right_span.stop)
            if overlap_end > overlap_start:
                min_width = min(left_span.width, right_span.width)
                if min_width > 0 and (overlap_end - overlap_start) / min_width > 0.5:
                    return True
        return False

    def _shift_constraints(
        self,
        constraints: list[Constraint],
        parent_spans: list[Span],
        parent_constraint: Constraint,
        scope_name: str,
        depth: int,
    ) -> list[Constraint]:
        parent_span_map = {span.message_index: span for span in parent_spans}
        shifted_constraints: list[Constraint] = []

        for constraint in constraints:
            shifted_spans: list[Span] = []
            for span in constraint.spans:
                parent_span = parent_span_map.get(span.message_index)
                if parent_span is None:
                    shifted_spans = []
                    break
                shifted_spans.append(
                    Span(
                        message_index=span.message_index,
                        start=parent_span.start + span.start,
                        stop=parent_span.start + span.stop,
                        type_code=span.type_code,
                    )
                )
            if len(shifted_spans) != len(constraint.spans):
                continue

            metadata = dict(constraint.metadata)
            metadata.update(
                {
                    "recursion_depth": depth,
                    "parent_scope": scope_name,
                    "recursive_parent": parent_constraint.name,
                    "recursive_derived": True,
                }
            )
            shifted_constraints.append(
                replace(
                    constraint,
                    name=f"{constraint.name}@d{depth}:{parent_constraint.name}",
                    spans=shifted_spans,
                    confidence=min(1.0, constraint.confidence + 0.02 * depth),
                    metadata=metadata,
                )
            )

        return shifted_constraints

    def _count_by_source(self, constraints: list[Constraint]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for constraint in constraints:
            source_name = constraint.source.split(":")[-1]
            counts[source_name] = counts.get(source_name, 0) + 1
        return counts

    def _merge_counts(self, target: dict[str, int], source: dict[str, int]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0) + value

    def _constraint_average_width(self, constraint: Constraint) -> float:
        if not constraint.spans:
            return 0.0
        return sum(span.width for span in constraint.spans) / len(constraint.spans)

    @staticmethod
    def _candidate_covers_all_messages(constraint: Constraint, message_count: int) -> bool:
        if len(constraint.spans) != message_count:
            return False
        message_indexes = {span.message_index for span in constraint.spans}
        return len(message_indexes) == message_count and all(index in message_indexes for index in range(message_count))

    def _scan_limit_for_depth(self, batch: MessageBatch, depth: int) -> int:
        base_limit = self.header_scan_limit if depth <= 0 else max(self.header_scan_limit, self.recursive_header_scan_limit)
        return min(batch.min_length, base_limit)
