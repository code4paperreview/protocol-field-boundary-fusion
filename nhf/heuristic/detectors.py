from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterable

from ..common.messages import MessageBatch
from ..common.models import Constraint, ConstraintKind, ConstraintStrength, Span
from .rules import ContainerRule, iter_container_rules, iter_tlv_star_rules, parse_container_rule


@dataclass(slots=True)
class ParsedContainer:
    segments: dict[str, tuple[int, int]]
    item_count: int
    end_offset: int


@dataclass(slots=True)
class ParsedRepetition:
    segments: dict[str, tuple[int, int]]
    item_count: int
    item_width: int
    end_offset: int


@dataclass(slots=True)
class ParsedLengthPrefixedRegion:
    segments: dict[str, tuple[int, int]]
    body_length: int
    end_offset: int


def detect_constant_runs(batch: MessageBatch, min_run_width: int = 2) -> list[Constraint]:
    if batch.min_length < min_run_width:
        return []

    messages = batch.to_bytes_list()
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for offset in range(batch.min_length):
        column = {message[offset] for message in messages}
        if len(column) == 1:
            if run_start is None:
                run_start = offset
        else:
            if run_start is not None and offset - run_start >= min_run_width:
                runs.append((run_start, offset))
            run_start = None

    if run_start is not None and batch.min_length - run_start >= min_run_width:
        runs.append((run_start, batch.min_length))

    constraints: list[Constraint] = []
    for index, (start, stop) in enumerate(runs):
        shared_value = messages[0][start:stop].hex()
        spans = [Span(message_index=i, start=start, stop=stop, type_code="A") for i in range(batch.num_messages)]
        confidence = min(0.85, 0.55 + 0.05 * (stop - start))
        constraints.append(
            Constraint(
                name=f"constant_run:{index}",
                kind=ConstraintKind.CONSTANT,
                strength=ConstraintStrength.SOFT,
                source="native_heuristic:constant_run",
                spans=spans,
                annotation=f"Constant:{shared_value}",
                type_code="A",
                confidence=confidence,
                metadata={"value_hex": shared_value},
            )
        )
    return constraints


def detect_length_fields(
    batch: MessageBatch,
    widths: Iterable[int] = (1, 2, 3, 4),
    endians: Iterable[str] = ("big", "little"),
    offset_scan_limit: int = 8,
    soft_support_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 2:
        return []

    messages = batch.to_bytes_list()
    lengths = batch.byte_lengths
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)
    support_threshold = max(2, math.ceil(batch.num_messages * soft_support_ratio))

    for offset in range(max_offset):
        for width in widths:
            if offset + width > batch.min_length:
                continue
            for endian in endians:
                values: list[int] = []
                diffs: list[int] = []
                for message_index, message in enumerate(messages):
                    value = _read_uint(message, offset, width, endian)
                    if value is None:
                        values = []
                        break
                    values.append(value)
                    diffs.append(lengths[message_index] - value)
                if len(values) != batch.num_messages:
                    continue
                if len(set(values)) <= 1:
                    continue
                if any(diff < 0 for diff in diffs):
                    continue

                dominant_diff, dominant_count = Counter(diffs).most_common(1)[0]
                if dominant_count == batch.num_messages:
                    confidence = max(0.86, 0.98 - 0.03 * (width - 1))
                    strength = ConstraintStrength.HARD
                    annotation = f"LengthField[{width}][{endian}]=>total_length-{dominant_diff}"
                    name = f"length:{offset}:{width}:{endian}"
                    metadata = {
                        "offset": offset,
                        "width": width,
                        "endian": endian,
                        "offset_to_end": dominant_diff,
                        "values": values,
                        "mismatch_rows": [],
                        "fuzzy_length": False,
                    }
                elif dominant_count >= support_threshold:
                    confidence = 0.72
                    strength = ConstraintStrength.SOFT
                    annotation = f"ApproxLengthField[{width}][{endian}]=>total_length-{dominant_diff}"
                    name = f"length:{offset}:{width}:{endian}"
                    metadata = {
                        "offset": offset,
                        "width": width,
                        "endian": endian,
                        "offset_to_end": dominant_diff,
                        "values": values,
                        "mismatch_rows": [index for index, diff in enumerate(diffs) if diff != dominant_diff],
                        "fuzzy_length": False,
                    }
                else:
                    fuzzy_profile = _strict_fuzzy_length_profile(values, lengths)
                    if fuzzy_profile is None:
                        continue
                    confidence = min(0.69, 0.64 + 0.04 * max(0.0, fuzzy_profile["correlation"] - 0.985) / 0.015)
                    strength = ConstraintStrength.SOFT
                    annotation = (
                        f"FuzzyLengthField[{width}][{endian}]=>length≈"
                        f"{fuzzy_profile['slope']:.3f}*field+{fuzzy_profile['intercept']:.3f}"
                    )
                    name = f"length_fuzzy:{offset}:{width}:{endian}"
                    metadata = {
                        "offset": offset,
                        "width": width,
                        "endian": endian,
                        "offset_to_end": None,
                        "values": values,
                        "mismatch_rows": list(fuzzy_profile["mismatch_rows"]),
                        "fuzzy_length": True,
                        "fit_correlation": round(float(fuzzy_profile["correlation"]), 6),
                        "fit_slope": round(float(fuzzy_profile["slope"]), 6),
                        "fit_intercept": round(float(fuzzy_profile["intercept"]), 6),
                        "mean_abs_residual": round(float(fuzzy_profile["mean_abs_residual"]), 6),
                    }

                spans = [Span(message_index=i, start=offset, stop=offset + width, type_code="L") for i in range(batch.num_messages)]
                constraints.append(
                    Constraint(
                        name=name,
                        kind=ConstraintKind.LENGTH,
                        strength=strength,
                        source="native_heuristic:length_relation",
                        spans=spans,
                        annotation=annotation,
                        type_code="L",
                        confidence=confidence,
                        metadata=metadata,
                    )
                )

    return deduplicate_constraints(constraints)


def _pearson_correlation(xs: list[int], ys: list[int]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered = [(x - mean_x, y - mean_y) for x, y in zip(xs, ys)]
    numerator = sum(dx * dy for dx, dy in centered)
    denominator_x = sum(dx * dx for dx, _ in centered)
    denominator_y = sum(dy * dy for _, dy in centered)
    if denominator_x <= 0.0 or denominator_y <= 0.0:
        return 0.0
    return numerator / math.sqrt(denominator_x * denominator_y)


def _linear_fit(xs: list[int], ys: list[int]) -> tuple[float, float] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered = [(x - mean_x, y - mean_y) for x, y in zip(xs, ys)]
    denominator_x = sum(dx * dx for dx, _ in centered)
    if denominator_x <= 0.0:
        return None
    numerator = sum(dx * dy for dx, dy in centered)
    slope = numerator / denominator_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _strict_fuzzy_length_profile(values: list[int], lengths: list[int]) -> dict[str, Any] | None:
    if len(values) != len(lengths) or len(values) < 5:
        return None
    distinct_threshold = min(4, len(values))
    if len(set(values)) < distinct_threshold or len(set(lengths)) < distinct_threshold:
        return None
    if any(value < 0 or value > length for value, length in zip(values, lengths)):
        return None
    correlation = _pearson_correlation(values, lengths)
    if correlation < 0.985:
        return None
    fit = _linear_fit(values, lengths)
    if fit is None:
        return None
    slope, intercept = fit
    average_length = sum(lengths) / len(lengths)
    if not 0.9 <= slope <= 1.1:
        return None
    if abs(intercept) > max(16.0, 0.2 * average_length):
        return None
    residuals = [length - (slope * value + intercept) for value, length in zip(values, lengths)]
    max_abs_residual = max(abs(residual) for residual in residuals)
    mean_abs_residual = sum(abs(residual) for residual in residuals) / len(residuals)
    if max_abs_residual > 1.25 or mean_abs_residual > 0.45:
        return None
    mismatch_rows = [index for index, residual in enumerate(residuals) if abs(residual) > 0.5]
    return {
        "correlation": correlation,
        "slope": slope,
        "intercept": intercept,
        "mean_abs_residual": mean_abs_residual,
        "mismatch_rows": mismatch_rows,
    }


def detect_tlv_structures(
    batch: MessageBatch,
    offset_scan_limit: int = 8,
    q_widths: Iterable[int] = (0, 1, 2, 3),
    t_widths: Iterable[int] = (0, 1, 2, 3),
    l_widths: Iterable[int] = (1, 2, 3),
    endians: Iterable[str] = ("big", "little"),
    max_items: int = 32,
    soft_support_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 2:
        return []

    messages = batch.to_bytes_list()
    rules = iter_container_rules(q_widths=q_widths, t_widths=t_widths, l_widths=l_widths, endians=endians)
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)
    support_threshold = max(2, math.ceil(batch.num_messages * soft_support_ratio))

    for offset in range(max_offset):
        for rule in rules:
            parsed_by_index: dict[int, ParsedContainer] = {}
            for message_index, message in enumerate(messages):
                parsed = _parse_container(message, offset, rule, max_items)
                if parsed is not None:
                    parsed_by_index[message_index] = parsed
            if len(parsed_by_index) < support_threshold:
                continue
            if not _passes_container_batch_filters(messages, parsed_by_index, rule):
                continue
            constraints.extend(_constraints_from_parsed(batch, offset, rule, parsed_by_index))

    dominant_endian = _select_container_endian(constraints)
    if dominant_endian is not None:
        constraints = [constraint for constraint in constraints if _container_constraint_endian(constraint) == dominant_endian]
    return deduplicate_constraints(constraints)


def detect_tlv_star_structures(
    batch: MessageBatch,
    offset_scan_limit: int = 8,
    preferred_endian: str | None = None,
    max_items: int = 32,
    soft_support_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 2:
        return []

    messages = batch.to_bytes_list()
    rules = iter_tlv_star_rules()
    if preferred_endian is not None:
        rules = [rule for rule in rules if rule.endian == preferred_endian]
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)
    support_threshold = max(2, math.ceil(batch.num_messages * soft_support_ratio))

    for offset in range(max_offset):
        for rule in rules:
            star_states_by_index: dict[int, dict[int, ParsedContainer]] = {}
            suffix_support: dict[int, list[int]] = {}
            for message_index, message in enumerate(messages):
                parsed_by_suffix = _parse_tlv_star_states(message, offset, rule, max_items)
                if not parsed_by_suffix:
                    continue
                star_states_by_index[message_index] = parsed_by_suffix
                for suffix_width in parsed_by_suffix:
                    suffix_support.setdefault(suffix_width, []).append(message_index)
            chosen_suffix_width = _select_tlv_star_suffix(suffix_support, support_threshold)
            if chosen_suffix_width is None:
                continue
            support_rows = sorted(set(suffix_support[chosen_suffix_width]))
            option_counts = [len(star_states_by_index[index]) for index in support_rows]
            if len(set(option_counts)) == 1:
                continue
            trimmed_counts = [
                sum(1 for suffix_width in star_states_by_index[index] if suffix_width >= chosen_suffix_width)
                for index in support_rows
            ]
            if len(set(trimmed_counts)) == 1:
                continue
            parsed_by_index = {
                index: star_states_by_index[index][chosen_suffix_width]
                for index in support_rows
            }
            if not _passes_container_batch_filters(messages, parsed_by_index, rule):
                continue
            constraints.extend(
                _constraints_from_tlv_star_parsed(
                    batch,
                    offset,
                    rule,
                    chosen_suffix_width,
                    parsed_by_index,
                )
            )

    if preferred_endian is None:
        dominant_endian = _select_container_endian(constraints)
        if dominant_endian is not None:
            constraints = [constraint for constraint in constraints if _container_constraint_endian(constraint) == dominant_endian]
    return deduplicate_constraints(constraints)


def detect_counted_repetition(
    batch: MessageBatch,
    offset_scan_limit: int = 6,
    count_widths: Iterable[int] = (1, 2, 3, 4),
    count_endians: Iterable[str] = ("big", "little"),
    min_item_width: int = 2,
    max_item_width: int = 24,
    max_items: int = 32,
    soft_support_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 2:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)
    support_threshold = max(2, math.ceil(batch.num_messages * soft_support_ratio))

    for offset in range(max_offset):
        for count_width in count_widths:
            for count_endian in count_endians:
                parsed_by_index: dict[int, ParsedRepetition] = {}
                for message_index, message in enumerate(messages):
                    parsed = _parse_counted_repetition(
                        message,
                        offset,
                        count_width,
                        count_endian,
                        min_item_width=min_item_width,
                        max_item_width=max_item_width,
                        max_items=max_items,
                    )
                    if parsed is not None:
                        parsed_by_index[message_index] = parsed
                if len(parsed_by_index) < support_threshold:
                    continue
                if len({parsed.item_width for parsed in parsed_by_index.values()}) != 1:
                    continue
                if not _has_similar_repetition_items(messages, parsed_by_index, offset, count_width):
                    continue
                constraints.extend(
                    _constraints_from_repetition(
                        batch,
                        offset,
                        count_width,
                        count_endian,
                        parsed_by_index,
                        source="native_heuristic:counted_repetition",
                    )
                )

    return deduplicate_constraints(constraints)


def detect_length_prefixed_vectors(
    batch: MessageBatch,
    offset_scan_limit: int = 8,
    length_widths: Iterable[int] = (1, 2, 3, 4),
    length_endians: Iterable[str] = ("big", "little"),
    min_body_width: int = 4,
    soft_support_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 2:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)
    support_threshold = max(2, math.ceil(batch.num_messages * soft_support_ratio))
    allow_constant_body_lengths = int(batch.metadata.get("recursion_depth", 0)) > 0

    for offset in range(max_offset):
        for length_width in length_widths:
            for length_endian in length_endians:
                parsed_by_index: dict[int, ParsedLengthPrefixedRegion] = {}
                for message_index, message in enumerate(messages):
                    parsed = _parse_length_prefixed_region(
                        message,
                        offset,
                        length_width,
                        length_endian,
                        min_body_width=min_body_width,
                    )
                    if parsed is not None:
                        parsed_by_index[message_index] = parsed
                if len(parsed_by_index) < support_threshold:
                    continue
                if len({parsed.body_length for parsed in parsed_by_index.values()}) <= 1 and not allow_constant_body_lengths:
                    continue
                constraints.extend(
                    _constraints_from_length_prefixed_region(
                        batch,
                        offset,
                        length_width,
                        length_endian,
                        parsed_by_index,
                    )
                )

    return deduplicate_constraints(constraints)


def deduplicate_constraints(constraints: list[Constraint]) -> list[Constraint]:
    deduplicated: dict[tuple[Any, ...], Constraint] = {}
    for constraint in constraints:
        key = (
            constraint.kind.value,
            constraint.type_code,
            constraint.annotation,
            constraint.metadata.get("semantic_role"),
            tuple((span.message_index, span.start, span.stop, span.type_code) for span in constraint.spans),
            constraint.metadata.get("rule_family_id"),  # Ensure different rules don't incorrectly overwrite each other
        )
        current = deduplicated.get(key)
        if current is None:
            deduplicated[key] = constraint
            continue
        if _constraint_rank(constraint) > _constraint_rank(current):
            deduplicated[key] = constraint
            
    surviving_constraints = sorted(
        deduplicated.values(),
        key=lambda constraint: (
            0 if constraint.strength == ConstraintStrength.HARD else 1,
            constraint.average_start(),
            constraint.average_stop(),
            -constraint.confidence,
        ),
    )
    
    # Drop orphaned children when their container is absent.
    valid_family_ids = {
        c.metadata["rule_family_id"]
        for c in surviving_constraints
        if str(c.metadata.get("semantic_role", "")).endswith("_container")
        and "rule_family_id" in c.metadata
    }
    
    final_constraints: list[Constraint] = []
    for c in surviving_constraints:
        family_id = c.metadata.get("rule_family_id")
        role = str(c.metadata.get("semantic_role", ""))
        
        # Drop this child if its container was removed.
        if family_id and role.startswith(("tlv_", "rep_")) and not role.endswith("_container"):
            if family_id not in valid_family_ids:
                continue
                
        final_constraints.append(c)
        
    return final_constraints


def _constraint_rank(constraint: Constraint) -> tuple[int, float]:
    return (1 if constraint.strength == ConstraintStrength.HARD else 0, constraint.confidence)


def _read_uint(message: bytes, offset: int, width: int, endian: str) -> int | None:
    stop = offset + width
    if offset < 0 or stop > len(message):
        return None
    return int.from_bytes(message[offset:stop], byteorder=endian, signed=False)


def _parse_container(message: bytes, offset: int, rule: ContainerRule, max_items: int) -> ParsedContainer | None:
    if offset < 0 or offset + rule.header_width > len(message):
        return None

    pos = offset
    segments: dict[str, tuple[int, int]] = {}

    if rule.has_q:
        qty = _read_uint(message, pos, rule.q_width, rule.endian)
        if qty is None or qty <= 0 or qty > max_items:
            return None
        segments["TLV_Q"] = (pos, pos + rule.q_width)
        pos += rule.q_width
        item_count = qty
        for item_index in range(qty):
            parsed = _parse_single_tlv_item(message, pos, item_index, rule)
            if parsed is None:
                return None
            item_segments, pos = parsed
            segments.update(item_segments)
        if pos != len(message) and not rule.allow_suffix:
            return None
        if pos == len(message) and rule.allow_suffix:
            return None
        segments["TLV_CONTAINER"] = (offset, pos)
        return ParsedContainer(segments=segments, item_count=item_count, end_offset=pos)

    item_index = 0
    while pos < len(message):
        parsed = _parse_single_tlv_item(message, pos, item_index, rule)
        if parsed is None:
            if rule.allow_suffix and item_index > 0:
                break
            return None
        item_segments, pos = parsed
        segments.update(item_segments)
        item_index += 1
        if item_index > max_items:
            return None
    if item_index == 0:
        return None
    if rule.allow_suffix and pos == len(message):
        return None
    segments["TLV_CONTAINER"] = (offset, pos)
    return ParsedContainer(segments=segments, item_count=item_index, end_offset=pos)


def _parse_single_tlv_item(
    message: bytes,
    pos: int,
    item_index: int,
    rule: ContainerRule,
) -> tuple[dict[str, tuple[int, int]], int] | None:
    t_start = pos
    t_stop = t_start + rule.t_width
    l_start = t_stop
    l_stop = l_start + rule.l_width
    if l_stop > len(message):
        return None

    value_length = _read_uint(message, l_start, rule.l_width, rule.endian)
    if value_length is None or value_length <= 0:
        return None

    v_start = l_stop
    v_stop = v_start + value_length
    if v_stop > len(message):
        return None

    segments = {
        f"TLV_ITEM_{item_index}": (t_start, v_stop),
        f"TLV_T_{item_index}": (t_start, t_stop),
        f"TLV_L_{item_index}": (l_start, l_stop),
        f"TLV_V_{item_index}": (v_start, v_stop),
    }
    return segments, v_stop


def _parse_tlv_star_states(
    message: bytes,
    offset: int,
    rule: ContainerRule,
    max_items: int,
) -> dict[int, ParsedContainer]:
    if rule.has_q or not rule.allow_suffix or offset < 0 or offset + rule.header_width > len(message):
        return {}

    pos = offset
    item_index = 0
    segments: dict[str, tuple[int, int]] = {}
    parsed_by_suffix: dict[int, ParsedContainer] = {}

    while pos < len(message):
        parsed = _parse_single_tlv_item(message, pos, item_index, rule)
        if parsed is None:
            break
        item_segments, pos = parsed
        segments.update(item_segments)
        item_index += 1
        if item_index > max_items:
            return {}
        suffix_width = len(message) - pos
        if suffix_width <= 0:
            break
        candidate_segments = dict(segments)
        candidate_segments["TLV_CONTAINER"] = (offset, pos)
        parsed_by_suffix[suffix_width] = ParsedContainer(
            segments=candidate_segments,
            item_count=item_index,
            end_offset=pos,
        )
    return parsed_by_suffix


def _select_tlv_star_suffix(
    suffix_support: dict[int, list[int]],
    support_threshold: int,
) -> int | None:
    best_suffix_width: int | None = None
    best_support = 0
    for suffix_width, rows in sorted(suffix_support.items()):
        support_count = len(set(rows))
        if suffix_width <= 0 or support_count < support_threshold:
            continue
        if support_count > best_support:
            best_support = support_count
            best_suffix_width = suffix_width
            continue
        if support_count == best_support and best_suffix_width is not None and suffix_width < best_suffix_width:
            best_suffix_width = suffix_width
    return best_suffix_width


def _container_constraint_endian(constraint: Constraint) -> str | None:
    rule = parse_container_rule(str(constraint.metadata.get("rule", "")))
    if rule is None:
        return None
    return rule.endian


def _select_container_endian(constraints: list[Constraint]) -> str | None:
    scores: dict[str, float] = {}
    for constraint in constraints:
        endian = _container_constraint_endian(constraint)
        if endian is None or not constraint.spans:
            continue
        average_width = sum(span.width for span in constraint.spans) / len(constraint.spans)
        support = len(constraint.spans)
        hard_bonus = 8.0 if constraint.strength == ConstraintStrength.HARD else 0.0
        scores[endian] = scores.get(endian, 0.0) + support * average_width * (1.0 + constraint.confidence) + hard_bonus
    if not scores:
        return None
    return max(sorted(scores), key=lambda endian: scores[endian])


def _discrete_entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    total = float(len(values))
    return -sum((count / total) * math.log2(count / total) for count in Counter(values).values())


def _passes_container_batch_filters(
    messages: list[bytes],
    parsed_by_index: dict[int, ParsedContainer],
    rule: ContainerRule,
) -> bool:
    support_rows = sorted(parsed_by_index)
    if len(support_rows) <= 1:
        return True

    length_chunks: list[bytes] = []
    type_chunks: list[bytes] = []
    for message_index in support_rows:
        parsed = parsed_by_index[message_index]
        message = messages[message_index]
        l_span = parsed.segments.get("TLV_L_0")
        if l_span is None:
            return False
        length_chunks.append(message[l_span[0] : l_span[1]])
        if rule.t_width > 0:
            t_span = parsed.segments.get("TLV_T_0")
            if t_span is None:
                return False
            type_chunks.append(message[t_span[0] : t_span[1]])

    length_values = [int.from_bytes(chunk, byteorder=rule.endian, signed=False) for chunk in length_chunks]
    msg_lengths = {len(messages[index]) for index in support_rows}
    if len(set(length_values)) == 1 and len(msg_lengths) > 1:
        return False

    if not type_chunks:
        return True

    type_values = [int.from_bytes(chunk, byteorder="big", signed=False) for chunk in type_chunks]
    if all(value == 0 for value in type_values):
        return False
    if len(set(type_values)) == len(type_values):
        return False
    if _discrete_entropy(type_values) > 3.0:
        return False
    if all(chunk and chunk[-1] == 0 for chunk in type_chunks):
        return False
    return True


def _has_similar_repetition_items(
    messages: list[bytes],
    parsed_by_index: dict[int, ParsedRepetition],
    offset: int,
    count_width: int,
) -> bool:
    coherent_messages = 0
    evaluated_messages = 0
    for message_index, parsed in parsed_by_index.items():
        message = messages[message_index]
        body_start = offset + count_width
        items: list[bytes] = []
        for item_index in range(parsed.item_count):
            start = body_start + item_index * parsed.item_width
            stop = start + parsed.item_width
            items.append(message[start:stop])
        if len(items) < 2:
            continue
        evaluated_messages += 1
        if _items_share_repetition_family(items):
            coherent_messages += 1

    if evaluated_messages == 0:
        return False

    required_messages = 1 if evaluated_messages == 1 else max(2, math.ceil(evaluated_messages / 2))
    return coherent_messages >= required_messages


def _items_share_repetition_family(items: list[bytes]) -> bool:
    item_width = len(items[0])
    if item_width <= 0:
        return False

    if any(len(item) != item_width for item in items):
        return False

    if len(items) == 2:
        shared_byte_count = sum(1 for left, right in zip(items[0], items[1]) if left == right)
        shared_ratio = shared_byte_count / item_width
        required_ratio = 0.5 if item_width <= 4 else 0.35
        required_shared_bytes = 2 if item_width <= 6 else 3
        return shared_ratio >= required_ratio or shared_byte_count >= required_shared_bytes

    strong_anchor_columns = 0
    moderate_anchor_columns = 0
    for column_index in range(item_width):
        column_values = [item[column_index] for item in items]
        total = len(column_values)
        common_ratio = Counter(column_values).most_common(1)[0][1] / total
        if common_ratio >= 0.75:
            strong_anchor_columns += 1
        elif common_ratio >= 0.5:
            moderate_anchor_columns += 1

    required_strong_columns = 1 if item_width <= 4 else 2
    required_total_columns = max(required_strong_columns, min(4, (item_width + 3) // 4))
    average_similarity = _average_pairwise_item_similarity(items)
    required_similarity = 0.3 if len(items) <= 3 else 0.25
    return (
        strong_anchor_columns >= required_strong_columns
        and strong_anchor_columns + moderate_anchor_columns >= required_total_columns
        and average_similarity >= required_similarity
    )


def _average_pairwise_item_similarity(items: list[bytes]) -> float:
    pair_count = 0
    similarity_total = 0.0
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            pair_count += 1
            similarity_total += sum(1 for left, right in zip(items[left_index], items[right_index]) if left == right) / len(items[left_index])
    if pair_count == 0:
        return 0.0
    return similarity_total / pair_count


def _parse_counted_repetition(
    message: bytes,
    offset: int,
    count_width: int,
    count_endian: str,
    min_item_width: int,
    max_item_width: int,
    max_items: int,
) -> ParsedRepetition | None:
    count_stop = offset + count_width
    if offset < 0 or count_stop > len(message):
        return None

    item_count = _read_uint(message, offset, count_width, count_endian)
    if item_count is None or item_count <= 1 or item_count > max_items:
        return None

    body_start = count_stop
    body_stop = len(message)
    body_length = body_stop - body_start
    if body_length <= 0 or body_length % item_count != 0:
        return None

    item_width = body_length // item_count
    if item_width < min_item_width or item_width > max_item_width:
        return None

    segments: dict[str, tuple[int, int]] = {
        "REP_Q": (offset, count_stop),
        "REP_BODY": (body_start, body_stop),
        "REP_CONTAINER": (offset, body_stop),
    }
    for item_index in range(item_count):
        start = body_start + item_index * item_width
        stop = start + item_width
        segments[f"REP_ITEM_{item_index}"] = (start, stop)

    return ParsedRepetition(
        segments=segments,
        item_count=item_count,
        item_width=item_width,
        end_offset=body_stop,
    )


def _parse_length_prefixed_region(
    message: bytes,
    offset: int,
    length_width: int,
    length_endian: str,
    min_body_width: int,
) -> ParsedLengthPrefixedRegion | None:
    length_stop = offset + length_width
    if offset < 0 or length_stop > len(message):
        return None

    body_length = _read_uint(message, offset, length_width, length_endian)
    if body_length is None or body_length < min_body_width:
        return None

    body_start = length_stop
    body_stop = body_start + body_length
    if body_stop > len(message):
        return None

    segments = {
        "REP_L": (offset, length_stop),
        "REP_BODY": (body_start, body_stop),
        "REP_CONTAINER": (offset, body_stop),
    }
    return ParsedLengthPrefixedRegion(
        segments=segments,
        body_length=body_length,
        end_offset=body_stop,
    )


def _constraints_from_parsed(
    batch: MessageBatch,
    offset: int,
    rule: ContainerRule,
    parsed_by_index: dict[int, ParsedContainer],
) -> list[Constraint]:
    support_rows = sorted(parsed_by_index)
    parsed_messages = [parsed_by_index[index] for index in support_rows]
    common_labels = set(parsed_messages[0].segments)
    for parsed in parsed_messages[1:]:
        common_labels &= set(parsed.segments)
    if not common_labels:
        return []

    ordered_labels = sorted(common_labels, key=lambda label: parsed_messages[0].segments[label])
    constraints: list[Constraint] = []
    support_ratio = len(support_rows) / batch.num_messages
    mismatch_rows = [index for index in range(batch.num_messages) if index not in parsed_by_index]
    strength = ConstraintStrength.HARD if len(support_rows) == batch.num_messages else ConstraintStrength.SOFT
    if rule.allow_suffix:
        strength = ConstraintStrength.SOFT

    for label in ordered_labels:
        spans: list[Span] = []
        for message_index in support_rows:
            parsed = parsed_by_index[message_index]
            start, stop = parsed.segments[label]
            if stop <= start:
                spans = []
                break
            spans.append(Span(message_index=message_index, start=start, stop=stop, type_code="R"))
        if len(spans) != len(support_rows):
            continue
        kind = ConstraintKind.TLV
        if rule.allow_suffix:
            semantic_role, base_confidence = _tlv_star_semantic_role(label)
        else:
            semantic_role, base_confidence = _tlv_semantic_role(label)
        if strength == ConstraintStrength.HARD:
            confidence = base_confidence
        else:
            penalty = 0.06 if rule.allow_suffix else 0.04
            confidence = max(0.68, min(base_confidence - penalty, 0.55 + 0.35 * support_ratio))
            
        confidence = max(0.20, confidence)
        constraints.append(
            Constraint(
                name=f"tlv:{offset}:{rule.to_annotation()}:{label}",
                kind=kind,
                strength=strength,
                source="native_heuristic:tlv_rule",
                spans=spans,
                annotation=label,
                type_code="R",
                confidence=confidence,
                metadata={
                    "rule": rule.to_annotation(),
                    "rule_family_id": f"tlv_family:{offset}:{rule.to_annotation()}",
                    "offset": offset,
                    "item_counts": [parsed.item_count for parsed in parsed_messages],
                    "suffix_widths": [batch.byte_lengths[index] - parsed_by_index[index].end_offset for index in support_rows],
                    "support_rows": support_rows,
                    "mismatch_rows": mismatch_rows,
                    "support_ratio": round(support_ratio, 4),
                    "semantic_role": semantic_role,
                    "recursive_candidate": label.startswith("TLV_V_") and strength == ConstraintStrength.HARD,
                },
            )
        )
    return constraints


def _constraints_from_repetition(
    batch: MessageBatch,
    offset: int,
    count_width: int,
    count_endian: str,
    parsed_by_index: dict[int, ParsedRepetition],
    source: str,
) -> list[Constraint]:
    support_rows = sorted(parsed_by_index)
    parsed_messages = [parsed_by_index[index] for index in support_rows]
    common_labels = set(parsed_messages[0].segments)
    for parsed in parsed_messages[1:]:
        common_labels &= set(parsed.segments)
    if not common_labels:
        return []

    ordered_labels = sorted(common_labels, key=lambda label: parsed_messages[0].segments[label])
    constraints: list[Constraint] = []
    support_ratio = len(support_rows) / batch.num_messages
    mismatch_rows = [index for index in range(batch.num_messages) if index not in parsed_by_index]
    strength = ConstraintStrength.HARD if len(support_rows) == batch.num_messages else ConstraintStrength.SOFT

    for label in ordered_labels:
        spans: list[Span] = []
        for message_index in support_rows:
            parsed = parsed_by_index[message_index]
            start, stop = parsed.segments[label]
            if stop <= start:
                spans = []
                break
            spans.append(Span(message_index=message_index, start=start, stop=stop, type_code="R"))
        if len(spans) != len(support_rows):
            continue
        semantic_role, base_confidence = _repetition_semantic_role(label)
        confidence = base_confidence if strength == ConstraintStrength.HARD else max(0.7, min(base_confidence - 0.04, 0.55 + 0.35 * support_ratio))
        constraints.append(
            Constraint(
                name=f"repetition:{offset}:{count_width}:{count_endian}:{label}",
                kind=ConstraintKind.REPETITION,
                strength=strength,
                source=source,
                spans=spans,
                annotation=label,
                type_code="R",
                confidence=confidence,
                metadata={
                    "offset": offset,
                    "rule_family_id": f"rep_family:{offset}:{count_width}:{count_endian}",
                    "count_width": count_width,
                    "count_endian": count_endian,
                    "item_counts": [parsed.item_count for parsed in parsed_messages],
                    "item_width": parsed_messages[0].item_width,
                    "support_rows": support_rows,
                    "mismatch_rows": mismatch_rows,
                    "support_ratio": round(support_ratio, 4),
                    "semantic_role": semantic_role,
                    "recursive_candidate": label.startswith("REP_ITEM_") and strength == ConstraintStrength.HARD,
                },
            )
        )
    return constraints


def _constraints_from_tlv_star_parsed(
    batch: MessageBatch,
    offset: int,
    rule: ContainerRule,
    suffix_width: int,
    parsed_by_index: dict[int, ParsedContainer],
) -> list[Constraint]:
    support_rows = sorted(parsed_by_index)
    parsed_messages = [parsed_by_index[index] for index in support_rows]
    common_labels = set(parsed_messages[0].segments)
    for parsed in parsed_messages[1:]:
        common_labels &= set(parsed.segments)
    if not common_labels:
        return []

    ordered_labels = sorted(common_labels, key=lambda label: parsed_messages[0].segments[label])
    constraints: list[Constraint] = []
    support_ratio = len(support_rows) / batch.num_messages
    mismatch_rows = [index for index in range(batch.num_messages) if index not in parsed_by_index]

    for label in ordered_labels:
        spans: list[Span] = []
        for message_index in support_rows:
            parsed = parsed_by_index[message_index]
            start, stop = parsed.segments[label]
            if stop <= start:
                spans = []
                break
            spans.append(Span(message_index=message_index, start=start, stop=stop, type_code="R"))
        if len(spans) != len(support_rows):
            continue
        semantic_role, base_confidence = _tlv_star_semantic_role(label)
        confidence = max(0.64, min(base_confidence, 0.5 + 0.28 * support_ratio))
        constraints.append(
            Constraint(
                name=f"tlv_star:{offset}:{rule.to_annotation()}:{suffix_width}:{label}",
                kind=ConstraintKind.TLV,
                strength=ConstraintStrength.SOFT,
                source="native_heuristic:tlv_star_rule",
                spans=spans,
                annotation=label,
                type_code="R",
                confidence=confidence,
                metadata={
                    "rule": rule.to_annotation(),
                    "rule_family_id": f"tlvstar_family:{offset}:{rule.to_annotation()}:{suffix_width}",
                    "offset": offset,
                    "item_counts": [parsed.item_count for parsed in parsed_messages],
                    "suffix_width": suffix_width,
                    "suffix_widths": [suffix_width for _ in support_rows],
                    "support_rows": support_rows,
                    "mismatch_rows": mismatch_rows,
                    "support_ratio": round(support_ratio, 4),
                    "semantic_role": semantic_role,
                    "recursive_candidate": False,
                    "star_rule": True,
                },
            )
        )
    return constraints


def _constraints_from_length_prefixed_region(
    batch: MessageBatch,
    offset: int,
    length_width: int,
    length_endian: str,
    parsed_by_index: dict[int, ParsedLengthPrefixedRegion],
) -> list[Constraint]:
    support_rows = sorted(parsed_by_index)
    parsed_messages = [parsed_by_index[index] for index in support_rows]
    common_labels = set(parsed_messages[0].segments)
    for parsed in parsed_messages[1:]:
        common_labels &= set(parsed.segments)
    if not common_labels:
        return []

    ordered_labels = sorted(common_labels, key=lambda label: parsed_messages[0].segments[label])
    constraints: list[Constraint] = []
    support_ratio = len(support_rows) / batch.num_messages
    mismatch_rows = [index for index in range(batch.num_messages) if index not in parsed_by_index]
    strength = ConstraintStrength.SOFT

    for label in ordered_labels:
        spans: list[Span] = []
        for message_index in support_rows:
            parsed = parsed_by_index[message_index]
            start, stop = parsed.segments[label]
            if stop <= start:
                spans = []
                break
            spans.append(Span(message_index=message_index, start=start, stop=stop, type_code="R"))
        if len(spans) != len(support_rows):
            continue
        semantic_role, base_confidence = _length_prefixed_vector_role(label)
        confidence = max(0.64, min(base_confidence, 0.5 + 0.28 * support_ratio))
        constraints.append(
            Constraint(
                name=f"repetition:vector:{offset}:{length_width}:{length_endian}:{label}",
                kind=ConstraintKind.REPETITION,
                strength=strength,
                source="native_heuristic:length_prefixed_vector",
                spans=spans,
                annotation=label,
                type_code="R",
                confidence=confidence,
                metadata={
                    "offset": offset,
                    "rule_family_id": f"repvec_family:{offset}:{length_width}:{length_endian}",
                    "length_width": length_width,
                    "length_endian": length_endian,
                    "body_lengths": [parsed.body_length for parsed in parsed_messages],
                    "support_rows": support_rows,
                    "mismatch_rows": mismatch_rows,
                    "support_ratio": round(support_ratio, 4),
                    "semantic_role": semantic_role,
                    "recursive_candidate": label == "REP_BODY",
                },
            )
        )
    return constraints


def _tlv_semantic_role(label: str) -> tuple[str, float]:
    if label == "TLV_CONTAINER":
        return "tlv_container", 0.94
    if label == "TLV_Q":
        return "tlv_q", 0.95
    if label.startswith("TLV_ITEM_"):
        return "tlv_item", 0.93
    if "_T_" in label:
        return "tlv_t", 0.93
    if "_L_" in label:
        return "tlv_l", 0.96
    if "_V_" in label:
        return "tlv_v", 0.92
    return "tlv_field", 0.9


def _tlv_star_semantic_role(label: str) -> tuple[str, float]:
    if label == "TLV_CONTAINER":
        return "tlv_star_container", 0.8
    if label == "TLV_Q":
        return "tlv_star_q", 0.8
    if label.startswith("TLV_ITEM_"):
        return "tlv_star_item", 0.79
    if "_T_" in label:
        return "tlv_star_t", 0.78
    if "_L_" in label:
        return "tlv_star_l", 0.81
    if "_V_" in label:
        return "tlv_star_v", 0.77
    return "tlv_star", 0.76


def _repetition_semantic_role(label: str) -> tuple[str, float]:
    if label == "REP_CONTAINER":
        return "rep_container", 0.97
    if label == "REP_Q":
        return "rep_q", 0.98
    if label == "REP_L":
        return "rep_l", 0.97
    if label == "REP_BODY":
        return "rep_body", 0.93
    if label.startswith("REP_ITEM_"):
        return "rep_item", 0.94
    return "repetition", 0.9


def _length_prefixed_vector_role(label: str) -> tuple[str, float]:
    if label == "REP_CONTAINER":
        return "rep_vector_container", 0.8
    if label == "REP_L":
        return "rep_vector_l", 0.82
    if label == "REP_BODY":
        return "rep_vector_body", 0.79
    return "repetition", 0.76
