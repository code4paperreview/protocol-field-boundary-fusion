from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta
from functools import lru_cache
import gzip
import math
from pathlib import Path
from typing import Iterable, Mapping
import zlib

from ..common.messages import MessageBatch
from ..common.models import Constraint, ConstraintKind, ConstraintStrength, Span
from .detectors import deduplicate_constraints

_FILETIME_UNIX_EPOCH_SECONDS = 11644473600.0
_ENTROPY_REFERENCE_FILENAME = "N_entropy_result.csv"
_ENTROPY_BLOCK_SIZE = 1000
_ENTROPY_TOLERANCE = 4.0


def detect_sequence_fields(
    batch: MessageBatch,
    widths: Iterable[int] = (1, 2, 3, 4),
    endians: Iterable[str] = ("big", "little"),
    offset_scan_limit: int = 10,
    min_increase_ratio: float = 0.85,
    min_distinct_ratio: float = 0.6,
) -> list[Constraint]:
    if batch.num_messages < 3:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)

    for offset in range(max_offset):
        for width in widths:
            if offset + width > batch.min_length:
                continue
            for endian in endians:
                values = [_read_uint(message, offset, width, endian) for message in messages]
                if any(value is None for value in values):
                    continue
                numeric_values = [int(value) for value in values if value is not None]
                if len(set(numeric_values)) <= 1:
                    continue
                distinct_ratio = _distinct_ratio(numeric_values)
                if distinct_ratio < min_distinct_ratio:
                    continue
                increase_ratio = _strict_increase_ratio(numeric_values)
                if increase_ratio < min_increase_ratio:
                    continue
                spans = [Span(message_index=i, start=offset, stop=offset + width, type_code="S") for i in range(batch.num_messages)]
                confidence = min(0.95, 0.76 + 0.12 * increase_ratio + 0.07 * distinct_ratio)
                constraints.append(
                    Constraint(
                        name=f"sequence:{offset}:{width}:{endian}",
                        kind=ConstraintKind.SEQUENCE,
                        strength=ConstraintStrength.SOFT,
                        source="native_heuristic:sequence_counter",
                        spans=spans,
                        annotation=f"Sequence[{width}][{endian}]",
                        type_code="S",
                        confidence=confidence,
                        metadata={
                            "offset": offset,
                            "width": width,
                            "endian": endian,
                            "values": numeric_values,
                            "increase_ratio": increase_ratio,
                            "distinct_ratio": distinct_ratio,
                            "median_delta": _median_positive_delta(numeric_values),
                            "semantic_role": "sequence_counter",
                        },
                    )
                )

    return deduplicate_constraints(constraints)


def detect_identifier_fields(
    batch: MessageBatch,
    widths: Iterable[int] = (1, 2),
    offset_scan_limit: int = 10,
    min_unique_count: int = 2,
    max_unique_ratio: float = 0.50,
    max_unique_count: int = 30,
) -> list[Constraint]:
    if batch.num_messages < 3:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)

    for offset in range(max_offset):
        for width in widths:
            if offset + width > batch.min_length:
                continue
            chunks = [message[offset : offset + width] for message in messages]
            unique_count = len(set(chunks))
            if unique_count < min_unique_count:
                continue
            if unique_count > max_unique_count:
                continue
            unique_ratio = unique_count / len(chunks)
            if unique_ratio > max_unique_ratio:
                continue
            numbers = [int.from_bytes(c, byteorder="big", signed=False) for c in chunks]
            if _strict_increase_ratio(numbers) >= 0.85:
                continue
            if width > 1:
                skip = False
                for j in range(width):
                    col = {message[offset + j] for message in messages}
                    if len(col) == 1:
                        const_val = next(iter(col))
                        if const_val != 0 or j != 0:
                            skip = True
                            break
                if skip:
                    continue
            spans = [Span(message_index=i, start=offset, stop=offset + width, type_code="X") for i in range(batch.num_messages)]
            confidence = min(
                0.9,
                0.70 + 0.10 * (1.0 - unique_ratio) + 0.05 * min(width, 2),
            )
            constraints.append(
                Constraint(
                    name=f"identifier:{offset}:{width}",
                    kind=ConstraintKind.IDENTIFIER,
                    strength=ConstraintStrength.SOFT,
                    source="native_heuristic:identifier_pattern",
                    spans=spans,
                    annotation=f"Identifier[{width}]",
                    type_code="X",
                    confidence=confidence,
                    metadata={
                        "offset": offset,
                        "width": width,
                        "unique_count": unique_count,
                        "unique_ratio": unique_ratio,
                        "semantic_role": "identifier",
                    },
                )
            )

    return deduplicate_constraints(constraints)


def detect_timestamp_fields(
    batch: MessageBatch,
    widths: Iterable[int] = (4, 8),
    endians: Iterable[str] = ("big", "little"),
    offset_scan_limit: int = 12,
    min_valid_ratio: float = 0.8,
) -> list[Constraint]:
    if batch.num_messages < 3:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length, offset_scan_limit)

    for offset in range(max_offset):
        for width in widths:
            if offset + width > batch.min_length:
                continue
            for endian in endians:
                normalized_chunks = [message[offset : offset + width] if endian == "little" else message[offset : offset + width][::-1] for message in messages]
                if _has_stripes(normalized_chunks):
                    continue
                if not _has_low_order_variation(normalized_chunks):
                    continue
                values = [_read_uint(message, offset, width, endian) for message in messages]
                if any(value is None for value in values):
                    continue
                raw_values = [int(value) for value in values if value is not None]
                for scale_name, seconds_values in _timestamp_seconds_candidates(raw_values, width):
                    evaluation = _evaluate_timestamp_seconds(seconds_values, min_valid_ratio=min_valid_ratio)
                    if evaluation is None:
                        continue
                    spans = [Span(message_index=i, start=offset, stop=offset + width, type_code="T") for i in range(batch.num_messages)]
                    confidence = min(0.96, 0.78 + 0.1 * evaluation["valid_ratio"] + 0.06 * evaluation["distinct_ratio"] + 0.04 * evaluation["monotonic_ratio"])
                    constraints.append(
                        Constraint(
                            name=f"timestamp:{offset}:{width}:{endian}:{scale_name}",
                            kind=ConstraintKind.TIMESTAMP,
                            strength=ConstraintStrength.SOFT,
                            source="native_heuristic:timestamp_value",
                            spans=spans,
                            annotation=f"Timestamp[{width}][{endian}][{scale_name}]",
                            type_code="T",
                            confidence=confidence,
                            metadata={
                                "offset": offset,
                                "width": width,
                                "endian": endian,
                                "scale": scale_name,
                                "min_datetime": evaluation["min_datetime"],
                                "max_datetime": evaluation["max_datetime"],
                                "valid_ratio": evaluation["valid_ratio"],
                                "distinct_ratio": evaluation["distinct_ratio"],
                                "monotonic_ratio": evaluation["monotonic_ratio"],
                                "near_now_ratio": evaluation["near_now_ratio"],
                                "near_epoch_ratio": evaluation["near_epoch_ratio"],
                                "semantic_role": "timestamp",
                            },
                        )
                    )

    return deduplicate_constraints(constraints)


def detect_entropy_delimiters(
    batch: MessageBatch,
    offset_scan_limit: int = 24,
    min_entropy_delta: float = 1.0,
    min_right_entropy: float = 1.5,
    max_left_entropy: float = 1.25,
) -> list[Constraint]:
    if batch.num_messages < 3 or batch.min_length < 2:
        return []

    messages = batch.to_bytes_list()
    constraints: list[Constraint] = []
    max_offset = min(batch.min_length - 1, offset_scan_limit)

    for offset in range(max_offset):
        left_column = [message[offset] for message in messages]
        right_column = [message[offset + 1] for message in messages]
        left_entropy = _symbol_entropy(left_column)
        right_entropy = _symbol_entropy(right_column)
        delta = right_entropy - left_entropy
        if delta < min_entropy_delta or right_entropy < min_right_entropy or left_entropy > max_left_entropy:
            continue
        spans = [Span(message_index=i, start=offset + 1, stop=offset + 1, type_code="|") for i in range(batch.num_messages)]
        confidence = min(0.9, 0.62 + 0.12 * delta)
        constraints.append(
            Constraint(
                name=f"delimiter:{offset + 1}",
                kind=ConstraintKind.BOUNDARY,
                strength=ConstraintStrength.SOFT,
                source="native_heuristic:entropy_delimiter",
                spans=spans,
                annotation=f"EntropyDelimiter[{offset + 1}]",
                type_code="|",
                confidence=confidence,
                metadata={
                    "offset": offset + 1,
                    "left_entropy": left_entropy,
                    "right_entropy": right_entropy,
                    "entropy_delta": delta,
                    "semantic_role": "delimiter",
                },
            )
        )

    return deduplicate_constraints(constraints)


def detect_payload_regions(
    batch: MessageBatch,
    offset_scan_limit: int = 24,
    min_payload_width: int = 17,
    min_support_ratio: float = 0.75,
    allowed_intervals: dict[int, list[tuple[int, int]]] | None = None,
) -> list[Constraint]:
    if batch.num_messages < 2 or batch.max_length < min_payload_width:
        return []

    messages = batch.to_bytes_list()
    support_threshold = max(2, math.ceil(batch.num_messages * min_support_ratio))
    constraints: list[Constraint] = []
    if allowed_intervals is not None:
        for start, grouped_rows in _iter_allowed_region_groups(allowed_intervals):
            if len(grouped_rows) < support_threshold:
                continue
            constraints.extend(
                _classify_payload_group(
                    batch,
                    messages,
                    start,
                    grouped_rows,
                    min_payload_width=min_payload_width,
                )
            )
        return deduplicate_constraints(constraints)

    delimiter_candidates = detect_entropy_delimiters(batch, offset_scan_limit=offset_scan_limit)
    candidate_offsets = {
        int(span.start)
        for constraint in delimiter_candidates
        for span in constraint.spans
        if span.start > 0
    }
    candidate_offsets.update(_compression_magic_offsets(messages, max_scan_limit=offset_scan_limit + 8, support_threshold=support_threshold))

    for offset in sorted(candidate_offsets):
        grouped_rows = [
            (message_index, len(message))
            for message_index, message in enumerate(messages)
            if len(message) - offset >= min_payload_width
        ]
        if len(grouped_rows) < support_threshold:
            continue
        constraints.extend(
            _classify_payload_group(
                batch,
                messages,
                offset,
                grouped_rows,
                min_payload_width=min_payload_width,
            )
        )

    return deduplicate_constraints(constraints)


def _iter_allowed_region_groups(
     allowed_intervals: dict[int, list[tuple[int, int]]],
 ) -> list[tuple[int, list[tuple[int, int]]]]:
     grouped: dict[int, list[tuple[int, int]]] = {}
     for message_index, intervals in allowed_intervals.items():
         seen_starts: set[int] = set()
         for start, stop in intervals:
             if stop <= start or start in seen_starts:
                 continue
             grouped.setdefault(start, []).append((message_index, stop))
             seen_starts.add(start)
     return [(start, sorted(rows)) for start, rows in sorted(grouped.items())]


def _classify_payload_group(
     batch: MessageBatch,
     messages: list[bytes],
     start: int,
     grouped_rows: list[tuple[int, int]],
     min_payload_width: int,
 ) -> list[Constraint]:
     valid_rows: list[tuple[int, int]] = []
     region_chunks: list[bytes] = []
     for message_index, stop in sorted(grouped_rows):
         if stop <= start or stop > len(messages[message_index]):
             continue
         valid_rows.append((message_index, stop))
         region_chunks.append(messages[message_index][start:stop])
     if len(valid_rows) < 2:
         return []

     if len(set(region_chunks)) == 1:
         constant_constraint = _build_constant_group_constraint(
             batch,
             valid_rows,
             start,
             valid_rows[0][1],
             region_chunks[0],
             source="native_heuristic:constant_payload_prefix",
             derived_role="payload_constant",
         )
         return [] if constant_constraint is None else [constant_constraint]

     prefix_width = _stable_prefix_length(region_chunks)
     split_constant_prefix = _should_split_constant_prefix(region_chunks, prefix_width)
     constraints: list[Constraint] = []
     tail_start = start
     if split_constant_prefix:
         tail_start = start + prefix_width
         prefix_constraint = _build_constant_group_constraint(
             batch,
             valid_rows,
             start,
             tail_start,
             region_chunks[0][:prefix_width],
             source="native_heuristic:constant_payload_prefix",
             derived_role="payload_prefix_constant",
         )
         if prefix_constraint is not None:
             constraints.append(prefix_constraint)

     tail_rows: list[tuple[int, int]] = []
     tail_chunks: list[bytes] = []
     for message_index, stop in valid_rows:
         if stop <= tail_start:
             return constraints
         tail_rows.append((message_index, stop))
         tail_chunks.append(messages[message_index][tail_start:stop])
     if not tail_chunks:
         return constraints

     compressed_hits: list[tuple[int, int, str]] = []
     for (message_index, stop), payload in zip(tail_rows, tail_chunks):
         compression_format = _detect_compression_format(payload)
         if compression_format is not None:
             compressed_hits.append((message_index, stop, compression_format))

     if len(compressed_hits) > len(tail_rows) / 2:
         spans = [Span(message_index=message_index, start=tail_start, stop=stop, type_code="P") for message_index, stop in tail_rows]
         format_counts = Counter(format_name for _, _, format_name in compressed_hits)
         support_ratio = len(tail_rows) / batch.num_messages
         vote_ratio = len(compressed_hits) / len(tail_rows)
         confidence = min(0.96, 0.7 + 0.1 * support_ratio + 0.14 * vote_ratio + 0.02 * len(format_counts))
         constraints.append(
             Constraint(
                 name=f"payload:compressed:{start}:{tail_start}",
                 kind=ConstraintKind.PAYLOAD,
                 strength=ConstraintStrength.SOFT,
                 source="native_heuristic:compressed_payload",
                 spans=spans,
                 annotation=f"CompressedPayload[{tail_start}]",
                 type_code="P",
                 confidence=confidence,
                 metadata={
                     "offset": tail_start,
                     "field_start": start,
                     "support_ratio": round(support_ratio, 4),
                     "vote_ratio": round(vote_ratio, 4),
                     "formats": dict(format_counts),
                     "support_rows": [message_index for message_index, _ in tail_rows],
                     "compressed_rows": [message_index for message_index, _, _ in compressed_hits],
                     "constant_prefix_width": prefix_width if split_constant_prefix else 0,
                     "semantic_role": "compressed_payload",
                 },
             )
         )
         return constraints

     if not all(len(payload) >= min_payload_width for payload in tail_chunks):
         return constraints

     encrypted_rows: list[int] = []
     entropy_reference = _load_entropy_reference()
     for (message_index, _), payload in zip(tail_rows, tail_chunks):
         if _is_data_encrypted(payload, entropy_reference):
             encrypted_rows.append(message_index)

     if len(encrypted_rows) > len(tail_rows) / 2:
         spans = [Span(message_index=message_index, start=tail_start, stop=stop, type_code="M") for message_index, stop in tail_rows]
         support_ratio = len(tail_rows) / batch.num_messages
         vote_ratio = len(encrypted_rows) / len(tail_rows)
         confidence = min(0.95, 0.68 + 0.1 * support_ratio + 0.16 * vote_ratio + 0.01 * min(prefix_width, 4))
         constraints.append(
             Constraint(
                 name=f"payload:encrypted:{start}:{tail_start}",
                 kind=ConstraintKind.PAYLOAD,
                 strength=ConstraintStrength.SOFT,
                 source="native_heuristic:encrypted_payload",
                 spans=spans,
                 annotation=f"EncryptedPayload[{tail_start}]",
                 type_code="M",
                 confidence=confidence,
                 metadata={
                     "offset": tail_start,
                     "field_start": start,
                     "support_ratio": round(support_ratio, 4),
                     "vote_ratio": round(vote_ratio, 4),
                     "support_rows": [message_index for message_index, _ in tail_rows],
                     "encrypted_rows": encrypted_rows,
                     "constant_prefix_width": prefix_width if split_constant_prefix else 0,
                     "classifier": "truncated_entropy",
                     "semantic_role": "encrypted_payload",
                 },
             )
         )

     return constraints


def _should_split_constant_prefix(chunks: list[bytes], prefix_width: int) -> bool:
     if prefix_width <= 0:
         return False
     min_width = min(len(chunk) for chunk in chunks)
     if prefix_width >= min_width:
         return False
     if prefix_width == 1 and all(chunk[:1] == b"\x00" for chunk in chunks):
         return False
     return True


def _build_constant_group_constraint(
     batch: MessageBatch,
     grouped_rows: list[tuple[int, int]],
     start: int,
     stop: int,
     value: bytes,
     source: str,
     derived_role: str,
 ) -> Constraint | None:
     if stop <= start:
         return None
     spans = [Span(message_index=message_index, start=start, stop=stop, type_code="A") for message_index, _ in grouped_rows]
     support_ratio = len(grouped_rows) / batch.num_messages
     confidence = min(0.92, 0.62 + 0.04 * (stop - start) + 0.12 * support_ratio)
     return Constraint(
         name=f"constant_payload:{start}:{stop}",
         kind=ConstraintKind.CONSTANT,
         strength=ConstraintStrength.SOFT,
         source=source,
         spans=spans,
         annotation=f"Constant:{value.hex()}",
         type_code="A",
         confidence=confidence,
         metadata={
             "value_hex": value.hex(),
             "support_rows": [message_index for message_index, _ in grouped_rows],
             "support_ratio": round(support_ratio, 4),
             "semantic_role": "constant_field",
             "derived_from": derived_role,
         },
     )


@lru_cache(maxsize=1)
def _load_entropy_reference() -> dict[int, dict[str, float]]:
     csv_path = _resolve_entropy_reference_path()
     reference: dict[int, dict[str, float]] = {}
     with csv_path.open("r", encoding="utf-8", newline="") as handle:
         reader = csv.DictReader(handle)
         for row in reader:
             n_value = int(row["N"])
             reference[n_value] = {
                 "H_N(u)": float(row["H_N(u)"]),
                 "SD(H^NMLE)": float(row["SD(H^NMLE)"]),
                 "Confidence Interval Low": float(row["Confidence Interval Low"]),
                 "Confidence Interval High": float(row["Confidence Interval High"]),
             }
     return reference


def _resolve_entropy_reference_path() -> Path:
     local_path = Path(__file__).with_name(_ENTROPY_REFERENCE_FILENAME)
     if local_path.exists():
         return local_path
     raise FileNotFoundError(_ENTROPY_REFERENCE_FILENAME)


def _is_data_encrypted(
     data: bytes,
     reference: Mapping[int, Mapping[str, float]],
     block_size: int = _ENTROPY_BLOCK_SIZE,
     tolerance: float = _ENTROPY_TOLERANCE,
 ) -> bool:
     if len(data) < 16:
         return False

     full_blocks = len(data) // block_size
     remainder = len(data) % block_size
     full_block_reference = reference.get(block_size)
     if full_blocks > 0 and full_block_reference is None:
         return False

     for block_index in range(full_blocks):
         start = block_index * block_size
         stop = start + block_size
         entropy = _byte_entropy(data[start:stop])
         if not (
             float(full_block_reference["Confidence Interval Low"])
             <= entropy
             <= float(full_block_reference["Confidence Interval High"])
         ):
             return False

     if remainder >= 16:
         remainder_reference = reference.get(remainder)
         if remainder_reference is None:
             return False
         entropy = _byte_entropy(data[full_blocks * block_size :])
         center = float(remainder_reference["H_N(u)"])
         standard_deviation = float(remainder_reference["SD(H^NMLE)"])
         if not (center - standard_deviation * tolerance <= entropy <= center + standard_deviation * tolerance):
             return False

     return True


def _read_uint(message: bytes, offset: int, width: int, endian: str) -> int | None:
     stop = offset + width
     if offset < 0 or stop > len(message):
         return None
     return int.from_bytes(message[offset:stop], byteorder=endian, signed=False)


def _distinct_ratio(values: list[object]) -> float:
    if not values:
        return 0.0
    return len(set(values)) / len(values)


def _strict_increase_ratio(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    increases = sum(1 for left, right in zip(values, values[1:]) if right > left)
    return increases / (len(values) - 1)


def _nondecrease_ratio(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    nondecreases = sum(1 for left, right in zip(values, values[1:]) if right >= left)
    return nondecreases / (len(values) - 1)


def _median_positive_delta(values: list[int]) -> int:
    deltas = sorted(right - left for left, right in zip(values, values[1:]) if right > left)
    if not deltas:
        return 0
    return deltas[len(deltas) // 2]


def _stable_prefix_length(chunks: list[bytes]) -> int:
    if not chunks:
        return 0
    width = min(len(chunk) for chunk in chunks)
    prefix = 0
    for offset in range(width):
        if len({chunk[offset] for chunk in chunks}) != 1:
            break
        prefix += 1
    return prefix


def _stable_suffix_length(chunks: list[bytes]) -> int:
    if not chunks:
        return 0
    width = min(len(chunk) for chunk in chunks)
    suffix = 0
    for back_offset in range(1, width + 1):
        if len({chunk[-back_offset] for chunk in chunks}) != 1:
            break
        suffix += 1
    return suffix


def _allowed_region_starts(allowed_intervals: dict[int, list[tuple[int, int]]]) -> set[int]:
    return {
        start
        for intervals in allowed_intervals.values()
        for start, stop in intervals
        if stop > start
    }


def _offset_support_in_regions(offset: int, allowed_intervals: dict[int, list[tuple[int, int]]]) -> int:
    return sum(
        1
        for intervals in allowed_intervals.values()
        if _region_for_offset(intervals, offset) is not None
    )


def _region_for_offset(intervals: list[tuple[int, int]], offset: int) -> tuple[int, int] | None:
    for start, stop in intervals:
        if start <= offset < stop:
            return (start, stop)
    return None


def _region_has_supported_inner_candidate(
    start: int,
    candidate_offsets: set[int],
    allowed_intervals: dict[int, list[tuple[int, int]]],
    support_threshold: int,
) -> bool:
    for offset in candidate_offsets:
        if offset <= start:
            continue
        shared_support = 0
        for intervals in allowed_intervals.values():
            region = _region_for_offset(intervals, start)
            if region is None:
                continue
            region_start, region_stop = region
            if region_start == start and region_start < offset < region_stop:
                shared_support += 1
        if shared_support >= support_threshold:
            return True
    return False


def _has_stripes(chunks: list[bytes]) -> bool:
    if not chunks:
        return False
    width = min(len(chunk) for chunk in chunks)
    pattern = "".join("c" if len({chunk[offset] for chunk in chunks}) == 1 else "n" for offset in range(width))
    compact: list[str] = []
    for char in pattern:
        if not compact or compact[-1] != char:
            compact.append(char)
    reduced = "".join(compact)
    return "ncn" in reduced or "cn" in reduced or "cnc" in reduced


def _varying_position_count(chunks: list[bytes]) -> int:
    if not chunks:
        return 0
    width = min(len(chunk) for chunk in chunks)
    return sum(1 for offset in range(width) if len({chunk[offset] for chunk in chunks}) > 1)


def _has_low_order_variation(chunks: list[bytes]) -> bool:
    if not chunks:
        return False
    width = min(len(chunk) for chunk in chunks)
    varying_positions = [offset for offset in range(width) if len({chunk[offset] for chunk in chunks}) > 1]
    if not varying_positions:
        return False
    return varying_positions == list(range(len(varying_positions)))


def _timestamp_seconds_candidates(raw_values: list[int], width: int) -> list[tuple[str, list[float]]]:
    candidates = [("unix_s", [float(value) for value in raw_values])]
    if width == 8:
        candidates.extend(
            [
                ("unix_ms", [value / 1_000.0 for value in raw_values]),
                ("unix_us", [value / 1_000_000.0 for value in raw_values]),
                ("unix_ns", [value / 1_000_000_000.0 for value in raw_values]),
                ("filetime_100ns", [value / 10_000_000.0 - _FILETIME_UNIX_EPOCH_SECONDS for value in raw_values]),
            ]
        )
    return candidates


def _evaluate_timestamp_seconds(seconds_values: list[float], min_valid_ratio: float) -> dict[str, object] | None:
    datetimes: list[datetime] = []
    valid_seconds: list[float] = []
    for seconds_value in seconds_values:
        dt = _to_datetime(seconds_value)
        if dt is None:
            continue
        datetimes.append(dt)
        valid_seconds.append(seconds_value)

    if not datetimes:
        return None

    valid_ratio = len(datetimes) / len(seconds_values)
    if valid_ratio < min_valid_ratio:
        return None

    distinct_ratio = _distinct_ratio([int(seconds_value) for seconds_value in valid_seconds])
    if distinct_ratio < 0.5:
        return None

    now = datetime.utcnow()
    recent_lower = now - timedelta(days=3650)
    recent_upper = now + timedelta(days=30)
    epoch_lower = datetime(1970, 1, 1)
    epoch_upper = epoch_lower + timedelta(days=365)
    near_now_ratio = sum(1 for dt in datetimes if recent_lower <= dt <= recent_upper) / len(datetimes)
    near_epoch_ratio = sum(1 for dt in datetimes if epoch_lower <= dt <= epoch_upper) / len(datetimes)
    if max(near_now_ratio, near_epoch_ratio) < min_valid_ratio:
        return None

    monotonic_ratio = _nondecrease_ratio(valid_seconds)
    if monotonic_ratio < 0.8:
        return None

    min_dt = min(datetimes)
    max_dt = max(datetimes)
    return {
        "valid_ratio": valid_ratio,
        "distinct_ratio": distinct_ratio,
        "monotonic_ratio": monotonic_ratio,
        "near_now_ratio": near_now_ratio,
        "near_epoch_ratio": near_epoch_ratio,
        "min_datetime": min_dt.isoformat(),
        "max_datetime": max_dt.isoformat(),
    }


def _to_datetime(seconds_value: float) -> datetime | None:
    try:
        return datetime(1970, 1, 1) + timedelta(seconds=float(seconds_value))
    except (OverflowError, OSError, ValueError):
        return None


def _symbol_entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _normalized_entropy_ratio(data: bytes, entropy: float | None = None) -> float:
    if not data:
        return 0.0
    max_entropy = math.log2(min(256, len(data)))
    if max_entropy <= 0:
        return 0.0
    value = _byte_entropy(data) if entropy is None else entropy
    return value / max_entropy


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(1 for byte in data if 32 <= byte <= 126 or byte in {9, 10, 13})
    return printable / len(data)


def _compression_magic_offsets(messages: list[bytes], max_scan_limit: int, support_threshold: int) -> set[int]:
    offsets: set[int] = set()
    max_length = max((len(message) for message in messages), default=0)
    for offset in range(max(0, min(max_length - 1, max_scan_limit))):
        hits = 0
        for message in messages:
            if offset + 2 > len(message):
                continue
            if _looks_like_gzip_header(message[offset:offset + 2]) or _looks_like_zlib_header(message[offset:offset + 2]):
                hits += 1
        if hits >= support_threshold:
            offsets.add(offset)
    return offsets


def _detect_compression_format(data: bytes) -> str | None:
    if len(data) < 4:
        return None
    if _looks_like_gzip_header(data[:2]):
        try:
            gzip.decompress(data)
            return "gzip"
        except Exception:
            pass
    if _looks_like_zlib_header(data[:2]):
        try:
            decompressor = zlib.decompressobj()
            decompressor.decompress(data)
            consumed = len(data) - len(decompressor.unused_data)
            if consumed >= 8 and (decompressor.eof or consumed >= min(len(data), 12)):
                return "zlib"
        except zlib.error:
            pass
    return None


def _looks_like_gzip_header(header: bytes) -> bool:
    return len(header) >= 2 and header[:2] == b"\x1f\x8b"


def _looks_like_zlib_header(header: bytes) -> bool:
    if len(header) < 2:
        return False
    cmf, flg = header[0], header[1]
    if cmf & 0x0F != 8:
        return False
    return ((cmf << 8) + flg) % 31 == 0
