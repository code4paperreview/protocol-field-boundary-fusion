"""Boundary evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(slots=True, frozen=True)
class BoundaryMetrics:
    acc: float
    f1: float
    precision: float
    recall: float
    fpr: float
    perfection: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def segment_spec_to_offsets(segment_spec: Any, data_length: int | None = None) -> list[int]:
    if not isinstance(segment_spec, Sequence) or isinstance(segment_spec, (bytes, bytearray, str)):
        raise TypeError(f"Unsupported segment specification: {type(segment_spec)!r}")
    if not segment_spec:
        return []
    if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in segment_spec):
        offsets = [int(item) for item in segment_spec]
    else:
        offsets = []
        for item in segment_spec:
            if not isinstance(item, Sequence) or isinstance(item, (bytes, bytearray, str)) or len(item) < 2:
                raise TypeError(f"Unsupported segment item: {item!r}")
            offsets.extend((int(item[0]), int(item[1])))
    normalized = sorted(set(offsets))
    if data_length is not None:
        normalized = [offset for offset in normalized if 0 <= offset <= data_length]
    return normalized


def calculate_boundary_metrics(
    pred_offsets: Sequence[int], true_offsets: Sequence[int], data_length: int
) -> BoundaryMetrics:
    pred_set = {int(offset) for offset in pred_offsets if 0 <= int(offset) < data_length}
    true_set = {int(offset) for offset in true_offsets if 0 <= int(offset) < data_length}

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len(set(range(data_length)) - pred_set - true_set)
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def intervals(boundaries: set[int]) -> set[tuple[int, int]]:
        values = sorted(boundaries | {0, data_length})
        return set(zip(values, values[1:]))

    truth_fields = intervals(true_set)
    predicted_fields = intervals(pred_set)
    perfection = len(predicted_fields & truth_fields) / len(truth_fields) if truth_fields else 0.0

    return BoundaryMetrics(
        acc=(tp + tn) / total if total else 0.0,
        f1=f1,
        precision=precision,
        recall=recall,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
        perfection=perfection,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def average_boundary_metrics(metrics: Sequence[BoundaryMetrics]) -> dict[str, float]:
    if not metrics:
        return {key: 0.0 for key in ("acc", "f1", "precision", "recall", "fpr", "perfection")}
    count = float(len(metrics))
    return {
        key: sum(getattr(metric, key) for metric in metrics) / count
        for key in ("acc", "f1", "precision", "recall", "fpr", "perfection")
    }
