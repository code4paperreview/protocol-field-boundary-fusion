"""Shared I/O and coordinate helpers."""

import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nhf.evaluation.metrics import (
    average_boundary_metrics, calculate_boundary_metrics, segment_spec_to_offsets,
)

DATA_ROOT = Path(os.environ.get("NHF_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))


@dataclass(frozen=True)
class Message:
    row_index: int
    payload: bytes


def load_messages(path: str | Path) -> list[Message]:
    """Read Hex only. Duplicate messages retain distinct row identities."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [
            Message(index, bytes.fromhex(row["Hex"]))
            for index, row in enumerate(csv.DictReader(stream))
        ]


def bit_boundaries_to_bytes(offsets: Iterable[int], length: int) -> list[int]:
    """Project native bit offsets onto byte boundaries without rounding."""
    return sorted({p // 8 for p in offsets if 0 <= p <= 8 * length and p % 8 == 0})


def field_end_labels_to_boundaries(labels: Iterable[int]) -> list[int]:
    """TransRE's positive label marks a field's LAST byte (PDF p. 6)."""
    return [index + 1 for index, label in enumerate(labels) if label == 1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_csv(protocol: str) -> Path:
    for suffix in ["", "_realistic", "_balanced"]:
        path = DATA_ROOT / protocol / f"{protocol}{suffix}.csv"
        if path.exists():
            return path
    raise FileNotFoundError(f"No input CSV for {protocol}")


def read_truth(path: Path, messages: list[Message]) -> dict[int, list[int]]:
    """Used for labelled source data, or ONLY after target prediction is saved."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        message.row_index: segment_spec_to_offsets(rows[message.row_index]["Segment"], len(message.payload))
        for message in messages
    }


def load_sources(target: str, target_messages: list[Message], protocols: list[str] | None,
                 limit: int, seed: int):
    """Fixed protocol-family holdout, exact-payload overlap removal, grouped split."""
    excluded = {target}
    if target in {"smb", "smb2"}:
        excluded |= {"smb", "smb2"}
    if target in {"tls", "ssl"}:
        excluded |= {"tls", "ssl"}
    if protocols is None:
        protocols = []
        for directory in sorted(DATA_ROOT.iterdir()):
            if directory.is_dir() and any((directory / f"{directory.name}{s}.csv").exists() for s in ["", "_realistic", "_balanced"]):
                protocols.append(directory.name)
    protocols = [p for p in protocols if p not in excluded]
    target_payloads = {message.payload for message in target_messages}
    all_keys = set()
    pools, paths, overlap = {}, {}, {}
    for protocol in protocols:
        path = resolve_csv(protocol)
        original = load_messages(path)
        messages = [m for m in original if m.payload not in target_payloads]
        overlap[protocol] = len(original) - len(messages)
        messages = messages[:limit] if limit else messages
        if not messages:
            raise ValueError(f"Source {protocol} has no messages after overlap filtering")
        pools[protocol] = (messages, read_truth(path, messages))
        paths[protocol] = path
        all_keys.update(m.payload for m in messages)
    if len(all_keys) < 2:
        raise ValueError("At least two distinct source messages are needed for a validation split")
    keys = sorted(all_keys)
    random.Random(seed).shuffle(keys)
    validation = set(keys[:max(1, len(keys) // 5)])
    train, valid = {}, {}
    for protocol, (messages, truth) in pools.items():
        train[protocol] = ([m for m in messages if m.payload not in validation], truth)
        valid[protocol] = ([m for m in messages if m.payload in validation], truth)
    metadata = {
        "excluded_protocol_families": sorted(excluded), "removed_exact_target_overlap": overlap,
        "split": "seeded global unique source payload 80/20, before images/windows",
        "sources": [{"protocol": p, "path": str(paths[p]), "sha256": hashlib.sha256(paths[p].read_bytes()).hexdigest(),
                     "train_row_indices": [m.row_index for m in train[p][0]],
                     "validation_row_indices": [m.row_index for m in valid[p][0]]} for p in pools],
    }
    return train, valid, metadata


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_predictions(output: Path, messages: list[Message], predictions: list[dict]) -> None:
    if [p["message_index"] for p in predictions] != [m.row_index for m in messages]:
        raise ValueError("Prediction rows must exactly match input identities and order")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def evaluate_predictions(output: Path, path: Path, messages: list[Message], predictions: list[dict]) -> None:
    truth = read_truth(path, messages)
    metrics = [calculate_boundary_metrics(p["byte_boundaries"], truth[m.row_index], len(m.payload))
               for m, p in zip(messages, predictions)]
    write_json(output / "metrics.json", {
        "message_count": len(metrics), "average_metrics": average_boundary_metrics(metrics),
        "exact_counts": {key: sum(getattr(m, key) for m in metrics) for key in ["tp", "fp", "fn", "tn"]},
        "messages": [{"message_index": m.row_index, "truth": truth[m.row_index], "metrics": value.to_dict()}
                     for m, value in zip(messages, metrics)],
    })
