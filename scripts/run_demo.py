"""Decode example messages with the fusion decoder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhf.common.messages import MessageBatch
from nhf.common.models import BoundaryPrior
from nhf.fusion.constrained_decoder import ConstrainedBoundaryDecoder
from nhf.heuristic.native_engine import StandaloneHeuristicEngine


def read_messages(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row["group"], []).append(row["hex"].strip())
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, default=ROOT / "examples" / "sample_messages.csv")
    parser.add_argument("--scores", type=Path, default=ROOT / "examples" / "sample_boundary_scores.json")
    args = parser.parse_args()

    scores_by_group = json.loads(args.scores.read_text(encoding="utf-8"))
    output = []
    for group, messages in read_messages(args.messages).items():
        batch = MessageBatch.from_iterable(messages, group_label=group)
        engine = StandaloneHeuristicEngine()
        evidence = engine.infer(batch)
        heuristic_boundaries = engine.infer_flat_boundaries(batch)

        scores = [float(value) for value in scores_by_group.get(group, [])]
        scores = (scores + [0.0] * batch.max_length)[: batch.max_length]
        prior = BoundaryPrior(source="supplied_boundary_scores", scores=scores)

        decoder = ConstrainedBoundaryDecoder()
        predictions = decoder.decode_boundaries(
            batch, evidence.constraints, prior, heuristic_boundaries
        )
        output.append(
            {
                "group": group,
                "messages": messages,
                "heuristic_boundaries": heuristic_boundaries,
                "fused_boundaries": predictions,
                "num_constraints": len(evidence.constraints),
            }
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
