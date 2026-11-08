"""Use completed ProsegDL weights with NetPlier clusters, without retraining."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .common import (Message, bit_boundaries_to_bytes, evaluate_predictions, load_messages,
                     resolve_csv, save_predictions, write_json)
from .prepare_netplier import payload_identity, sha256
from .prosegdl import ProsegSiamese, ProsegUNet, iter_bit_windows, merge_votes, replace_text, slice_pairs
from .run_prosegdl import full_image


def load_groups(messages, path, target_sha256):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["target_sha256"] != target_sha256 or result["payload_identity_sha256"] != payload_identity([m.payload for m in messages]):
        raise ValueError("NetPlier clusters belong to a different target input")
    assignments = result["assignments"]
    if len(assignments) != len(messages):
        raise ValueError("NetPlier row count mismatch")
    groups = defaultdict(list)
    for message, assignment in zip(messages, assignments):
        if assignment["message_index"] != message.row_index or assignment["payload_sha256"] != hashlib.sha256(message.payload).hexdigest():
            raise ValueError("NetPlier row order or payload identity mismatch")
        groups[assignment["cluster_id"]].append(message)
    return groups, result


def predict(messages, groups, unet, siamese, device, purge_threshold=.2):
    transformed = {m.row_index: replace_text(m.payload) for m in messages}
    original_lengths = {m.row_index: len(m.payload) for m in messages}
    predictions, diagnostics = {}, []
    with torch.no_grad():
        for cluster_id, original in sorted(groups.items()):
            group = [Message(m.row_index, transformed[m.row_index].payload) for m in original]
            width = max(len(m.payload) for m in group) * 8
            votes, validity = defaultdict(list), defaultdict(list)
            for window in iter_bit_windows(group):
                votes[window.start_bit].append((unet(window.image[None].to(device)).sigmoid().flatten() > .5).cpu())
                validity[window.start_bit].append(window.valid.any(0))
            candidates = []
            for start in sorted(votes):
                merged = merge_votes(torch.stack(votes[start]), torch.stack(validity[start]))
                candidates.extend(start + position for position in merged.nonzero().flatten().tolist()
                                  if 0 < start + position < width)
            # A short block cannot vote on a candidate beyond its real data.
            confidence_sum = defaultdict(float)
            confidence_count = defaultdict(int)
            block_rows = []
            for first in range(0, len(group), 256):
                block = group[first:first + 256]
                block_rows.append(len(block))
                block_width = max(len(m.payload) for m in block) * 8
                selected = [p for p in candidates if p < block_width]
                if not selected:
                    continue
                image = full_image(block).to(device)
                for offset in range(0, len(selected), 64):
                    positions = selected[offset:offset + 64]
                    left, right = slice_pairs(image, positions)
                    probabilities = (1 - siamese(left, right).sigmoid()).cpu().tolist()
                    for position, score in zip(positions, probabilities):
                        confidence_sum[position] += score
                        confidence_count[position] += 1
            purged = [p for p in candidates if confidence_count[p] and confidence_sum[p] / confidence_count[p] >= purge_threshold]
            snapped = {p - p % 2 for p in purged}
            for message in group:
                # NetPlier clusters may contain different lengths. Keep each
                # message's terminal boundary and never extrapolate its body.
                end = len(message.payload) * 8
                aligned = sorted({p for p in snapped if 0 < p < end} | {0, end})
                restored = transformed[message.row_index].restore_bits(aligned)
                length = original_lengths[message.row_index]
                predictions[message.row_index] = {
                    "message_index": message.row_index, "length": length, "cluster_id": cluster_id,
                    "native_unit": "bit", "transformed_native_boundaries": aligned,
                    "native_boundaries": restored,
                    "byte_boundaries": bit_boundaries_to_bytes(restored, length),
                    "unmappable_digest_boundaries": len(aligned) - len(restored),
                }
            diagnostics.append({"cluster_id": cluster_id, "rows": len(group),
                                "minimum_transformed_bytes": min(len(m.payload) for m in group),
                                "maximum_transformed_bytes": width // 8,
                                "images_per_window_at_cluster_start": math.ceil(len(group) / 256),
                                "real_rows_per_block": block_rows,
                                "blocks_below_paper_130_rows": sum(n < 130 for n in block_rows),
                                "candidates": len(candidates), "purged_boundaries": len(purged)})
    return [predictions[m.row_index] for m in messages], diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--purge-threshold", type=float, default=.2, help="Siamese boundary-confidence threshold (default: 0.2)")
    args = parser.parse_args()
    if not 0 <= args.purge_threshold <= 1:
        parser.error("Purge threshold must be between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Preserving existing run: {args.output}")
    torch.set_num_threads(args.threads)
    target = resolve_csv(args.protocol)
    messages = load_messages(target)
    groups, cluster_result = load_groups(messages, args.clusters, sha256(target))
    training_manifest = json.loads((args.training_run / "manifest.json").read_text(encoding="utf-8"))
    if training_manifest["status"] != "completed" or training_manifest["arguments"]["protocol"] != args.protocol:
        raise ValueError("Require a completed training run for the same held-out target")
    if training_manifest["target_sha256"] != sha256(target):
        raise ValueError("Training and inference target input mismatch")
    for name in ["prosegdl.py", "run_prosegdl.py"]:
        if training_manifest["implementation_sha256"][name] != sha256(Path(__file__).with_name(name)):
            raise ValueError(f"Training model implementation changed: {name}")
    checkpoint_path = args.training_run / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    unet, siamese = ProsegUNet().to(args.device), ProsegSiamese().to(args.device)
    unet.load_state_dict(checkpoint["unet"])
    siamese.load_state_dict(checkpoint["siamese"])
    unet.eval()
    siamese.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "started", "run_type": training_manifest["run_type"],
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "target_sha256": sha256(target), "training_checkpoint_sha256": sha256(checkpoint_path),
                "training_manifest_sha256": sha256(args.training_run / "manifest.json"),
                "clusters_sha256": sha256(args.clusters), "adapter_sha256": sha256(Path(__file__)),
                "target_labels_used_before_predictions_saved": False,
                "target_grouping": "upstream NetPlier inferred keyword + observed direction; no length subdivision",
                "remote_coupling_available": cluster_result["remote_coupling_available"],
                "clustering_byte_cap": 500, "prosegdl_inference_byte_cap": None,
                "thresholds": {"binary": .5, "vote_minimum_support_strict": .1, "purge": args.purge_threshold},
                "source_training_grouping": training_manifest["settings"]["source_grouping"],
                "source_training_unchanged": True}
    write_json(args.output / "manifest.json", manifest)
    predictions, diagnostics = predict(messages, groups, unet, siamese, args.device, args.purge_threshold)
    save_predictions(args.output, messages, predictions)
    evaluate_predictions(args.output, target, messages, predictions)
    manifest.update(status="completed", target_groups=diagnostics)
    write_json(args.output / "manifest.json", manifest)
    print({"completed": args.protocol, "rows": len(predictions), "clusters": len(groups)}, flush=True)


if __name__ == "__main__":
    main()
