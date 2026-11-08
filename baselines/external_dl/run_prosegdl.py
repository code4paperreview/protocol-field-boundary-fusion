"""ProsegDL training and inference."""

import math
import random
from collections import defaultdict

import torch
from torch.nn import functional as F

from .common import Message, bit_boundaries_to_bytes, write_json
from .prosegdl import (
    ProsegSiamese, ProsegUNet, iter_bit_windows, merge_votes, replace_text, slice_pairs, weighted_mse,
)


def source_groups(pool):
    groups = defaultdict(list)
    for protocol, (messages, truth) in pool.items():
        for message in messages:
            transformed = replace_text(message.payload)
            offsets = tuple(i * 8 for i, original in enumerate(transformed.original_offsets)
                            if original in truth[message.row_index] and i < len(transformed.payload))
            groups[protocol, len(transformed.payload), offsets].append(Message(message.row_index, transformed.payload))
    return [(messages, key[2]) for key, messages in groups.items()]


def training_images(groups, seed, shuffle):
    rng = random.Random(seed)
    groups = list(groups)
    if shuffle:
        rng.shuffle(groups)
    for original, boundaries in groups:
        messages = list(original)
        if shuffle:
            rng.shuffle(messages)
        for window in iter_bit_windows(messages):
            target = torch.zeros((1, 1, 256))
            for offset in boundaries:
                relative = offset - window.start_bit
                if 0 <= relative < 256:
                    target[0, 0, relative] = 1
            yield window.image, target


def image_batches(groups, batch_size, seed, shuffle):
    images, targets = [], []
    for image, target in training_images(groups, seed, shuffle):
        images.append(image)
        targets.append(target)
        if len(images) == batch_size:
            yield torch.stack(images), torch.stack(targets)
            images, targets = [], []
    if images:
        yield torch.stack(images), torch.stack(targets)


def full_image(messages):
    windows = list(iter_bit_windows(messages))
    # Callers supply at most 256 messages, so all windows share row identities.
    return torch.cat([window.image[0] for window in windows], dim=1)


def pair_batches(groups, batch_size, seed, shuffle):
    rng = random.Random(seed)
    groups = list(groups)
    if shuffle:
        rng.shuffle(groups)
    left_batch, right_batch, labels = [], [], []
    for original, boundaries in groups:
        messages = list(original)
        if shuffle:
            rng.shuffle(messages)
        true_positions = set(boundaries)
        for start in range(0, len(messages), 256):
            group = messages[start:start + 256]
            image = full_image(group)
            width = max(len(m.payload) for m in group) * 8
            positions = [p for p in range(1, width) if p in true_positions or rng.random() < .1]
            for first in range(0, len(positions), batch_size):
                selected = positions[first:first + batch_size]
                left, right = slice_pairs(image, selected)
                for a, b, p in zip(left, right, selected):
                    left_batch.append(a)
                    right_batch.append(b)
                    labels.append(float(p not in true_positions))
                    if len(labels) == batch_size:
                        yield torch.stack(left_batch), torch.stack(right_batch), torch.tensor(labels)
                        left_batch, right_batch, labels = [], [], []
    if labels:
        yield torch.stack(left_batch), torch.stack(right_batch), torch.tensor(labels)


def fit(unet, siamese, train, valid, args, output, device):
    history = []
    for stage, model, epochs in [("unet", unet, args.epochs), ("siamese", siamese, args.siamese_epochs)]:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-7, momentum=.8)
        best = float("inf")
        path = output / f"{stage}_best.pt"
        for epoch in range(epochs):
            row = {"stage": stage, "epoch": epoch + 1}
            for name, groups in [("train", train), ("validation", valid)]:
                model.train(name == "train")
                total, count = 0.0, 0
                if stage == "unet":
                    batches = image_batches(groups, args.batch_size, args.seed + (epoch if name == "train" else 0), name == "train")
                else:
                    batches = pair_batches(groups, 64, args.seed + (epoch if name == "train" else 0), name == "train")
                with torch.set_grad_enabled(name == "train"):
                    for batch in batches:
                        batch = [value.to(device) for value in batch]
                        if name == "train":
                            optimizer.zero_grad(set_to_none=True)
                        if stage == "unet":
                            loss = weighted_mse(model(batch[0]), batch[1], .8)
                        else:
                            loss = F.binary_cross_entropy_with_logits(model(batch[0], batch[1]), batch[2])
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite {stage} {name} loss")
                        if name == "train":
                            loss.backward()
                            optimizer.step()
                        total += loss.item() * len(batch[0])
                        count += len(batch[0])
                if not count:
                    raise ValueError(f"No {stage} {name} examples")
                row[name + "_loss"] = total / count
            if row["validation_loss"] < best:
                best = row["validation_loss"]
                torch.save(model.state_dict(), path)
            history.append(row)
            write_json(output / "history.json", history)
            print(row, flush=True)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def run(messages, train_pool, valid_pool, args, output, device):
    train, valid = source_groups(train_pool), source_groups(valid_pool)
    unet, siamese = ProsegUNet().to(device), ProsegSiamese().to(device)
    fit(unet, siamese, train, valid, args, output, device)
    torch.save({"unet": unet.state_dict(), "siamese": siamese.state_dict(), "arguments": vars(args)}, output / "checkpoint.pt")
    unet.eval()
    siamese.eval()
    transformed = {m.row_index: replace_text(m.payload) for m in messages}
    original_lengths = {m.row_index: len(m.payload) for m in messages}
    groups = defaultdict(list)
    for message in messages:
        replacement = transformed[message.row_index]
        groups[len(replacement.payload)].append(Message(message.row_index, replacement.payload))
    predictions, grouping = {}, []
    with torch.no_grad():
        for length, group in sorted(groups.items()):
            votes = defaultdict(list)
            validity = defaultdict(list)
            for window in iter_bit_windows(group):
                votes[window.start_bit].append((unet(window.image[None].to(device)).sigmoid().flatten() > .5).cpu())
                validity[window.start_bit].append(window.valid.any(0))
            candidates = []
            for start in sorted(votes):
                merged = merge_votes(torch.stack(votes[start]), torch.stack(validity[start]))
                candidates.extend(start + p for p in merged.nonzero().flatten().tolist() if 0 < start + p < length * 8)
            # Purge each candidate across image blocks, then average confidence.
            purging = []
            for first in range(0, len(group), 256):
                image = full_image(group[first:first + 256]).to(device)
                probabilities = []
                for start in range(0, len(candidates), 64):
                    left, right = slice_pairs(image, candidates[start:start + 64])
                    probabilities.extend((1 - siamese(left, right).sigmoid()).cpu().tolist())
                purging.append(torch.tensor(probabilities))
            confidence = torch.stack(purging).mean(0) if candidates else torch.empty(0)
            purged = [p for p, score in zip(candidates, confidence.tolist()) if score >= args.purge_threshold]
            aligned = sorted({p - p % 2 for p in purged} | {0, length * 8})
            for message in group:
                restored = transformed[message.row_index].restore_bits(aligned)
                original_length = original_lengths[message.row_index]
                predictions[message.row_index] = {
                    "message_index": message.row_index, "length": original_length,
                    "native_unit": "bit", "transformed_native_boundaries": aligned,
                    "native_boundaries": restored, "byte_boundaries": bit_boundaries_to_bytes(restored, original_length),
                    "unmappable_digest_boundaries": len(aligned) - len(restored),
                }
            grouping.append({"transformed_bytes": length, "rows": len(group), "images": math.ceil(len(group) / 256),
                             "minimum_real_rows_per_image": min(256, len(group) % 256 or 256),
                             "below_paper_130_rows": len(group) < 130 or 0 < len(group) % 256 < 130,
                             "candidates": len(candidates), "purged_boundaries": len(purged)})
    return [predictions[m.row_index] for m in messages], {
        "source_grouping": "source protocol + transformed length + supervised boundary pattern",
        "target_grouping": "transformed byte length only; no target labels or session identifiers",
        "target_groups": grouping, "boundary_weight": .8, "binary_threshold": .5,
        "minimum_support": .1, "purge_threshold": args.purge_threshold, "snap": "nearest even bit, tie downward",
        "text_run_minimum": 5, "text_digest": "MD5 first four bytes, with inverse boundary mapping",
    }
