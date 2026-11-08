"""TransRE training and inference."""

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from .common import field_end_labels_to_boundaries, write_json
from .transre import WINDOW, TransNet, covariance_distance, local_mmd, window_messages


def select_source(train_pool, target, args, output, device):
    histories = {}
    for protocol in sorted(train_pool):
        messages, truth = train_pool[protocol]
        if not messages:
            continue
        values, _, labels, _ = window_messages(messages, truth)
        # Same initialization and sampling seed for every candidate comparison.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(args.seed)
            projection = nn.Sequential(nn.Linear(WINDOW * 2, 256), nn.ReLU(), nn.Linear(256, 64)).to(device)
        optimizer = torch.optim.Adam(projection.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(args.seed)
        scores = []
        for _ in range(args.selection_steps):
            count = max(2, args.batch_size)
            indices_s = torch.randint(len(values), (count,), generator=generator)
            indices_t = torch.randint(len(target[0]), (count,), generator=generator)
            source_input = torch.cat((values[indices_s], labels[indices_s].float()), 1).to(device)
            target_input = torch.cat((target[0][indices_t], torch.zeros_like(target[0][indices_t])), 1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = covariance_distance(projection(source_input), projection(target_input))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite source-selection covariance")
            scores.append(loss.item())
            loss.backward()
            optimizer.step()
        histories[protocol] = scores
    if not histories:
        raise ValueError("No source training messages")
    votes = {protocol: 0 for protocol in histories}
    for step in range(args.selection_steps):
        winner = min(histories, key=lambda p: (histories[p][step], p))
        votes[winner] += 1
    selected = min(votes, key=lambda p: (-votes[p], p))
    write_json(output / "source_selection.json", {"selected": selected, "minimum_distance_votes": votes,
                                                   "covariance_history": histories})
    print({"selected_source": selected, "votes": votes}, flush=True)
    return selected


def run(messages, train_pool, valid_pool, args, output, device):
    target = window_messages(messages)
    original_lengths = {message.row_index: len(message.payload) for message in messages}
    target_clusters = defaultdict(list)
    for index, (row, _, _) in enumerate(target[3]):
        target_clusters[original_lengths[row]].append(index)
    cluster_indices = {length: torch.tensor(indices) for length, indices in target_clusters.items()}
    eligible = {p: pool for p, pool in train_pool.items() if pool[0] and valid_pool[p][0]}
    selected = select_source(eligible, target, args, output, device)
    source = window_messages(*train_pool[selected])
    validation = window_messages(*valid_pool[selected])
    model = TransNet().to(device)
    optimizer = torch.optim.SGD([
        {"params": model.base.parameters(), "lr": .01},
        {"params": model.bottleneck_linear.parameters(), "lr": .1},
        {"params": model.bottleneck.parameters(), "lr": .1},
        {"params": model.classifier.parameters(), "lr": .1},
    ], momentum=.9, weight_decay=5e-4)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        totals = {"source_ce": 0., "lmmd": 0.}
        steps = 0
        for indices in torch.randperm(len(source[0])).split(args.batch_size):
            anchor = torch.randint(len(target[0]), (1,)).item()
            length = original_lengths[target[3][anchor][0]]
            cluster = cluster_indices[length]
            target_indices = cluster[torch.randint(len(cluster), (len(indices),))]
            values_s, mask_s, labels_s = [part[indices].to(device) for part in source[:3]]
            values_t, mask_t = [part[target_indices].to(device) for part in target[:2]]
            optimizer.zero_grad(set_to_none=True)
            # Joint forward gives BN the combined domain distribution.
            features, logits = model(torch.cat((values_s, values_t)))
            feature_s, feature_t = features[:len(indices)][mask_s], features[len(indices):][mask_t]
            logits_s, logits_t = logits[:len(indices)][mask_s], logits[len(indices):][mask_t]
            labels = labels_s[mask_s]
            ce = F.cross_entropy(logits_s, labels)
            # Bound quadratic kernel memory; uniformly sample real bytes only.
            subset_s = torch.randperm(len(feature_s), device=device)[:512]
            subset_t = torch.randperm(len(feature_t), device=device)[:512]
            transfer = local_mmd(feature_s[subset_s], feature_t[subset_t], labels[subset_s], logits_t[subset_t].softmax(-1))
            loss = ce + .01 * transfer
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite TransRE training loss")
            loss.backward()
            optimizer.step()
            totals["source_ce"] += ce.item()
            totals["lmmd"] += transfer.item()
            steps += 1
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for indices in torch.arange(len(validation[0])).split(args.batch_size):
                values, mask, labels = [part[indices].to(device) for part in validation[:3]]
                _, logits = model(values)
                total += F.cross_entropy(logits[mask], labels[mask], reduction="sum").item()
                count += mask.sum().item()
        validation_loss = total / count
        row = {"epoch": epoch + 1, **{key: value / steps for key, value in totals.items()}, "validation_ce": validation_loss}
        if validation_loss < best:
            best = validation_loss
            torch.save(model.state_dict(), output / "transnet_best.pt")
        history.append(row)
        write_json(output / "history.json", history)
        print(row, flush=True)
    model.load_state_dict(torch.load(output / "transnet_best.pt", map_location=device, weights_only=True))
    torch.save({"transnet": model.state_dict(), "selected_source": selected, "arguments": vars(args)}, output / "checkpoint.pt")
    model.eval()
    boundaries = {message.row_index: {0, len(message.payload)} for message in messages}
    with torch.no_grad():
        for indices in torch.arange(len(target[0])).split(args.batch_size):
            _, logits = model(target[0][indices].to(device))
            labels = logits.argmax(-1).cpu()
            for index, predicted in zip(indices.tolist(), labels):
                row, start, length = target[3][index]
                boundaries[row].update(start + offset for offset in field_end_labels_to_boundaries(predicted[:length].tolist()))
    predictions = [{"message_index": message.row_index, "length": len(message.payload), "native_unit": "byte",
                    "native_boundaries": sorted(boundaries[message.row_index]),
                    "byte_boundaries": sorted(boundaries[message.row_index])} for message in messages]
    return predictions, {"selected_source": selected, "window_bytes": WINDOW, "lambda_lmmd": .01,
                         "target_data_use": "unlabeled source selection and domain adaptation",
                         "lmmd_max_real_bytes_per_domain_batch": 512,
                         "ineligible_source_candidates": sorted(set(train_pool) - set(eligible)),
                         "target_grouping": "exact original message length as operational homogeneity proxy; no recursive content test",
                         "target_clusters": [{"original_bytes": length, "windows": len(indices)}
                                             for length, indices in sorted(target_clusters.items())]}
