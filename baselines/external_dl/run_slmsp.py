"""SLMSP training and inference."""

import random

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .common import write_json
from .slmsp import (
    GloveModel, WordOrderModel, byte_relation_curve, cooccurrences,
    horizontal_boundaries, positional_entropy, tokenize_messages, vertical_correction,
)


def next_token_step(model, tokens, optimizer, classifier_chunk=256):
    """One next-token training step with chunked classifier gradients."""
    optimizer.zero_grad(set_to_none=True)
    mask = tokens[:, 1:] != 0
    hidden = model.encode(tokens[:, :-1])[mask]
    targets = tokens[:, 1:][mask]
    if not targets.numel():
        return 0.0
    detached = hidden.detach().requires_grad_()
    loss_value = 0.0
    for start in range(0, len(targets), classifier_chunk):
        logits = model.classifier(detached[start:start + classifier_chunk])
        loss = F.cross_entropy(logits, targets[start:start + classifier_chunk], reduction="sum") / len(targets)
        loss.backward()
        loss_value += loss.item()
    hidden.backward(detached.grad)
    optimizer.step()
    return loss_value


def run(messages, args, output, device):
    payloads = [m.payload for m in messages]
    encoded, vocabulary = tokenize_messages(payloads)
    model = WordOrderModel(len(vocabulary) + 1).to(device)
    glove = GloveModel(len(vocabulary) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    glove_optimizer = torch.optim.Adam(glove.parameters(), lr=.001)
    rng = random.Random(args.seed)
    eligible = [i for i, sequence in enumerate(encoded) if len(sequence) >= 2]
    history = []
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(eligible)
        losses = []
        for start in range(0, len(eligible), args.batch_size):
            batch = eligible[start:start + args.batch_size]
            tokens = pad_sequence([torch.tensor(encoded[i]) for i in batch], batch_first=True).to(device)
            losses.append(next_token_step(model, tokens, optimizer))
        row = {"stage": "word_order", "epoch": epoch + 1, "mean_batch_loss": sum(losses) / max(1, len(losses))}
        history.append(row)
        write_json(output / "history.json", history)
        print(row, flush=True)
    words, contexts, counts = cooccurrences(encoded, window=5)
    best_glove, stale_epochs, glove_epochs_done = float("inf"), 0, 0
    for epoch in range(args.glove_epochs):
        glove.train()
        order = torch.randperm(len(counts))
        total, seen = 0.0, 0
        for indices in order.split(4096):
            glove_optimizer.zero_grad(set_to_none=True)
            loss = glove.loss(words[indices].to(device), contexts[indices].to(device), counts[indices].to(device), xmax=100, alpha=.75)
            loss.backward()
            glove_optimizer.step()
            total += loss.item() * len(indices)
            seen += len(indices)
        row = {"stage": "glove", "epoch": epoch + 1, "loss": total / max(1, seen)}
        history.append(row)
        write_json(output / "history.json", history)
        print(row, flush=True)
        glove_epochs_done = epoch + 1
        if best_glove == float("inf") or row["loss"] < best_glove - max(1e-6, abs(best_glove) * .001):
            best_glove, stale_epochs = row["loss"], 0
        else:
            stale_epochs += 1
        if stale_epochs >= 10:
            break
    torch.save({"word_order": model.state_dict(), "glove": glove.state_dict(),
                "vocabulary": [(value, offset, token) for (value, offset), token in vocabulary.items()],
                "arguments": vars(args)}, output / "checkpoint.pt")
    entropy = positional_entropy(payloads)
    model.eval()
    glove.eval()
    predictions = []
    with torch.no_grad():
        for message, sequence in zip(messages, encoded):
            tokens = torch.tensor([sequence], device=device)
            order_embedding = model.encode(tokens)[0]
            curve = byte_relation_curve(order_embedding, glove(tokens)[0]).cpu().tolist()
            horizontal = horizontal_boundaries(curve)
            boundaries = vertical_correction(horizontal, entropy, len(message.payload))
            predictions.append({"message_index": message.row_index, "length": len(message.payload),
                                "native_unit": "byte", "horizontal_boundaries": horizontal,
                                "native_boundaries": boundaries, "byte_boundaries": boundaries})
    return predictions, {"training": "all selected target messages, self-supervised; no target boundary labels",
                         "vocabulary_size": len(vocabulary) + 1, "glove_window": 5, "glove_xmax": 100,
                         "glove_alpha": .75, "beta": .85, "target_entropy": entropy,
                         "glove_epochs_completed": glove_epochs_done,
                         "glove_stopping": "up to configured maximum; 10 epochs without max(1e-6,0.1% best loss) improvement"}
