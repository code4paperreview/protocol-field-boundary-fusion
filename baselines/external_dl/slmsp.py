"""SLMSP tokenization and segmentation components."""

import math
from collections import Counter
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def tokenize_messages(messages: Sequence[bytes]) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    """byte#offset vocabulary; token 0 is reserved for paper's 256#NAN padding."""
    vocabulary = {}
    encoded = []
    for payload in messages:
        sequence = []
        for offset, value in enumerate(payload):
            token = (value, offset)
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary) + 1
            sequence.append(vocabulary[token])
        encoded.append(sequence)
    return encoded, vocabulary


class WordOrderModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 256, padding_idx=0)
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)
        self.lstm = nn.LSTM(256, 512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, vocabulary_size)

    def encode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0 or (tokens[:, 0] == 0).any():
            raise ValueError("Supply a batch of nonempty token sequences with right padding")
        embedding = self.embedding(tokens)
        scores = self.query(embedding) @ self.key(embedding).transpose(1, 2) / math.sqrt(256)
        future = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future | (tokens == 0)[:, None, :], float("-inf"))
        attended = scores.softmax(-1) @ self.value(embedding)
        hidden, _ = self.lstm(attended)
        hidden = hidden.masked_fill((tokens == 0)[:, :, None], 0)
        return hidden

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encode(tokens)
        return hidden, self.classifier(hidden)


def cooccurrences(sequences: Sequence[Sequence[int]], window: int) -> tuple[Tensor, Tensor, Tensor]:
    """Sparse symmetric co-occurrences with inverse-distance weighting."""
    if window < 1:
        raise ValueError("Co-occurrence window must be positive")
    counts = Counter()
    for sequence in sequences:
        for i, word in enumerate(sequence):
            for j in range(max(0, i - window), i):
                context = sequence[j]
                if word and context:
                    counts[word, context] += 1 / (i - j)
                    counts[context, word] += 1 / (i - j)
    pairs = list(counts)
    return (
        torch.tensor([a for a, _ in pairs], dtype=torch.long),
        torch.tensor([b for _, b in pairs], dtype=torch.long),
        torch.tensor(list(counts.values()), dtype=torch.float),
    )


class GloveModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.word = nn.Embedding(vocabulary_size, 512, padding_idx=0)
        self.context = nn.Embedding(vocabulary_size, 512, padding_idx=0)
        self.word_bias = nn.Embedding(vocabulary_size, 1, padding_idx=0)
        self.context_bias = nn.Embedding(vocabulary_size, 1, padding_idx=0)
        nn.init.uniform_(self.word.weight, -0.5 / 512, 0.5 / 512)
        nn.init.uniform_(self.context.weight, -0.5 / 512, 0.5 / 512)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)
        with torch.no_grad():
            self.word.weight[0].zero_()
            self.context.weight[0].zero_()

    def loss(self, words: Tensor, contexts: Tensor, counts: Tensor, *, xmax: float, alpha: float) -> Tensor:
        prediction = (self.word(words) * self.context(contexts)).sum(-1)
        prediction = prediction + self.word_bias(words).squeeze(-1) + self.context_bias(contexts).squeeze(-1)
        weights = (counts / xmax).pow(alpha).clamp(max=1)
        return (weights * (prediction - counts.log()).square()).mean()

    def forward(self, tokens: Tensor) -> Tensor:
        return self.word(tokens) + self.context(tokens)


def byte_relation_curve(order: Tensor, semantics: Tensor) -> Tensor:
    """Paper Eq. 8--9; accepts ONE unpadded message's [bytes, 512] embeddings."""
    if order.shape != semantics.shape or order.ndim != 2:
        raise ValueError("Both embeddings must have shape [bytes, dimensions]")
    fused = 0.15 * order + 0.85 * semantics
    return F.cosine_similarity(fused[:-1], fused[1:], dim=-1)


def horizontal_boundaries(curve: Sequence[float]) -> list[int]:
    """Select boundaries from the byte-relation curve."""
    boundaries = {0, len(curve) + 1}
    previous, rising = float("-inf"), True
    for k, relation in enumerate(curve):
        if relation > previous and not rising:
            boundaries.add(k)
            rising = True
        elif relation < previous and rising:
            boundaries.add(k + 1)
            rising = False
        previous = relation
    return sorted(boundaries)


def positional_entropy(messages: Sequence[bytes]) -> list[float]:
    """Use present bytes only; missing/padded positions are not observed values."""
    entropies = []
    for offset in range(max(map(len, messages), default=0)):
        counts = Counter(message[offset] for message in messages if len(message) > offset)
        total = sum(counts.values())
        entropies.append(-sum((count / total) * math.log2(count / total) for count in counts.values()))
    return entropies


def vertical_correction(boundaries: Sequence[int], entropy: Sequence[float], length: int) -> list[int]:
    """PDF p. 8: merge constant/constant; split constant/variable; else retain."""
    if len(entropy) < length:
        raise ValueError("Entropy does not cover the complete message")
    corrected = {p for p in boundaries if 0 < p < length}
    for offset in range(1, length):
        if entropy[offset - 1] == 0:
            if entropy[offset] == 0:
                corrected.discard(offset)
            else:
                corrected.add(offset)
    return sorted(corrected | {0, length})
