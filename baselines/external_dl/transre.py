"""TransRE syntax network and domain adaptation."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

WINDOW = 64


def window_messages(messages, truth=None):
    values, masks, labels, identity = [], [], [], []
    for message in messages:
        for start in range(0, len(message.payload), WINDOW):
            payload = message.payload[start:start + WINDOW]
            vector = torch.full((WINDOW,), -1.)
            vector[:len(payload)] = torch.tensor(list(payload)).float() / 255
            mask = torch.arange(WINDOW) < len(payload)
            label = torch.zeros(WINDOW, dtype=torch.long)
            if truth is not None:
                for boundary in truth[message.row_index]:
                    position = boundary - 1 - start
                    if 0 <= position < len(payload):
                        label[position] = 1
            values.append(vector)
            masks.append(mask)
            labels.append(label)
            identity.append((message.row_index, start, len(payload)))
    if not values:
        raise ValueError("No nonempty messages for TransRE")
    return torch.stack(values), torch.stack(masks), torch.stack(labels), identity


class TransNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(WINDOW, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU())
        self.bottleneck_linear = nn.Linear(512, WINDOW * 64)
        self.bottleneck = nn.Sequential(nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.5))
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(.5), nn.Linear(64, 2), nn.ReLU())

    def forward(self, values: Tensor):
        batch = values.shape[0]
        feature = self.bottleneck_linear(self.base(values)).reshape(-1, 64)
        feature = self.bottleneck(feature).reshape(batch, WINDOW, 64)
        return feature, self.classifier(feature)


def covariance_distance(source: Tensor, target: Tensor) -> Tensor:
    """Unbiased covariance distance between source and target features."""
    if min(len(source), len(target)) < 2:
        raise ValueError("Covariance needs at least two samples per domain")
    a = source - source.mean(0)
    b = target - target.mean(0)
    covariance_a = a.T @ a / (len(source) - 1)
    covariance_b = b.T @ b / (len(target) - 1)
    return torch.linalg.matrix_norm(covariance_a - covariance_b) / (4 * min(len(source), len(target)) ** 2)


def local_mmd(source: Tensor, target: Tensor, source_labels: Tensor, target_probabilities: Tensor) -> Tensor:
    """Class-conditional Gaussian RKHS distance for domain adaptation."""
    source_weights = F.one_hot(source_labels, num_classes=2).to(source.dtype)
    target_weights = target_probabilities.detach()
    common = set(source_labels.unique().tolist()) & set(target_weights.argmax(1).unique().tolist())
    if not common:
        return (source.sum() + target.sum()) * 0
    features = torch.cat((source, target))
    squared = torch.cdist(features, features).square()
    n = len(features)
    bandwidth = (squared.detach().sum() / max(1, n * (n - 1))).clamp_min(1e-8)
    kernel = sum(torch.exp(-squared / (bandwidth * 2. ** shift)) for shift in [-2, -1, 0, 1, 2])
    source_weights = source_weights / source_weights.sum(0).clamp_min(1e-8)
    target_weights = target_weights / target_weights.sum(0).clamp_min(1e-8)
    total = features.sum() * 0
    for label in sorted(common):
        weights = torch.cat((source_weights[:, label], -target_weights[:, label]))
        total = total + weights @ kernel @ weights
    return total / len(common)
