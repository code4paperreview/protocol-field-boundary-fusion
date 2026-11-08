"""ProsegDL network and preprocessing."""

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from .common import Message


def _conv(in_channels: int, out_channels: int, stride=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.01),
    )


def _block(in_channels: int, out_channels: int, unequal=False) -> nn.Sequential:
    return nn.Sequential(
        _conv(in_channels, out_channels),
        _conv(out_channels, out_channels, stride=(2, 1) if unequal else 1),
    )


class ProsegUNet(nn.Module):
    """Fig. 2: [batch, 1, 256, 256] -> [batch, 1, 1, 256] logits."""

    def __init__(self):
        super().__init__()
        channels = [64, 128, 256, 512, 1024]
        self.encoders = nn.ModuleList(
            _block(1 if i == 0 else channels[i - 1], channel)
            for i, channel in enumerate(channels)
        )
        self.downsamples = nn.ModuleList(_conv(c, c, stride=2) for c in channels[:-1])
        self.skips = nn.ModuleList(
            nn.Sequential(*[_block(c, c, unequal=True) for _ in range(3 - i)])
            for i, c in enumerate(channels[:-1])
        )
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(c, c // 2, 2, stride=2) for c in reversed(channels[1:])
        )
        self.decoders = nn.ModuleList(
            _block(c, c // 2, unequal=True) for c in reversed(channels[1:])
        )
        self.tail = nn.Sequential(
            *[_block(a, b, unequal=True) for a, b in [(64, 32), (32, 16), (16, 8), (8, 1)]]
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (1, 256, 256):
            raise ValueError("ProsegUNet requires [batch, 1, 256, 256] input")
        skips = []
        value = image
        for i, encoder in enumerate(self.encoders):
            value = encoder(value)
            if i < 4:
                skips.append(self.skips[i](value))
                value = self.downsamples[i](value)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            value = decoder(torch.cat((skip, upsample(value)), dim=1))
        return self.tail(value)


def weighted_mse(logits: Tensor, targets: Tensor, boundary_weight: float) -> Tensor:
    """Weighted MSE over boundary and non-boundary bit positions."""
    if not 0 <= boundary_weight <= 1:
        raise ValueError("boundary_weight must be between zero and one")
    if logits.shape != targets.shape:
        raise ValueError("Logits and targets must have the same shape")
    weights = boundary_weight * targets + (1 - boundary_weight) * (1 - targets)
    return (weights * (logits.sigmoid() - targets).square()).flatten(1).sum(1).mean()


@dataclass(frozen=True)
class BitWindow:
    row_indices: tuple[int, ...]
    start_bit: int
    image: Tensor
    valid: Tensor


def iter_bit_windows(messages: Sequence[Message]) -> Iterator[BitWindow]:
    """Yield 256x256 bit images over consecutive 32-byte windows."""
    shifts = torch.arange(7, -1, -1)
    for first in range(0, len(messages), 256):
        group = messages[first:first + 256]
        for start in range(0, max(len(m.payload) for m in group), 32):
            image = torch.full((1, 256, 256), -1.0)
            valid = torch.zeros((256, 256), dtype=torch.bool)
            for row, message in enumerate(group):
                values = torch.tensor(list(message.payload[start:start + 32]), dtype=torch.long)
                bits = ((values[:, None] >> shifts) & 1).flatten()
                image[0, row, :bits.numel()] = bits.float()
                valid[row, :bits.numel()] = True
            yield BitWindow(tuple(m.row_index for m in group), start * 8, image, valid)


def merge_votes(predictions: Tensor, valid: Tensor, minimum_support=0.1) -> Tensor:
    """Merge binary predictions using strict Pi > Pmin (p. 5), ignoring padding.

    Inputs are [images, positions] at the SAME absolute window coordinates.
    Binary conversion is performed by the runner before this merge step.
    """
    if predictions.shape != valid.shape or predictions.ndim != 2:
        raise ValueError("Predictions and validity masks must have shape [images, positions]")
    if predictions.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError("Voting requires binary predictions and boolean validity masks")
    count = valid.sum(0)
    support = (predictions & valid).sum(0) / count.clamp_min(1)
    return (count > 0) & (support > minimum_support)


@dataclass(frozen=True)
class TextReplacement:
    payload: bytes
    # Transformed byte boundaries -> original offsets. Interior digest offsets
    # have no unambiguous original coordinate and are intentionally None.
    original_offsets: tuple[int | None, ...]

    def restore_bits(self, offsets: Sequence[int]) -> list[int]:
        restored = set()
        for offset in offsets:
            byte, bit = divmod(offset, 8)
            if not 0 <= byte < len(self.original_offsets):
                continue
            original = self.original_offsets[byte]
            if original is None:
                continue
            if bit == 0:
                restored.add(8 * original)
            elif byte + 1 < len(self.original_offsets) and self.original_offsets[byte + 1] == original + 1:
                restored.add(8 * original + bit)
        return sorted(restored)


def replace_text(payload: bytes) -> TextReplacement:
    """Replace printable ASCII runs of at least five bytes by a four-byte digest."""
    result = bytearray()
    coordinates = [0]
    position = 0
    while position < len(payload):
        end = position
        while end < len(payload) and 32 <= payload[end] <= 126:
            end += 1
        if end - position >= 5:
            result.extend(hashlib.md5(payload[position:end]).digest()[:4])
            coordinates.extend([None, None, None, end])
            position = end
        else:
            result.append(payload[position])
            position += 1
            coordinates.append(position)
    return TextReplacement(bytes(result), tuple(coordinates))


class SiameseBackbone(nn.Module):
    """Fig. 3 LSTM-FCN; dimension shuffle gives one step with 256 features."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        layers = []
        for a, b, kernel in [(1, 128, 8), (128, 256, 5), (256, 128, 3)]:
            layers.extend([nn.Conv1d(a, b, kernel, padding="same"), nn.BatchNorm1d(b), nn.ReLU()])
        self.convolution = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        # Preserve -1 padding; map actual 0..255 values into 0..1.
        values = torch.where(values < 0, -torch.ones_like(values), values / 255)
        _, (hidden, _) = self.lstm(values[:, None, :])
        recurrent = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        convolved = self.convolution(values[:, None, :]).mean(-1)
        return torch.cat((recurrent, convolved), dim=1)


class ProsegSiamese(nn.Module):
    """Similarity logits; BCE target = 1 - field-boundary label (PDF p. 4)."""

    def __init__(self):
        super().__init__()
        self.backbone = SiameseBackbone()
        self.classifier = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        distance = (self.backbone(left) - self.backbone(right)).square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()
        return self.classifier(distance).squeeze(-1)


def slice_pairs(image: Tensor, positions: Sequence[int]) -> tuple[Tensor, Tensor]:
    """Extract two 8-bit columns at arbitrary bit offsets; invalid rows stay -1.

    Image shape [256, full_message_bits]. A missing bit in either slice makes
    that slice's row invalid. This also handles context at 32-byte window edges.
    """
    slices = []
    weights = 2 ** torch.arange(7, -1, -1, device=image.device)
    padded = torch.nn.functional.pad(image, (8, 8), value=-1)
    for side in [-8, 0]:
        columns = []
        for position in positions:
            bits = padded[:, position + side + 8:position + side + 16]
            value = (bits * weights).sum(1)
            columns.append(value.masked_fill((bits < 0).any(1), -1))
        slices.append(torch.stack(columns) if columns else image.new_empty((0, 256)))
    return slices[0], slices[1]
