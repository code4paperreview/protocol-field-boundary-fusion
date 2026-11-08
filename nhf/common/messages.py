from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import Span

_HEX_RE = re.compile(r"^[0-9a-fA-F]*$")


def normalize_hex_message(message: str) -> str:
    normalized = re.sub(r"\s+", "", message).lower()
    if not normalized:
        raise ValueError("Message cannot be empty")
    if len(normalized) % 2 != 0:
        raise ValueError(f"Hex message must contain an even number of characters: {message!r}")
    if _HEX_RE.fullmatch(normalized) is None:
        raise ValueError(f"Message contains non-hex characters: {message!r}")
    return normalized


@dataclass(slots=True)
class MessageBatch:
    messages: list[str]
    group_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = [normalize_hex_message(message) for message in self.messages]
        if not self.messages:
            raise ValueError("MessageBatch requires at least one message")

    @classmethod
    def from_iterable(
        cls,
        messages: Iterable[str],
        group_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MessageBatch":
        return cls(list(messages), group_label=group_label, metadata=metadata or {})

    @property
    def num_messages(self) -> int:
        return len(self.messages)

    @property
    def byte_lengths(self) -> list[int]:
        return [len(message) // 2 for message in self.messages]

    @property
    def min_length(self) -> int:
        return min(self.byte_lengths)

    @property
    def max_length(self) -> int:
        return max(self.byte_lengths)

    def to_bytes_list(self) -> list[bytes]:
        return [bytes.fromhex(message) for message in self.messages]

    def slice_by_spans(
        self,
        spans: list[Span],
        group_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MessageBatch":
        span_map = {span.message_index: span for span in spans}
        sliced_messages: list[str] = []

        for message_index, message in enumerate(self.messages):
            span = span_map.get(message_index)
            if span is None:
                raise ValueError(f"Missing span for message index {message_index}")
            if span.start < 0 or span.stop * 2 > len(message) or span.stop < span.start:
                raise ValueError(f"Invalid span for message index {message_index}: {span}")
            sliced_messages.append(message[span.start * 2 : span.stop * 2])

        combined_metadata = dict(self.metadata)
        if metadata:
            combined_metadata.update(metadata)
        return MessageBatch(
            sliced_messages,
            group_label=group_label or self.group_label,
            metadata=combined_metadata,
        )
