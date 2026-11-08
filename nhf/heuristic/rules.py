from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ContainerRule:
    q_width: int
    t_width: int
    l_width: int
    endian: str
    allow_suffix: bool = False

    @property
    def has_q(self) -> bool:
        return self.q_width > 0

    @property
    def header_width(self) -> int:
        return self.q_width + self.t_width + self.l_width

    def to_annotation(self) -> str:
        suffix = "*" if self.allow_suffix else ""
        if self.has_q:
            return f"{self.q_width}Q_{self.t_width}T_{self.l_width}L_V_{self.endian}{suffix}"
        return f"{self.t_width}T_{self.l_width}L_V_{self.endian}{suffix}"


_RULE_RE = re.compile(r"(?:(\d+)Q_)?(\d+)T_(\d+)L_V_(big|little)(\*)?")


def parse_container_rule(annotation: str) -> ContainerRule | None:
    match = _RULE_RE.match((annotation or "").strip())
    if match is None:
        return None
    t_raw, l_raw, e_raw = match.group(2), match.group(3), match.group(4)
    if t_raw is None or l_raw is None or e_raw is None:
        return None
    return ContainerRule(
        q_width=int(match.group(1) or 0),
        t_width=int(t_raw),
        l_width=int(l_raw),
        endian=e_raw,
        allow_suffix=bool(match.group(5)),
    )


_LEGACY_TLV_STAR_SPECS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "big"),
    (1, 1, "big"),
    (2, 2, "big"),
    (0, 1, "little"),
    (1, 1, "little"),
    (2, 2, "little"),
)


def iter_container_rules(
    q_widths: Iterable[int] = (0, 1, 2, 3),
    t_widths: Iterable[int] = (0, 1, 2, 3),
    l_widths: Iterable[int] = (1, 2, 3),
    endians: Iterable[str] = ("big", "little"),
) -> list[ContainerRule]:
    rules: list[ContainerRule] = []
    normalized_q_widths = {int(value) for value in q_widths}
    normalized_t_widths = [int(value) for value in t_widths if 0 <= int(value) <= 3]
    normalized_l_widths = [int(value) for value in l_widths if 1 <= int(value) <= 3]
    for t_width in normalized_t_widths:
        for l_width in normalized_l_widths:
            candidate_q_widths: list[int] = []
            if 0 in normalized_q_widths:
                candidate_q_widths.append(0)
            if l_width in normalized_q_widths:
                candidate_q_widths.append(l_width)
            for q_width in candidate_q_widths:
                for endian in endians:
                    rules.append(ContainerRule(q_width=q_width, t_width=t_width, l_width=l_width, endian=endian))
    return rules


def iter_tlv_star_rules(
    specs: Iterable[tuple[int, int, str]] = _LEGACY_TLV_STAR_SPECS,
) -> list[ContainerRule]:
    return [
        ContainerRule(q_width=0, t_width=t_width, l_width=l_width, endian=endian, allow_suffix=True)
        for t_width, l_width, endian in specs
    ]
