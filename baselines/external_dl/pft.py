"""Packet Field Tree adapter."""

import importlib.util
import os
import sys
from pathlib import Path

from .common import bit_boundaries_to_bytes


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_ROOT = Path(os.environ.get("PFT_ROOT", PACKAGE_ROOT / "third_party" / "PacketFieldTree"))
DEPENDENCIES = Path(os.environ.get("PFT_DEPENDENCIES", PACKAGE_ROOT / "third_party" / "pft_dependencies"))
FEATURES = [f"Bit {i} TP" for i in range(8)] + [f"Bit {i} AV" for i in range(8)]
CLASSES = ["ABSOLUTE_TIME", "ETHER", "IPv4", "STRING", "UINT16", "UINT32", "UINT8"]
# Hold out the whole nine-protocol target suite, including the SMB family.
EXCLUDED_SOURCE_PROTOCOLS = [
    "DNP3", "ModBus", "ModBusTCP", "S7Comm", "SMB", "SMB2", "MQTT",
    "HART-IP", "HART_IP", "IEC104", "Omron FINS", "OMRON-FINS", "TLS", "SSL",
]


def author_package():
    """Import the author's tree modules without unrelated database/training extras."""
    if str(DEPENDENCIES) not in sys.path:
        sys.path.insert(0, str(DEPENDENCIES))
    name = "_packet_field_tree_author"
    if name not in sys.modules:
        package = AUTHOR_ROOT / "src" / "PacketFeatureTree" / "PFT"
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prediction_from_lengths(message, syntaxes, lengths):
    """The author transform returns field widths in BITS, not boundary offsets."""
    offsets, offset = [0], 0
    for width in lengths:
        if int(width) != width or width <= 0:
            raise ValueError(f"Invalid author field width: {width}")
        offset += int(width)
        offsets.append(offset)
    if len(syntaxes) != len(lengths):
        raise ValueError("Author syntax and field-length counts differ")
    return {
        "message_index": message.row_index, "length": len(message.payload),
        "native_unit": "bit", "native_field_syntaxes": syntaxes,
        "native_field_bit_lengths": [int(width) for width in lengths],
        "native_boundaries": offsets,
        "byte_boundaries": bit_boundaries_to_bytes(offsets, len(message.payload)),
        "native_covered_bits": offset,
        "native_coverage_matches_message": offset == 8 * len(message.payload),
    }


def predict(messages, model_path):
    from .pft_numerics import label_tree

    package = author_package()
    # Ordinals identify rows for the author's cross-byte merge. They are not
    # reconstructed capture times; the supplied CSV order is retained.
    inputs = [(m.row_index, m.payload.hex()) for m in messages]
    depth = max(len(m.payload) for m in messages)
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 20 * depth + 2000))
    try:
        tree = package.OneBytePFT(numeric_thresh=8, depth=depth)
        print({"stage": "subfield_tree", "rows": len(messages), "depth": depth}, flush=True)
        tree.fit(inputs)
        node_count = len(tree.tree.descendants)
        print({"stage": "syntax_and_combination", "nodes": node_count}, flush=True)
        labelled = package.LabelledPFT(tree, str(model_path))
        numerical = label_tree(labelled.tree, labelled.sk_model, labelled.FIELD_LENGTHS, package)
        labelled.fit_optimal_pft()
        syntaxes, lengths = labelled.transform(inputs)
        if len(lengths) != len(messages):
            raise ValueError("Author transform omitted target rows")
        predictions = [prediction_from_lengths(m, s, w) for m, s, w in zip(messages, syntaxes, lengths)]
        diagnostics = {
            "numeric_threshold": 8, "depth": depth,
            "target_grouping": "one protocol-level author tree; no target labels",
            "row_order": "original CSV order; row ordinal used as unique identity",
            "subfield_nodes": node_count,
            "inferred_field_nodes": len(labelled.tree.descendants),
            **numerical,
            "native_coverage_mismatches": sum(not p["native_coverage_matches_message"] for p in predictions),
            "field_lengths": labelled.FIELD_LENGTHS,
        }
        return predictions, diagnostics
    finally:
        sys.setrecursionlimit(previous_limit)


def tree_nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(reversed(node.children))
