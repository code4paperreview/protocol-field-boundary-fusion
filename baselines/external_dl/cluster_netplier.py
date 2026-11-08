"""Run pinned upstream NetPlier in WSL using sanitized transport metadata."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .prepare_netplier import ROOT, payload_identity, read_payloads, sha256

NETPLIER_ROOT = Path(os.environ.get("NETPLIER_ROOT", ROOT / "third_party"))


def clustering_payload(data, protocol):
    # Match upstream Processing.import_messages before its global byte cap.
    # Modbus TCP payloads can contain multiple ADUs; upstream keeps the first.
    # The original full payload and row identity remain used for evaluation.
    if protocol == "modbus":
        length = int.from_bytes(data[4:6], byteorder="big", signed=True)
        if len(data) != length + 6:
            data = data[:length + 6]
    return data[:500]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Preserving existing run: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    target = Path(metadata["target_path"])
    payloads = read_payloads(target)
    if sha256(target) != metadata["target_sha256"] or payload_identity(payloads) != metadata["payload_identity_sha256"]:
        raise ValueError("Target changed since metadata extraction")
    rows = metadata["rows"]
    if len(rows) != len(payloads):
        raise ValueError("Metadata row count mismatch")
    for index, (row, data) in enumerate(zip(rows, payloads)):
        if row["message_index"] != index or row["payload_sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError("Metadata row identity mismatch")
    directions = [row["direction"] for row in rows]
    if set(directions) != {0, 1}:
        raise ValueError("Upstream paired inference requires both observed directions")
    sys.path.insert(0, str(NETPLIER_ROOT / "NetPlier/netplier"))
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from alignment import Alignment
    from clustering import Clustering
    from netplier import NetPlier

    # Upstream uses -1 for unavailable pairing evidence and removes those
    # factors in ProbabilisticInference.update_invalid_p. Do not invent flows.
    if not metadata["remote_coupling_available"]:
        from constraint import constraint

        class UnavailableRemoteCoupling:
            TEST_TYPE_REQUEST = 0
            TEST_TYPE_RESPONSE = 1

            def __init__(self, messages_all, symbols_request, symbols_response, direction_list):
                self.symbols = [symbols_request, symbols_response]

            def compute_pairs_by_directionlist(self):
                pass

            def compute_constraint_remote_coupling(self, direction):
                return [-1] * len(self.symbols[direction])

        constraint.RemoteCoupling = UnavailableRemoteCoupling
    clustering_payloads = [clustering_payload(data, metadata["protocol"]) for data in payloads]
    messages = [RawMessage(data=data, date=row["timestamp"] if row["timestamp"] is not None else float(index),
                           source=row["source"], destination=row["destination"])
                for index, (row, data) in enumerate(zip(rows, clustering_payloads))]
    mode = "linsi" if metadata["protocol"] == "dnp3" else "ginsi"
    manifest = {
        "status": "started", "pid": os.getpid(), "protocol": metadata["protocol"],
        "upstream": json.loads((NETPLIER_ROOT / "netplier_upstream.json").read_text(encoding="utf-8")),
        "metadata_sha256": sha256(args.metadata), "target_sha256": metadata["target_sha256"],
        "payload_identity_sha256": metadata["payload_identity_sha256"],
        "mafft_mode": mode, "mafft_multithread": False, "clustering_byte_cap": 500,
        "clustering_preprocessing": "upstream first Modbus/TCP ADU, then 500-byte cap" if metadata["protocol"] == "modbus" else "500-byte cap",
        "clustering_payload_identity_sha256": payload_identity(clustering_payloads),
        "preprocessed_rows": [i for i, (original, prepared) in enumerate(zip(payloads, clustering_payloads)) if original != prepared],
        "remote_coupling_available": metadata["remote_coupling_available"],
        "target_labels_used": False,
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pgmpy", "networkx", "scikit-learn"]},
        "adapter_sha256": sha256(Path(__file__)),
    }
    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    started = time.monotonic()
    try:
        model = NetPlier(messages, directions, str(args.output / "upstream"), mode=mode, multithread=False)
        fields = model.execute()
        aligned = Alignment.get_messages_aligned(messages, os.path.join(model.output_dir, Alignment.FILENAME_OUTPUT_ONELINE))
        keywords = Clustering(model.fields, protocol_type=None).cluster_by_kw_inferred(fields, aligned)
        if len(keywords) != len(payloads):
            raise ValueError("Upstream alignment changed row count")
        keys = [(direction, keyword) for direction, keyword in zip(directions, keywords)]
        key_ids = {key: index for index, key in enumerate(sorted(set(keys)))}
        assignments = [{"message_index": index, "cluster_id": key_ids[key],
                        "payload_sha256": row["payload_sha256"]}
                       for index, (key, row) in enumerate(zip(keys, rows))]
        result = {"protocol": metadata["protocol"], "payload_identity_sha256": metadata["payload_identity_sha256"],
                  "target_sha256": metadata["target_sha256"], "inferred_fields": fields,
                  "remote_coupling_available": metadata["remote_coupling_available"],
                  "assignments": assignments,
                  "cluster_sizes": dict(Counter(item["cluster_id"] for item in assignments))}
        (args.output / "clusters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        cluster_sizes=result["cluster_sizes"], inferred_fields=fields)
        print({"protocol": metadata["protocol"], "cluster_sizes": result["cluster_sizes"]}, flush=True)
    except Exception as error:
        manifest.update(status="failed", elapsed_seconds=time.monotonic() - started, error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
