"""Recover label-free NetPlier metadata in WSL; retain every CSV row identity."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(os.environ.get("NHF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PCAPS = {
    "dnp3": "format_label/data/pcap/dnp3/dnp3.pcap",
    "modbus": "format_label/data/pcap/modbus/ModbusTCP_1000.pcap",
    "s7comm": "format_label/data/pcap/s7comm/s7comm.pcap",
}
SUPPLEMENTAL = {"hart_ip", "iec104", "mqtt", "omron_fins"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_payloads(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [bytes.fromhex(row["Hex"]) for row in csv.DictReader(stream)]


def payload_identity(payloads):
    # Length prefixes distinguish e.g. [ab, cd] from [abcd]. Duplicates count.
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def capture_records(path):
    from scapy.all import IP, IPv6, PcapReader, TCP, UDP

    with PcapReader(str(path)) as capture:
        for index, packet in enumerate(capture, 1):
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            if network is None or transport is None or not bytes(transport.payload):
                continue
            kind = "tcp" if packet.haslayer(TCP) else "udp"
            source = f"{kind}://{network.src}:{transport.sport}"
            destination = f"{kind}://{network.dst}:{transport.dport}"
            yield bytes(transport.payload), {
                "packet_index": index, "timestamp": float(packet.time),
                "source": source, "destination": destination,
                "session_key": "|".join(sorted([source, destination])),
            }


def tls_records(data):
    """Yield complete TLS records and their offsets within a TCP payload."""
    offset = 0
    while offset + 5 <= len(data) and data[offset] in (20, 21, 22, 23, 24) and data[offset + 1] == 3:
        end = offset + 5 + int.from_bytes(data[offset + 3:offset + 5], "big")
        if end > len(data):
            break
        yield offset, data[offset:end]
        offset = end


def match_tls_capture(payloads, capture):
    wanted = set(payloads)
    occurrences = defaultdict(deque)
    for data, record in capture_records(capture):
        for offset, payload in tls_records(data):
            if payload in wanted:
                occurrences[payload].append(dict(record, payload_offset=offset))
    records = []
    for index, payload in enumerate(payloads):
        if not occurrences[payload]:
            raise ValueError(f"No unused exact TLS record occurrence for CSV row {index}")
        records.append(occurrences[payload].popleft())
    # Match upstream's generic direction convention on the selected messages:
    # the earliest observed source in each endpoint session is direction zero.
    first_sources = {}
    for record in sorted(records, key=lambda row: (row["timestamp"], row["packet_index"], row["payload_offset"])):
        first_sources.setdefault(record["session_key"], record["source"])
    for record in records:
        record["direction"] = int(record["source"] != first_sources[record["session_key"]])
    return records


def prepare(protocol, capture=None):
    folder = ROOT / "format_label/format_data" / protocol
    path = next((folder / f"{protocol}{suffix}.csv" for suffix in ["", "_realistic", "_balanced"]
                 if (folder / f"{protocol}{suffix}.csv").exists()), None)
    if path is None:
        raise FileNotFoundError(protocol)
    payloads = read_payloads(path)
    provenance = {}
    if protocol in SUPPLEMENTAL:
        manifest = path.with_name(path.stem + "_manifest.jsonl")
        with manifest.open(encoding="utf-8") as stream:
            # Deliberately select only payload identity and transport metadata.
            rows = [{key: row[key] for key in ["hex", "packet_index", "direction", "source", "session_key", "timestamp"]}
                    for row in map(json.loads, stream)]
        if [bytes.fromhex(row["hex"]) for row in rows] != payloads:
            raise ValueError("Manifest/CSV payload order mismatch")
        names = {row["source"] for row in rows}
        if len(names) != 1:
            raise ValueError("Expected one documented synthetic capture")
        pcap = ROOT / "csv/synthetic_pilot/_pcap_sources" / protocol / names.pop()
        packets = {} if protocol == "iec104" else {
            record["packet_index"]: (data, record) for data, record in capture_records(pcap)}
        records = []
        for payload, row in zip(payloads, rows):
            direction = {"client_to_server": 0, "server_to_client": 1}[row["direction"]]
            if protocol == "iec104":
                # This local capture was regenerated with TCP handshakes and
                # changed payloads. The exact CSV-matching manifest retains the
                # original transport metadata; do not substitute that capture.
                endpoints = row["session_key"].split("::", 1)[1].split("|")
                server = [endpoint for endpoint in endpoints if endpoint.endswith(":2404")]
                client = [endpoint for endpoint in endpoints if not endpoint.endswith(":2404")]
                if len(server) != 1 or len(client) != 1:
                    raise ValueError("Ambiguous IEC104 manifest server endpoint")
                source, destination = (client[0], server[0]) if direction == 0 else (server[0], client[0])
                record = {"packet_index": row["packet_index"], "timestamp": row["timestamp"],
                          "source": "tcp://" + source, "destination": "tcp://" + destination,
                          "session_key": row["session_key"]}
            else:
                data, record = packets[row["packet_index"]]
                if payload != data:
                    raise ValueError(f"Capture/manifest mismatch at packet {row['packet_index']}")
                record = dict(record)
            record["direction"] = direction
            record["payload_offset"] = 0
            records.append(record)
        provenance.update(manifest=str(manifest), manifest_sha256=sha256(manifest),
                          matching="exact manifest row identity; endpoints and timestamp from manifest" if protocol == "iec104"
                          else "exact manifest row and capture packet identity")
        if protocol != "iec104":
            provenance.update(capture=str(pcap), capture_sha256=sha256(pcap))
    elif protocol in PCAPS:
        pcap = ROOT / PCAPS[protocol]
        wanted = set(payloads)
        occurrences = defaultdict(deque)
        session_sources = {}
        for data, record in capture_records(pcap):
            session_sources.setdefault(record["session_key"], record["source"])
            for offset in [0, 4, 7]:
                if data[offset:] in wanted:
                    record = dict(record, payload_offset=offset)
                    payload = data[offset:]
                    if protocol == "dnp3":
                        record["direction"] = 0 if payload[3] & 0x80 else 1
                    elif protocol == "modbus":
                        if record["source"].endswith(":502"):
                            record["direction"] = 1
                        elif record["destination"].endswith(":502"):
                            record["direction"] = 0
                        else:
                            raise ValueError("Modbus direction unavailable")
                    else:
                        record["direction"] = int(record["source"] != session_sources[record["session_key"]])
                    occurrences[payload].append(record)
                    break
        records = []
        for index, payload in enumerate(payloads):
            if not occurrences[payload]:
                raise ValueError(f"No unused exact capture occurrence for CSV row {index}")
            records.append(occurrences[payload].popleft())
        provenance.update(capture=str(pcap), capture_sha256=sha256(pcap),
                          matching="exact payload, FIFO capture occurrence for repeated payloads",
                          duplicate_rows=len(payloads) - len(wanted),
                          matched_capture_order_preserved=all(a["packet_index"] < b["packet_index"] for a, b in zip(records, records[1:])))
    elif protocol == "tls" and capture is not None:
        records = match_tls_capture(payloads, capture)
        provenance.update(capture=str(capture), capture_sha256=sha256(capture),
                          matching="exact complete TLS record, FIFO capture occurrence for repeated records; no TCP stream reassembly",
                          duplicate_rows=len(payloads) - len(set(payloads)),
                          direction="upstream generic convention: earliest selected source per endpoint session is zero",
                          matched_capture_order_preserved=all(
                              (a["packet_index"], a["payload_offset"]) < (b["packet_index"], b["payload_offset"])
                              for a, b in zip(records, records[1:])))
    elif protocol == "smb":
        if any(len(data) < 32 or data[:4] != b"\xffSMB" for data in payloads):
            raise ValueError("Expected SMB1 payloads without NetBIOS wrapper")
        records = [{"direction": int(bool(data[9] & 0x80)), "source": None, "destination": None,
                    "session_key": None, "timestamp": None} for data in payloads]
        provenance["matching"] = "SMB1 response flag at byte 9; transport metadata unavailable"
    else:
        raise ValueError(f"Transport metadata unavailable for {protocol}; no endpoint/direction fabrication")
    for index, (record, payload) in enumerate(zip(records, payloads)):
        record.update(message_index=index, payload_sha256=hashlib.sha256(payload).hexdigest())
    return {
        "protocol": protocol, "target_path": str(path), "target_sha256": sha256(path),
        "payload_identity_sha256": payload_identity(payloads), "rows": records,
        "remote_coupling_available": protocol != "smb", "provenance": provenance,
        "direction_counts": dict(Counter(row["direction"] for row in records)),
        "target_labels_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--capture", type=Path, help="Original TLS capture containing the target records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Preserving existing metadata: {args.output}")
    if args.capture is not None and args.protocol != "tls":
        parser.error("--capture currently applies only to TLS")
    result = prepare(args.protocol, args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({"protocol": args.protocol, "rows": len(result["rows"]),
           "directions": result["direction_counts"], "remote_coupling": result["remote_coupling_available"]}, flush=True)


if __name__ == "__main__":
    main()
