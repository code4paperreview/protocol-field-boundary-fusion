"""Queue NetPlier clustering and ProsegDL inference."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("NHF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
NETPLIER_PYTHON = os.environ.get("NETPLIER_PYTHON", "python3")
MODULE = "baselines.external_dl."
PROTOCOLS = ["dnp3", "modbus", "s7comm", "smb", "hart_ip", "iec104", "mqtt", "omron_fins", "tls"]


def linux_path(path):
    path = path.resolve().as_posix()
    if len(path) < 3 or path[1:3] != ":/":
        raise ValueError("Run this queue from Windows with drive-based paths")
    return "/mnt/" + path[0].lower() + path[2:]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = output / "queue.json"
    if state_path.exists():
        parser.error(f"Queue state already exists; inspect it before restarting: {state_path}")
    states = {protocol: {"clustering": "pending", "inference": "waiting_for_clustering"} for protocol in PROTOCOLS}
    state = {"status": "running", "pid": os.getpid(), "protocols": states,
             "training_root": str(args.training_root.resolve()), "inference_device": "cpu", "inference_threads": 2}
    cluster_child = None
    infer_child = None
    streams = []
    def launch(command, log):
        stream = log.open("w", encoding="utf-8")
        streams.append(stream)
        print({"launch": command, "log": str(log)}, flush=True)
        return subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    while True:
        if cluster_child is not None and cluster_child[1].poll() is not None:
            protocol, child = cluster_child
            if child.returncode:
                states[protocol].update(clustering="failed", cluster_exit_code=child.returncode)
            cluster_child = None
        if infer_child is not None and infer_child[1].poll() is not None:
            protocol, child = infer_child
            states[protocol]["inference"] = "completed" if child.returncode == 0 else "failed"
            states[protocol]["inference_exit_code"] = child.returncode
            infer_child = None
        active_cluster = cluster_child is not None
        for protocol, item in states.items():
            metadata = output / "metadata" / (protocol + ".json")
            directory = output / "clustering" / protocol
            manifest_path = directory / "manifest.json"
            if not metadata.exists():
                item.update(clustering="blocked_missing_metadata", inference="blocked_missing_metadata")
                continue
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if manifest["status"] == "completed":
                    item["clustering"] = "completed"
                elif manifest["status"] == "failed":
                    item["clustering"] = "failed"
                elif cluster_child is not None and cluster_child[0] == protocol:
                    item["clustering"] = "running"
                    active_cluster = True
                elif item["clustering"] != "failed":
                    # Attach only to an explicitly recorded live upstream job.
                    pid = manifest.get("pid")
                    check = subprocess.run(["wsl", "-d", "ubuntu", "--", "kill", "-0", str(pid)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if pid else None
                    if check is not None and check.returncode == 0:
                        item["clustering"] = "running_attached"
                        active_cluster = True
                    else:
                        item.update(clustering="failed", error="Incomplete clustering run has no live recorded worker")
            elif item["clustering"] not in {"failed", "running"} and not active_cluster:
                command = ["wsl", "-d", "ubuntu", "--", "env",
                           "PYTHONPATH=" + linux_path(ROOT / "netzob/src") + ":" + linux_path(ROOT),
                           "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
                           NETPLIER_PYTHON, "-u", "-m", MODULE + "cluster_netplier",
                           "--metadata", linux_path(metadata), "--output", linux_path(directory)]
                cluster_child = (protocol, launch(command, logs / ("cluster_" + protocol + ".log")))
                item["clustering"] = "running"
                active_cluster = True
            if item["clustering"] == "failed":
                item["inference"] = "blocked_clustering_failed"
                continue
            if item["clustering"] != "completed" or item["inference"] in {"completed", "failed", "running"}:
                continue
            training = args.training_root / f"prosegdl_{protocol}_seed20261021"
            training_path = training / "manifest.json"
            if not training_path.exists() or read_json(training_path)["status"] != "completed":
                item["inference"] = "waiting_for_training"
                batch_path = args.training_root / "batch_manifest.json"
                if batch_path.exists():
                    runs = read_json(batch_path)["runs"]
                    if any(r["method"] == "prosegdl" and r["protocol"] == protocol and r["status"] == "failed" for r in runs):
                        item["inference"] = "blocked_training_failed"
                continue
            if infer_child is None:
                command = [sys.executable, "-u", "-m", MODULE + "infer_prosegdl_netplier", "--protocol", protocol,
                           "--training-run", str(training), "--clusters", str(directory / "clusters.json"),
                           "--output", str(output / "inference" / protocol), "--device", "cpu", "--threads", "2"]
                infer_child = (protocol, launch(command, logs / ("infer_" + protocol + ".log")))
                item["inference"] = "running"
        active = any(item["clustering"] in {"pending", "running", "running_attached"}
                     or item["inference"] in {"waiting_for_training", "waiting_for_clustering", "running"}
                     for item in states.values())
        state["status"] = "running" if active else "completed" if all(item["inference"] == "completed" for item in states.values()) else "finished_with_incomplete_protocols"
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if not active:
            break
        time.sleep(30)
    for stream in streams:
        stream.close()


if __name__ == "__main__":
    main()
