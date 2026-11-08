"""Run TransRE once for every single-source / held-out-target pair."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import DATA_ROOT, load_messages, resolve_csv, write_json

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


PROTOCOLS = ["dnp3", "modbus", "s7comm", "smb", "hart_ip", "iec104", "mqtt", "omron_fins", "tls"]
SEED = 20261022
HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run(directory, source, target):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    args = manifest["arguments"]
    expected = {"method": "transre", "protocol": target, "source_protocols": [source],
                "seed": SEED, "epochs": 100, "batch_size": 16, "target_limit": 0,
                "source_limit": 0, "selection_steps": 100, "smoke": False}
    if manifest["status"] != "completed" or manifest["run_type"] != "experiment":
        raise ValueError("run is incomplete or marked as smoke")
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("run arguments do not match this matrix")
    if manifest.get("settings", {}).get("selected_source") != source:
        raise ValueError("selected source differs from requested source")
    if manifest.get("target_labels_used_in_training"):
        raise ValueError("target labels were recorded as used during training")
    for name in ["common.py", "run.py", "transre.py", "run_transre.py"]:
        if manifest["implementation_sha256"].get(name) != sha256(HERE / name):
            raise ValueError(f"implementation changed: {name}")
    sources = manifest.get("source_data", {}).get("sources", [])
    if len(sources) != 1:
        raise ValueError("expected exactly one source corpus")
    for item in sources:
        if item["protocol"] != source or item["sha256"] != sha256(item["path"]):
            raise ValueError("source corpus differs from requested source")
    target_path = resolve_csv(target)
    messages = load_messages(target_path)
    if len(messages) != 1000 or manifest["target_sha256"] != sha256(target_path):
        raise ValueError("target corpus is not the frozen 1,000-row input")
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if len(predictions) != len(messages) or metrics["message_count"] != len(messages):
        raise ValueError("prediction or metric row count differs from target corpus")
    if [p["message_index"] for p in predictions] != manifest["target_row_indices"] or manifest["target_row_indices"] != [m.row_index for m in messages]:
        raise ValueError("prediction row identities differ from target input")
    if not (directory / "checkpoint.pt").is_file():
        raise ValueError("checkpoint is missing")
    return metrics["average_metrics"]


def write_summary(output, runs):
    fields = ["source", "target", "status", "reused", "directory", "f1", "perfection", "precision", "recall", "error"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run.get(key, "") for key in fields}
            row.update({key: run.get("metrics", {}).get(key, "") for key in ["f1", "perfection", "precision", "recall"]})
            writer.writerow(row)
    best = []
    for target in PROTOCOLS:
        completed = [r for r in runs if r["target"] == target and r["status"] == "completed"]
        planned = sum(r["target"] == target for r in runs)
        if not completed:
            continue
        for criterion in ["f1", "perfection"]:
            ranked = sorted(completed, key=lambda r: (-r["metrics"][criterion], r["source"]))
            winner = ranked[0]
            best.append({"target": target, "criterion": criterion, "completed_sources": len(completed),
                         "planned_sources": planned, "source": winner["source"],
                         **{key: winner["metrics"][key] for key in ["f1", "perfection", "precision", "recall"]}})
    with (output / "best_sources.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["target", "criterion", "completed_sources", "planned_sources", "source", "f1", "perfection", "precision", "recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(best)
    for metric in ["f1", "perfection"]:
        sources = list(dict.fromkeys(r["source"] for r in runs))
        with (output / f"{metric}_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["target", *sources])
            writer.writeheader()
            for target in PROTOCOLS:
                writer.writerow({"target": target, **{r["source"]: r["metrics"][metric] for r in runs
                                 if r["target"] == target and r["status"] == "completed"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    available = [d.name for d in sorted(DATA_ROOT.iterdir()) if d.is_dir() and
                 any((d / f"{d.name}{suffix}.csv").exists() for suffix in ["", "_realistic", "_balanced"])]
    source_protocols = PROTOCOLS + [p for p in available if p not in PROTOCOLS]
    jobs = [{"source": source, "target": target} for target in PROTOCOLS
            for source in source_protocols if source != target
            and not (target == "smb" and source == "smb2")
            and not (target == "tls" and source == "ssl")]
    if args.plan_only:
        print(json.dumps({"protocols": PROTOCOLS, "source_protocols": source_protocols, "runs": len(jobs)}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    state_path = output / "batch_manifest.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("seed") != SEED or state.get("protocols") != PROTOCOLS:
            parser.error("existing batch manifest belongs to a different configuration")
        indexed = {(run["source"], run["target"]): run for run in state["runs"]}
        runs = [indexed.get((job["source"], job["target"]), {**job, "status": "pending", "reused": False}) for job in jobs]
    else:
        runs = [{**job, "status": "pending", "reused": False} for job in jobs]
        indexed = {(run["source"], run["target"]): run for run in runs}
    indexed = {(run["source"], run["target"]): run for run in runs}
    state = {"status": "running", "pid": os.getpid(), "seed": SEED, "protocols": PROTOCOLS,
             "source_protocols": source_protocols,
             "evaluation_sha256": sha256(PACKAGE_ROOT / "nhf/evaluation/metrics.py"),
             "selection": "best source per target on evaluation metrics",
             "started_utc": datetime.now(timezone.utc).isoformat(), "runs": runs}
    write_json(state_path, state)
    write_summary(output, runs)

    for source, target in [(job["source"], job["target"]) for job in jobs]:
        run = indexed[(source, target)]
        if run.get("status") == "completed":
            try:
                run["metrics"] = validate_run(Path(run["directory"]), source, target)
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                run.update(status="invalid_existing_run", error=str(error))
                continue
        previous = run.get("directory")
        candidates = []
        if previous:
            candidates.append(Path(previous))
        destination = output / f"transre_{source}_to_{target}_seed{SEED}"
        if destination.exists() and any(destination.iterdir()):
            candidates.append(destination)
        valid = False
        for candidate in candidates:
            try:
                metrics = validate_run(candidate, source, target)
                run.update(status="completed", reused=True, directory=str(candidate), metrics=metrics,
                           finished_utc=datetime.now(timezone.utc).isoformat())
                valid = True
                break
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        if valid:
            write_json(state_path, state)
            write_summary(output, runs)
            continue
        if destination.exists() and any(destination.iterdir()):
            run.update(status="incomplete_preserved", directory=str(destination),
                       error="nonempty destination did not validate; preserved without overwrite")
        else:
            run.update(status="running", directory=str(destination), started_utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
            write_summary(output, runs)
            log = output / "logs" / f"{source}_to_{target}_seed{SEED}.log"
            command = [sys.executable, "-u", "-m", "baselines.external_dl.run",
                       "--method", "transre", "--protocol", target, "--source-protocols", source,
                       "--seed", str(SEED), "--output", str(destination)]
            print(f"START {source} -> {target}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=PACKAGE_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                try:
                    metrics = validate_run(destination, source, target)
                    run.update(status="completed", reused=False, metrics=metrics,
                               finished_utc=datetime.now(timezone.utc).isoformat(), log=str(log))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    run.update(status="validation_failed", error=str(error), log=str(log))
            else:
                run.update(status="failed", returncode=result.returncode, log=str(log))
        write_json(state_path, state)
        write_summary(output, runs)
        done = sum(item.get("status") == "completed" for item in runs)
        print(f"PROGRESS {done}/{len(runs)}", flush=True)

    state["status"] = "completed" if all(run.get("status") == "completed" for run in runs) else "finished_with_incomplete_runs"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    write_summary(output, runs)
    print({"status": state["status"], "completed": sum(r.get("status") == "completed" for r in runs), "total": len(runs)}, flush=True)


if __name__ == "__main__":
    main()
