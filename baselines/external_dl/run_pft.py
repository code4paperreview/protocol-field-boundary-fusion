"""Source-only PFT classifier training and full nine-protocol evaluation."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .common import evaluate_predictions, load_messages, resolve_csv, save_predictions, write_json
from .pft import AUTHOR_ROOT, CLASSES, EXCLUDED_SOURCE_PROTOCOLS, FEATURES, predict


PROTOCOLS = ["dnp3", "modbus", "s7comm", "smb", "hart_ip", "iec104", "mqtt", "omron_fins", "tls"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_frame():
    path = AUTHOR_ROOT / "src/PacketFeatureTree/Training/FULL_DF.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.Class.isin(CLASSES) & ~frame.Protocol.isin(EXCLUDED_SOURCE_PROTOCOLS)]
    frame = frame.drop_duplicates(subset=FEATURES + ["Class", "Protocol"])
    return path, frame


def train_classifier(output):
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Source training directory is not empty: {output}")
    path, frame = source_frame()
    operations_path = AUTHOR_ROOT / "src/PacketFeatureTree/Training/df_operations.py"
    spec = importlib.util.spec_from_file_location("_pft_source_splits", operations_path)
    operations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operations)
    train_protocols, validation_protocols = operations.distribute_items(
        frame[["Class", "Protocol"]].value_counts().to_frame("counts").reset_index(level=[0, 1])
    )
    train = frame[frame.Protocol.isin(train_protocols)]
    validation = frame[frame.Protocol.isin(validation_protocols)]
    calibration, development = train_test_split(
        validation, test_size=0.5, random_state=42, stratify=validation.Class,
    )
    for subset in (train, calibration, development):
        if sorted(subset.Class.unique()) != CLASSES:
            raise ValueError("Each source split must cover all seven author syntax classes")
    manifest = {
        "status": "started", "source_path": str(path.resolve()), "source_sha256": sha256(path),
        "feature_columns": FEATURES, "classes": CLASSES,
        "excluded_target_protocol_families": EXCLUDED_SOURCE_PROTOCOLS,
        "retained_protocols": sorted(frame.Protocol.unique()),
        "split": "author distribute_items by source protocol; validation split 50/50 stratified by source syntax with random_state=42",
        "train_protocols": train_protocols, "validation_protocols": validation_protocols,
        "train_feature_rows": train.index.tolist(), "calibration_feature_rows": calibration.index.tolist(),
        "development_feature_rows": development.index.tolist(),
        "target_labels_used": False,
        "adaptation": "one shared classifier; all nine target protocols and SMB2 excluded together",
    }
    write_json(output / "manifest.json", manifest)
    candidates = [
        ("logistic_regression", LogisticRegressionCV(random_state=1337, class_weight="balanced", solver="newton-cg")),
        ("gradient_boosting", GradientBoostingClassifier(loss="log_loss", max_depth=8, random_state=1337)),
        ("random_forest", RandomForestClassifier(criterion="log_loss", random_state=1337, class_weight="balanced")),
        ("sgd_logistic", SGDClassifier(loss="log_loss", random_state=1337, class_weight="balanced")),
    ]
    results = []
    for name, base in candidates:
        started = time.perf_counter()
        print({"stage": "source_training", "candidate": name, "rows": len(train)}, flush=True)
        base.fit(train[FEATURES], train.Class)
        calibrated = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        calibrated.fit(calibration[FEATURES], calibration.Class)
        loss = log_loss(development.Class, calibrated.predict_proba(development[FEATURES]), labels=CLASSES)
        model_path = output / (name + ".pkl")
        with model_path.open("wb") as stream:
            pickle.dump(calibrated, stream)
        results.append({"candidate": name, "development_log_loss": float(loss),
                        "seconds": time.perf_counter() - started, "model": model_path.name,
                        "sha256": sha256(model_path), "parameters": base.get_params()})
        write_json(output / "candidate_results.json", results)
        print({"candidate": name, "development_log_loss": loss, "seconds": results[-1]["seconds"]}, flush=True)
    selected = min(results, key=lambda row: row["development_log_loss"])
    manifest.update(status="completed", selected=selected,
                    selection="minimum source development cross-entropy; no target results")
    write_json(output / "manifest.json", manifest)
    return output / selected["model"]


def summarize(output, protocols):
    rows = []
    for protocol in protocols:
        folder = output / protocol
        if not (folder / "metrics.json").exists():
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed":
            continue
        rows.append({"method": "pft_2024_author", "protocol": protocol, "status": "completed",
                     "messages": metrics["message_count"], **{k: metrics["average_metrics"][k]
                     for k in ["precision", "recall", "f1", "perfection"]}})
    if rows:
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if len(rows) == len(protocols):
        macro = {"method": "pft_2024_author", "protocols": len(rows), "messages": sum(r["messages"] for r in rows),
                 **{k: sum(r[k] for r in rows) / len(rows) for k in ["precision", "recall", "f1", "perfection"]}}
        with (output / "macro_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(macro))
            writer.writeheader()
            writer.writerow(macro)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--model", help="Reuse the source-only classifier fitted by an earlier PFT run")
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.target_limit < 0 or args.threads < 1 or len(set(args.protocols)) != len(args.protocols):
        parser.error("Invalid row limit, thread count or duplicate protocols")
    output = Path(args.output).resolve()
    implementation = {name: sha256(Path(__file__).with_name(name))
                      for name in ["pft.py", "pft_numerics.py", "run_pft.py", "common.py"]}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AUTHOR_ROOT, text=True).strip()
    configuration = {"protocols": args.protocols, "target_limit": args.target_limit,
                     "author_commit": commit, "implementation_sha256": implementation,
                     "numeric_threshold": 8, "source_features_sha256": sha256(AUTHOR_ROOT / "src/PacketFeatureTree/Training/FULL_DF.csv")}
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            parser.error(f"Output directory is not empty: {output}")
        previous = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
        if previous["configuration"] != configuration:
            parser.error("Resume configuration, source, author commit or adapter hashes differ")
    else:
        output.mkdir(parents=True, exist_ok=True)
    batch = {"status": "started", "pid": os.getpid(), "configuration": configuration,
             "python": platform.python_version(), "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
             "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
             "device": "CPU; scikit-learn and Python tree implementation", "target_labels_used_in_inference": False,
             "run_type": "smoke" if args.target_limit else "experiment"}
    write_json(output / "batch_manifest.json", batch)
    with threadpool_limits(limits=args.threads):
        if args.model:
            model = Path(args.model).resolve()
        elif (output / "training/manifest.json").exists():
            training = json.loads((output / "training/manifest.json").read_text(encoding="utf-8"))
            if training["status"] != "completed":
                raise ValueError("Incomplete source training is preserved; inspect before continuing")
            model = output / "training" / training["selected"]["model"]
            if sha256(model) != training["selected"]["sha256"]:
                raise ValueError("Saved source classifier hash differs")
        else:
            model = train_classifier(output / "training")
        batch.update(model=str(model), model_sha256=sha256(model))
        write_json(output / "batch_manifest.json", batch)
        for protocol in args.protocols:
            folder = output / protocol
            path = resolve_csv(protocol)
            identity = {"protocol": protocol, "target_sha256": sha256(path), "model_sha256": sha256(model),
                        "target_limit": args.target_limit, "implementation_sha256": implementation}
            if (folder / "manifest.json").exists():
                saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                if saved["identity"] != identity or saved["status"] != "completed":
                    raise ValueError(f"Existing incomplete or incompatible protocol run: {folder}")
                print({"reused": protocol}, flush=True)
                continue
            folder.mkdir()
            manifest = {"status": "started", "identity": identity, "target_path": str(path.resolve()),
                        "model": str(model), "author_commit": commit, "target_labels_used_in_inference": False}
            write_json(folder / "manifest.json", manifest)
            messages = load_messages(path)
            if args.target_limit:
                messages = messages[:args.target_limit]
            started = time.perf_counter()
            print({"protocol": protocol, "stage": "inference", "rows": len(messages)}, flush=True)
            predictions, diagnostics = predict(messages, model)
            save_predictions(folder, messages, predictions)
            # Target boundary labels are first accessed after predictions are persisted.
            evaluate_predictions(folder, path, messages, predictions)
            manifest.update(status="completed", settings=diagnostics, seconds=time.perf_counter() - started,
                            predicted_rows=len(predictions), predictions_sha256=sha256(folder / "predictions.jsonl"))
            write_json(folder / "manifest.json", manifest)
            summarize(output, args.protocols)
            print({"completed": protocol, "seconds": manifest["seconds"], "diagnostics": diagnostics}, flush=True)
        summarize(output, args.protocols)
    batch["status"] = "completed"
    write_json(output / "batch_manifest.json", batch)


if __name__ == "__main__":
    main()
