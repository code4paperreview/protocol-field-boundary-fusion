"""Run ProsegDL, SLMSP, or TransRE on one protocol."""

import argparse
import hashlib
import platform
from pathlib import Path

import torch

from .common import (
    evaluate_predictions, load_messages, load_sources, resolve_csv, save_predictions,
    seed_everything, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["prosegdl", "slmsp", "transre"], required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True, help="New/empty run directory; existing runs are preserved")
    parser.add_argument("--source-protocols", nargs="+", help="Default: available protocols excluding target family")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--glove-epochs", type=int)
    parser.add_argument("--siamese-epochs", type=int)
    parser.add_argument("--purge-threshold", type=float, default=.2, help="ProsegDL Siamese boundary-confidence threshold (default: 0.2)")
    parser.add_argument("--selection-steps", type=int)
    parser.add_argument("--target-limit", type=int, default=0, help="0 keeps every target row")
    parser.add_argument("--source-limit", type=int, default=0, help="0 keeps every source row")
    parser.add_argument("--smoke", action="store_true", help="Small integration run, explicitly marked as nonfinal")
    args = parser.parse_args()
    if not 0 <= args.purge_threshold <= 1:
        parser.error("Purge threshold must be between zero and one")
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke else {"prosegdl": 80, "slmsp": 30, "transre": 100}[args.method])
    args.batch_size = args.batch_size if args.batch_size is not None else (1 if args.smoke and args.method == "prosegdl" else 16)
    args.glove_epochs = args.glove_epochs if args.glove_epochs is not None else (1 if args.smoke else 300)
    args.siamese_epochs = args.siamese_epochs if args.siamese_epochs is not None else (1 if args.smoke else 25)
    args.selection_steps = args.selection_steps if args.selection_steps is not None else (2 if args.smoke else 100)
    if min(args.epochs, args.batch_size, args.glove_epochs, args.siamese_epochs, args.selection_steps) < 1 or min(args.source_limit, args.target_limit) < 0:
        parser.error("Epochs, batch size and selection steps must be positive; row limits must be nonnegative")
    if args.smoke:
        args.source_limit = args.source_limit or 64
        args.target_limit = args.target_limit or 32
        if args.source_protocols is None:
            args.source_protocols = ["dns" if args.protocol != "dns" else "modbus"]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    target_path = resolve_csv(args.protocol)
    all_messages = load_messages(target_path)
    messages = all_messages[:args.target_limit] if args.target_limit else all_messages
    if not messages or any(not m.payload for m in messages):
        parser.error("Target input must contain nonempty messages")
    device = torch.device(args.device)
    manifest = {
        "status": "started", "run_type": "smoke" if args.smoke else "experiment",
        "arguments": vars(args), "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "target_path": str(target_path), "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "target_row_indices": [m.row_index for m in messages],
        "target_labels_used_in_training": False,
        "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    write_json(output / "manifest.json", manifest)
    print({"method": args.method, "protocol": args.protocol, "rows": len(messages), "run_type": manifest["run_type"]}, flush=True)
    if args.method == "slmsp":
        from .run_slmsp import run
        predictions, settings = run(messages, args, output, device)
    else:
        train, valid, source_manifest = load_sources(args.protocol, all_messages, args.source_protocols, args.source_limit, args.seed)
        manifest["source_data"] = source_manifest
        write_json(output / "manifest.json", manifest)
        if args.method == "prosegdl":
            from .run_prosegdl import run
        else:
            from .run_transre import run
        predictions, settings = run(messages, train, valid, args, output, device)
    save_predictions(output, messages, predictions)
    evaluate_predictions(output, target_path, messages, predictions)
    manifest.update(status="completed", settings=settings)
    write_json(output / "manifest.json", manifest)
    print({"completed": str(output), "predicted_rows": len(predictions), "run_type": manifest["run_type"]}, flush=True)


if __name__ == "__main__":
    main()
