"""
Evaluate trained models on any split of a dataset.

train.py already evaluates every model on the test split when training ends. Use this
script to evaluate trained runs again, on another split (e.g. the Pompeian Styles
`testset_square` control set), or on a different dataset with the same classes.

Choose the models with either
  --checkpoint RUN_DIR [RUN_DIR ...]   specific runs (folders made by train.py, or best_model.pth files)
  --results_dir DIR [--archs ...]      the latest run of every architecture in DIR

Writes <output_dir>/<arch>/{metrics.json, predictions.npz} and <output_dir>/summary.json.

Examples:
    # Test split of Pompeian Styles, all trained models
    python evaluate.py --dataset pompeian --data_dir data/pompeian-styles \
        --results_dir results/pompeian --split test --output_dir results/eval_pompeian_test

    # Square-tile control set
    python evaluate.py --dataset pompeian --data_dir data/pompeian-styles \
        --results_dir results/pompeian --split testset_square --output_dir results/eval_testset_square

    # One CLEOPATRA model
    python evaluate.py --dataset cleopatra --data_dir /path/to/CLEOPATRA \
        --checkpoint results/cleopatra/dinov3_large_20260214_172217 --output_dir results/eval_cleopatra
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import build_eval_dataset
from models import get_input_size, get_model
from runs import find_runs
from train import evaluate

DEFAULT_SPLIT = {"pompeian": "test", "cleopatra": "test", "folder": "test"}


def load_run(path):
    """Return (arch, run_name, checkpoint) for a run folder or a best_model.pth file."""
    path = Path(path)
    ckpt_path = path / "best_model.pth" if path.is_dir() else path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_file = ckpt_path.parent / "args.json"
    arch = json.load(open(args_file))["arch"] if args_file.exists() else ckpt["args"]["arch"]
    return arch, ckpt_path.parent.name, ckpt


def evaluate_run(arch, ckpt, dataset, data_dir, split, device, batch_size, num_workers):
    img_size = get_input_size(arch)
    ds, class_names = build_eval_dataset(dataset, data_dir, split, img_size, ckpt.get("class_names"))
    model, _ = get_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metrics, preds, labels, probs = evaluate(model, loader, nn.CrossEntropyLoss(), device, class_names)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, preds, labels, probs, class_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="pompeian", choices=["pompeian", "cleopatra", "folder"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default=None, help="Split to evaluate (default: test)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", nargs="+", help="Run folder(s) or best_model.pth file(s)")
    group.add_argument("--results_dir", help="Evaluate the latest run of every architecture here")
    ap.add_argument("--archs", nargs="*", default=None, help="With --results_dir: only these architectures")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    split = args.split or DEFAULT_SPLIT[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_paths = args.checkpoint or list(find_runs(args.results_dir, args.archs).values())
    if not run_paths:
        raise SystemExit("No trained runs found.")

    summary, used_names = [], set()
    for path in run_paths:
        arch, run_name, ckpt = load_run(path)
        print(f"\n=== {arch} ({run_name}) on {args.dataset}/{split} ===")
        metrics, preds, labels, probs, class_names = evaluate_run(
            arch, ckpt, args.dataset, args.data_dir, split, device, args.batch_size, args.num_workers)
        print(f"  Acc={metrics['accuracy']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f}  "
              f"F1(macro)={metrics['f1_macro']:.4f}")

        name = arch if arch not in used_names else run_name  # two runs of one arch → use run names
        used_names.add(name)
        (out_dir / name).mkdir(exist_ok=True)
        result = {"architecture": arch, "run": run_name, "dataset": args.dataset, "split": split,
                  "n_samples": int(len(labels)), "class_names": class_names, **metrics}
        with open(out_dir / name / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        np.savez(out_dir / name / "predictions.npz", predictions=preds, labels=labels, probabilities=probs)
        summary.append({k: v for k, v in result.items() if k != "classification_report"})

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
