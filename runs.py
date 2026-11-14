"""Helpers for finding trained runs produced by train.py."""

import json
from pathlib import Path


def find_runs(results_dir, archs=None):
    """Map arch -> run directory for every run in `results_dir` that has a checkpoint.

    If an architecture was trained more than once, the most recent run (by folder
    name, which ends with a timestamp) is used.
    """
    runs = {}
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not (run_dir / "best_model.pth").exists() or not (run_dir / "args.json").exists():
            continue
        arch = json.load(open(run_dir / "args.json"))["arch"]
        if archs is None or arch in archs:
            runs[arch] = run_dir  # sorted order → the latest run wins
    return runs
