"""
Collect test results of all runs in a results folder into one ranking table.

Reads <results_dir>/*/test_results.json (deep models, latest run per architecture) and
<results_dir>/*/all_results.json (Random Forest / Gradient Boosting baselines), prints a
table ranked by balanced accuracy, and writes it as Markdown and JSON.

Example:
    python collect_results.py --results_dir results/pompeian
    python collect_results.py --results_dir paper_results/pompeian   # the paper's numbers
"""

import argparse
import json
from pathlib import Path

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
HEADERS = ["Acc.", "Bal. Acc.", "F1 macro", "F1 wt.", "Prec.", "Rec."]


def load_rows(results_dir):
    deep, classical = {}, []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "test_results.json").exists():
            r = json.load(open(run_dir / "test_results.json"))
            params = r.get("total_params")
            deep[r["architecture"]] = {  # sorted order → the latest run wins
                "model": r["architecture"],
                "params_M": round(params / 1e6, 1) if isinstance(params, (int, float)) else None,
                **{m: r["test_metrics"][m] for m in METRICS},
                "per_class_f1": r["test_metrics"].get("per_class_f1"),
                "run": run_dir.name,
            }
        if (run_dir / "all_results.json").exists():
            for name, r in json.load(open(run_dir / "all_results.json")).items():
                classical.append({"model": name.replace(" ", ""), "params_M": None,
                                  **{m: r["test_metrics"][m] for m in METRICS},
                                  "per_class_f1": r["test_metrics"].get("per_class_f1"),
                                  "run": run_dir.name})
    rank = lambda rows: sorted(rows, key=lambda x: x["balanced_accuracy"], reverse=True)
    return rank(deep.values()), rank(classical)


def to_markdown(rows, title):
    lines = [f"### {title}", "", "| Model | Params (M) | " + " | ".join(HEADERS) + " |",
             "|---|---|" + "---|" * len(HEADERS)]
    for r in rows:
        params = "—" if r["params_M"] is None else f"{r['params_M']}"
        lines.append(f"| {r['model']} | {params} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", default=None, help="Default: <results_dir>")
    args = ap.parse_args()
    out_dir = Path(args.output_dir or args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deep, classical = load_rows(args.results_dir)
    if not deep and not classical:
        print(f"No results found in {args.results_dir}")
        return
    md = to_markdown(deep, "Deep models (ranked by balanced accuracy)")
    if classical:
        md += "\n\n" + to_markdown(classical, "Traditional ML baselines")
    print(md)
    (out_dir / "results_table.md").write_text(md + "\n")
    with open(out_dir / "results_table.json", "w") as f:
        json.dump({"deep_models": deep, "traditional_baselines": classical}, f, indent=2)
    print(f"\nSaved {out_dir / 'results_table.md'} and {out_dir / 'results_table.json'}")


if __name__ == "__main__":
    main()
