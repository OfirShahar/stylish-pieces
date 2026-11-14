"""
Train and evaluate a fragment style classifier.

Trains with early stopping on validation balanced accuracy, then evaluates the best
checkpoint on the test split. Everything is saved to <output_dir>/<arch>_<timestamp>/:
args.json, best_model.pth, training_history.json, test_results.json, test_predictions.npz.

Examples:
    python train.py --dataset pompeian  --data_dir /path/to/pompeian-styles --arch convnext_small --lr 5e-5
    python train.py --dataset cleopatra --data_dir /path/to/CLEOPATRA       --arch dinov3_large --batch_size 16 --lr 5e-5
    python train.py --dataset folder    --data_dir /path/to/my_dataset      --arch resnet50
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from data import build_dataloaders
from models import count_parameters, get_input_size, get_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a fragment style classifier")
    p.add_argument("--arch", type=str, required=True, help="Architecture name (see models.py)")
    p.add_argument("--dataset", type=str, default="pompeian", choices=["pompeian", "cleopatra", "folder"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where runs are saved (default: results/<dataset>)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_epochs", type=int, default=5, help="Only used by the onecycle scheduler")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0 disables mixup)")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path("results") / args.dataset)
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, mixup_alpha=0.0):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_pred, y_true, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    labels_idx = list(range(len(class_names)))
    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_idx).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels_idx).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels_idx, target_names=class_names, output_dict=True),
    }
    return metrics, y_pred, y_true, probs


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dir = Path(args.output_dir) / f"{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data first, so the number of classes is known for "folder" datasets
    img_size = get_input_size(args.arch)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.dataset, args.data_dir, img_size=img_size,
        batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = len(class_names)

    model, img_size = get_model(args.arch, num_classes=num_classes, pretrained=True)
    total_params, trainable_params = count_parameters(model)
    print(f"Architecture: {args.arch} | dataset: {args.dataset} ({num_classes} classes) | "
          f"image size: {img_size}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    model = model.to(device)

    # Inverse-frequency class weights, normalised to sum to num_classes
    class_counts = np.bincount(train_loader.dataset.labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10, epochs=args.epochs,
                               steps_per_epoch=len(train_loader),
                               pct_start=args.warmup_epochs / args.epochs)

    best_val_acc, best_val_bal_acc, patience_counter = 0.0, 0.0, 0
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc",
                               "val_bal_acc", "val_f1", "lr"]}
    print(f"\n{'=' * 60}\nStarting training: {args.arch} for {args.epochs} epochs\n{'=' * 60}\n")
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scheduler if args.scheduler == "onecycle" else None,
                device, args.mixup_alpha)
            if args.scheduler == "cosine":
                scheduler.step()

            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, class_names)
            current_lr = optimizer.param_groups[0]["lr"]
            for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                         ("val_loss", val_metrics["loss"]), ("val_acc", val_metrics["accuracy"]),
                         ("val_bal_acc", val_metrics["balanced_accuracy"]),
                         ("val_f1", val_metrics["f1_macro"]), ("lr", current_lr)]:
                history[k].append(v)

            print(f"Epoch [{epoch + 1:3d}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                  f"BalAcc: {val_metrics['balanced_accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {time.time() - epoch_start:.1f}s")

            # Model selection on validation balanced accuracy
            if val_metrics["balanced_accuracy"] > best_val_bal_acc:
                best_val_bal_acc = val_metrics["balanced_accuracy"]
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "val_bal_acc": best_val_bal_acc,
                    "args": vars(args),
                    "class_names": class_names,
                }, run_dir / "best_model.pth")
                print(f"  * New best model saved (BalAcc: {best_val_bal_acc:.4f})")
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {args.patience} epochs)")
                break
    except Exception as e:  # keep the best checkpoint usable if training crashes mid-way
        print(f"\nWARNING: training stopped with an error: {e}\n"
              f"Continuing to test evaluation with the best saved model...\n")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes. "
          f"Best val balanced accuracy: {best_val_bal_acc:.4f}")
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if not (run_dir / "best_model.pth").exists():
        print("ERROR: no checkpoint was saved, skipping test evaluation.")
        return

    # ── Test evaluation with the best checkpoint ─────────────────────────────
    print(f"\n{'=' * 60}\nEvaluating best model on the TEST split...\n{'=' * 60}\n")
    checkpoint = torch.load(run_dir / "best_model.pth", map_location="cpu")
    model_eval, _ = get_model(args.arch, num_classes=num_classes, pretrained=False)
    model_eval.load_state_dict(checkpoint["model_state_dict"])
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        test_metrics, test_preds, test_labels, test_probs = evaluate(
            model_eval.to(device), test_loader, criterion, device, class_names)
    except RuntimeError as e:
        if "CUDA" not in str(e):
            raise
        print(f"  GPU evaluation failed ({e}), falling back to CPU...")
        cpu = torch.device("cpu")
        test_metrics, test_preds, test_labels, test_probs = evaluate(
            model_eval.to(cpu), test_loader, nn.CrossEntropyLoss(), cpu, class_names)

    print(f"\n{'-' * 40}\nTEST RESULTS for {args.arch}\n{'-' * 40}")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "precision_macro", "recall_macro"]:
        print(f"  {k:<18} {test_metrics[k]:.4f}")
    print(f"\n  Per-class F1: {dict(zip(class_names, np.round(test_metrics['per_class_f1'], 4)))}")
    print("\n  Confusion matrix (rows = true, cols = predicted):")
    for name, row in zip(class_names, test_metrics["confusion_matrix"]):
        print(f"  {name:>14} " + "".join(f"{v:>7d}" for v in row))

    with open(run_dir / "test_results.json", "w") as f:
        json.dump({
            "architecture": args.arch,
            "dataset": args.dataset,
            "num_classes": num_classes,
            "class_names": class_names,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_epoch": checkpoint["epoch"],
            "training_time_minutes": total_time / 60,
            "val_metrics": {"accuracy": best_val_acc, "balanced_accuracy": best_val_bal_acc},
            "test_metrics": test_metrics,
        }, f, indent=2)
    np.savez(run_dir / "test_predictions.npz",
             predictions=test_preds, labels=test_labels, probabilities=test_probs)
    print(f"\nAll results saved to {run_dir}")


if __name__ == "__main__":
    main()
