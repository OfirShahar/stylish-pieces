"""
Traditional-ML baselines (Random Forest and Gradient Boosting) for the Pompeian Styles dataset.

Features: frozen pretrained deep features (ResNet-50, DINOv2-S, ConvNeXt-S), HOG + colour
histograms, or raw pixels; optionally reduced with PCA.

Example (as in the paper):
    python train_random_forest.py --data_dir /path/to/pompeian-styles \
        --feature_extractor all --n_estimators 500 --pca_components 256
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torchvision import transforms, models


def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder with the Pompeian Styles npz files")
    parser.add_argument("--output_dir", type=str, default="results/pompeian")
    parser.add_argument("--feature_extractor", type=str, default="resnet50",
                        choices=["resnet50", "dinov2_small", "convnext_small", "raw_pixels", "hog",
                                 "all"],
                        help="Feature extractor to use")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--pca_components", type=int, default=256,
                        help="PCA components (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(data_dir, split):
    """Load images and labels for a split."""
    images = np.load(f"{data_dir}/{split}_images.npz")["images"]  # (N, 256, 256, 4)
    labels = np.load(f"{data_dir}/{split}_labels.npz")["labels"]  # (N,)
    labels = labels - 1  # Convert 1-4 to 0-3
    return images, labels


def rgba_to_rgb(images):
    """Convert RGBA images to RGB with white background."""
    alpha = images[:, :, :, 3:4].astype(np.float32) / 255.0
    rgb = images[:, :, :, :3].astype(np.float32)
    white_bg = np.ones_like(rgb) * 255.0
    return (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)


def extract_hog_features(images_rgb, cell_size=16, nbins=9):
    """Extract HOG-like features from images."""
    from skimage.feature import hog as skimage_hog
    features = []
    for img in images_rgb:
        # Convert to grayscale
        gray = np.mean(img.astype(np.float32), axis=2)
        feat = skimage_hog(
            gray, orientations=nbins, pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(2, 2), feature_vector=True
        )
        features.append(feat)
    return np.array(features)


def extract_color_histogram_features(images_rgb, nbins=32):
    """Extract color histogram features."""
    features = []
    for img in images_rgb:
        hist_r = np.histogram(img[:, :, 0], bins=nbins, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=nbins, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=nbins, range=(0, 256))[0]
        feat = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        feat = feat / (feat.sum() + 1e-8)  # normalize
        features.append(feat)
    return np.array(features)


@torch.no_grad()
def extract_deep_features(images_rgb, backbone_name="resnet50", batch_size=64):
    """Extract features using a pretrained deep network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load backbone
    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()  # Remove classification head → 2048-dim features
        feat_dim = 2048
    elif backbone_name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier = nn.Identity()
        feat_dim = 768
    elif backbone_name == "dinov2_small":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        feat_dim = model.embed_dim  # 384

        original_model = model

        class DINOFeatureExtractor(nn.Module):
            def __init__(self, dino_model):
                super().__init__()
                self.dino = dino_model

            def forward(self, x):
                features = self.dino.forward_features(x)
                return features["x_norm_clstoken"]

        model = DINOFeatureExtractor(original_model)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    model = model.to(device).eval()
    print(f"  Feature dimension: {feat_dim}")

    # Extract features
    all_features = []
    n = len(images_rgb)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for i in range(start, end):
            pil_img = Image.fromarray(images_rgb[i])
            batch_imgs.append(transform(pil_img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        features = model(batch_tensor)
        feat_np = features.cpu().numpy()
        # Flatten spatial dims if present (e.g. ConvNeXt outputs (B, C, 1, 1))
        if feat_np.ndim > 2:
            feat_np = feat_np.reshape(feat_np.shape[0], -1)
        all_features.append(feat_np)
        if (start // batch_size) % 10 == 0:
            print(f"  Extracted features for {end}/{n} images")

    return np.concatenate(all_features, axis=0)


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test,
                      n_estimators=500, max_depth=None, seed=42,
                      feature_name="", output_dir=None):
    """Train and evaluate a Random Forest classifier."""
    print(f"\n{'='*60}")
    print(f"Random Forest with {feature_name} features")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"{'='*60}")

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train RF
    print(f"\n  Training Random Forest (n_estimators={n_estimators})...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train_s, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    # Also train Gradient Boosting for comparison
    print(f"\n  Training Gradient Boosting (n_estimators=300)...")
    start = time.time()
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed,
        verbose=1,
    )
    gb.fit(X_train_s, y_train)
    gb_train_time = time.time() - start
    print(f"  Training time: {gb_train_time:.1f}s")

    class_names = ["Style I", "Style II", "Style III", "Style IV"]
    results = {}

    for name, model_clf in [("Random Forest", rf), ("Gradient Boosting", gb)]:
        # Validation
        val_preds = model_clf.predict(X_val_s)
        val_acc = accuracy_score(y_val, val_preds)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)

        # Test
        test_preds = model_clf.predict(X_test_s)
        test_probs = model_clf.predict_proba(X_test_s)

        test_acc = accuracy_score(y_test, test_preds)
        test_bal_acc = balanced_accuracy_score(y_test, test_preds)
        test_f1_macro = f1_score(y_test, test_preds, average="macro")
        test_f1_weighted = f1_score(y_test, test_preds, average="weighted")
        test_precision = precision_score(y_test, test_preds, average="macro")
        test_recall = recall_score(y_test, test_preds, average="macro")
        cm = confusion_matrix(y_test, test_preds)

        print(f"\n  {'─'*40}")
        print(f"  {name} + {feature_name} — TEST RESULTS")
        print(f"  {'─'*40}")
        print(f"    Accuracy:          {test_acc:.4f}")
        print(f"    Balanced Accuracy: {test_bal_acc:.4f}")
        print(f"    F1 (macro):        {test_f1_macro:.4f}")
        print(f"    F1 (weighted):     {test_f1_weighted:.4f}")
        print(f"    Precision (macro): {test_precision:.4f}")
        print(f"    Recall (macro):    {test_recall:.4f}")
        print(f"\n    Per-class F1: {f1_score(y_test, test_preds, average=None).tolist()}")
        print(f"\n    Confusion Matrix:")
        print(f"    {'':>12} {'Pred I':>8} {'Pred II':>8} {'Pred III':>8} {'Pred IV':>8}")
        for i, cname in enumerate(class_names):
            print(f"    {cname:>12} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d} {cm[i][3]:>8d}")

        result = {
            "method": f"{name}_{feature_name}",
            "feature_extractor": feature_name,
            "classifier": name,
            "feature_dim": X_train.shape[1],
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(val_bal_acc),
            "test_metrics": {
                "accuracy": float(test_acc),
                "balanced_accuracy": float(test_bal_acc),
                "f1_macro": float(test_f1_macro),
                "f1_weighted": float(test_f1_weighted),
                "precision_macro": float(test_precision),
                "recall_macro": float(test_recall),
                "confusion_matrix": cm.tolist(),
                "per_class_f1": f1_score(y_test, test_preds, average=None).tolist(),
                "classification_report": classification_report(
                    y_test, test_preds, target_names=class_names, output_dict=True
                ),
            },
        }
        results[f"{name}_{feature_name}"] = result

        # Save predictions
        if output_dir:
            safe_name = f"{name}_{feature_name}".replace(" ", "_").lower()
            np.savez(
                output_dir / f"test_predictions_{safe_name}.npz",
                predictions=test_preds,
                labels=y_test,
                probabilities=test_probs,
            )

    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create output directory
    run_dir = Path(args.output_dir) / f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print("Loading data...")
    train_imgs, y_train = load_split(args.data_dir, "train")
    val_imgs, y_val = load_split(args.data_dir, "val")
    test_imgs, y_test = load_split(args.data_dir, "test")
    print(f"  Train: {train_imgs.shape}, Val: {val_imgs.shape}, Test: {test_imgs.shape}")

    # Convert to RGB
    print("Converting RGBA to RGB...")
    train_rgb = rgba_to_rgb(train_imgs)
    val_rgb = rgba_to_rgb(val_imgs)
    test_rgb = rgba_to_rgb(test_imgs)

    all_results = {}
    feature_extractors = [args.feature_extractor] if args.feature_extractor != "all" else \
        ["resnet50", "dinov2_small", "convnext_small", "hog"]

    for feat_name in feature_extractors:
        print(f"\n{'#'*60}")
        print(f"Feature extractor: {feat_name}")
        print(f"{'#'*60}")

        if feat_name == "raw_pixels":
            # Downscale and flatten
            from skimage.transform import resize
            size = 64
            X_train = np.array([resize(img, (size, size), anti_aliasing=True).flatten()
                                for img in train_rgb])
            X_val = np.array([resize(img, (size, size), anti_aliasing=True).flatten()
                              for img in val_rgb])
            X_test = np.array([resize(img, (size, size), anti_aliasing=True).flatten()
                               for img in test_rgb])

        elif feat_name == "hog":
            print("  Extracting HOG features...")
            X_train = extract_hog_features(train_rgb)
            X_val = extract_hog_features(val_rgb)
            X_test = extract_hog_features(test_rgb)
            # Also add color histograms
            print("  Extracting color histogram features...")
            ch_train = extract_color_histogram_features(train_rgb)
            ch_val = extract_color_histogram_features(val_rgb)
            ch_test = extract_color_histogram_features(test_rgb)
            X_train = np.concatenate([X_train, ch_train], axis=1)
            X_val = np.concatenate([X_val, ch_val], axis=1)
            X_test = np.concatenate([X_test, ch_test], axis=1)

        else:
            # Deep features
            print(f"  Extracting {feat_name} features...")
            X_train = extract_deep_features(train_rgb, feat_name)
            X_val = extract_deep_features(val_rgb, feat_name)
            X_test = extract_deep_features(test_rgb, feat_name)

        print(f"  Feature shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

        # Optional PCA
        if args.pca_components > 0 and X_train.shape[1] > args.pca_components:
            print(f"  Applying PCA: {X_train.shape[1]} → {args.pca_components}")
            pca = PCA(n_components=args.pca_components, random_state=args.seed)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

        # Run classifiers
        results = run_random_forest(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            seed=args.seed, feature_name=feat_name, output_dir=run_dir
        )
        all_results.update(results)

    # Save all results
    with open(run_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL RANDOM FOREST / GRADIENT BOOSTING RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':<45} {'Accuracy':>10} {'BalAcc':>10} {'F1-macro':>10}")
    print(f"{'─'*75}")
    for name, res in sorted(all_results.items(), key=lambda x: x[1]["test_metrics"]["balanced_accuracy"], reverse=True):
        tm = res["test_metrics"]
        print(f"{name:<45} {tm['accuracy']:>10.4f} {tm['balanced_accuracy']:>10.4f} {tm['f1_macro']:>10.4f}")

    print(f"\nAll results saved to {run_dir}")


if __name__ == "__main__":
    main()
