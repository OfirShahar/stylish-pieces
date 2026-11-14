"""
Datasets for fragment style classification.

Three formats are supported (select with `--dataset` in train.py):

  pompeian   Our Pompeian Styles dataset: <split>_images.npz / <split>_labels.npz
             (RGBA uint8, labels 1..K). Splits: train, val, test (+ testset_square).
  cleopatra  The CLEOPATRA benchmark: <root>/{train,valid,test}/*.png, with the class
             (1..11) encoded as the first number of the file name, e.g. "10.33.11.png".
  folder     Any new dataset laid out as <root>/{train,val,test}/<class_name>/*.png
             (RGBA or RGB images; class names sorted alphabetically give the label order).

All fragments are alpha-composited onto a white background and fed to the networks as RGB.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

POMPEIAN_CLASSES = ["Style I", "Style II", "Style III", "Style IV"]
CLEOPATRA_CLASSES = [
    "Prehistory", "Egyptian", "Etruscan", "Greek", "Roman",
    "Byzantine", "Gothic", "Renaissance", "Impressionism",
    "Cubism", "Surrealism",
]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def rgba_to_rgb(img):
    """Composite an RGBA uint8 array (..., H, W, 4) onto a white background."""
    alpha = img[..., 3:4].astype(np.float32) / 255.0
    rgb = img[..., :3].astype(np.float32)
    return (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)


def pil_to_rgb_on_white(img):
    """Composite a PIL image onto white (if it has transparency) and return RGB."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ─── Pompeian Styles (npz) ────────────────────────────────────────────────────
class NpzFragmentDataset(Dataset):
    """Fragments stored as <split>_images.npz (N,H,W,4 uint8) + <split>_labels.npz (1..K)."""

    def __init__(self, data_dir, split="train", transform=None, num_classes=4):
        self.transform = transform
        self.images = np.load(os.path.join(data_dir, f"{split}_images.npz"))["images"]
        self.labels = np.load(os.path.join(data_dir, f"{split}_labels.npz"))["labels"] - 1
        print(f"[{split}] Loaded {len(self.images)} fragments, "
              f"class distribution: {np.bincount(self.labels, minlength=num_classes)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(rgba_to_rgb(self.images[idx]), mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─── CLEOPATRA ────────────────────────────────────────────────────────────────
class CleopatraDataset(Dataset):
    """CLEOPATRA fragments: <root>/<split>/<label>.<...>.png with label in 1..11."""

    def __init__(self, data_dir, split="train", transform=None):
        self.transform = transform
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.file_paths, labels = [], []
        for fname in sorted(os.listdir(split_dir)):
            if not fname.lower().endswith(".png"):
                continue
            label = int(os.path.splitext(fname)[0].split(".")[0])
            if not 1 <= label <= len(CLEOPATRA_CLASSES):
                print(f"Warning: skipping {fname}, unexpected label {label}")
                continue
            self.file_paths.append(os.path.join(split_dir, fname))
            labels.append(label - 1)
        self.labels = np.array(labels, dtype=np.int64)
        print(f"[{split}] Loaded {len(self.file_paths)} fragments, class distribution: "
              f"{np.bincount(self.labels, minlength=len(CLEOPATRA_CLASSES))}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = pil_to_rgb_on_white(Image.open(self.file_paths[idx]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─── Generic class-folder layout (for new datasets) ──────────────────────────
class FolderFragmentDataset(Dataset):
    """<root>/<split>/<class_name>/*.png. Class order = sorted folder names of the train split."""

    def __init__(self, data_dir, split="train", transform=None, class_names=None):
        self.transform = transform
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        if class_names is None:
            class_names = sorted(d for d in os.listdir(split_dir)
                                 if os.path.isdir(os.path.join(split_dir, d)))
        self.class_names = list(class_names)

        self.file_paths, labels = [], []
        for label, cls in enumerate(self.class_names):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    self.file_paths.append(os.path.join(cls_dir, fname))
                    labels.append(label)
        self.labels = np.array(labels, dtype=np.int64)
        print(f"[{split}] Loaded {len(self.file_paths)} images, class distribution: "
              f"{np.bincount(self.labels, minlength=len(self.class_names))}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = pil_to_rgb_on_white(Image.open(self.file_paths[idx]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─── Transforms and loaders ──────────────────────────────────────────────────
def get_transforms(img_size=224, train=False, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _split_name(data_dir, *candidates):
    for c in candidates:
        if os.path.isdir(os.path.join(data_dir, c)):
            return c
    return candidates[0]


def build_datasets(dataset, data_dir, img_size=224):
    """Return (train_ds, val_ds, test_ds, class_names)."""
    train_tf, eval_tf = get_transforms(img_size, train=True), get_transforms(img_size)
    if dataset == "pompeian":
        mk = lambda s, tf: NpzFragmentDataset(data_dir, s, tf, len(POMPEIAN_CLASSES))
        return mk("train", train_tf), mk("val", eval_tf), mk("test", eval_tf), POMPEIAN_CLASSES
    if dataset == "cleopatra":
        mk = lambda s, tf: CleopatraDataset(data_dir, s, tf)
        return mk("train", train_tf), mk("valid", eval_tf), mk("test", eval_tf), CLEOPATRA_CLASSES
    if dataset == "folder":
        train_ds = FolderFragmentDataset(data_dir, "train", train_tf)
        names = train_ds.class_names
        val_ds = FolderFragmentDataset(data_dir, _split_name(data_dir, "val", "valid"), eval_tf, names)
        test_ds = FolderFragmentDataset(data_dir, "test", eval_tf, names)
        return train_ds, val_ds, test_ds, names
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: pompeian, cleopatra, folder")


def build_eval_dataset(dataset, data_dir, split, img_size=224, class_names=None):
    """One split, with evaluation transforms. Returns (dataset, class_names).

    Split names: pompeian → train / val / test / testset_square;
    cleopatra → train / valid / test; folder → any sub-folder of data_dir.
    """
    tf = get_transforms(img_size)
    if dataset == "pompeian":
        names = list(class_names or POMPEIAN_CLASSES)
        return NpzFragmentDataset(data_dir, split, tf, len(names)), names
    if dataset == "cleopatra":
        return CleopatraDataset(data_dir, split, tf), CLEOPATRA_CLASSES
    if dataset == "folder":
        ds = FolderFragmentDataset(data_dir, split, tf, class_names)
        return ds, ds.class_names
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: pompeian, cleopatra, folder")


def build_dataloaders(dataset, data_dir, img_size=224, batch_size=32, num_workers=4):
    """Return (train_loader, val_loader, test_loader, class_names)."""
    train_ds, val_ds, test_ds, class_names = build_datasets(dataset, data_dir, img_size)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
            class_names)
