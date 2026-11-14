# Stylish Pieces: Deep Visual Style Recognition of Archaeological Fragments

Official code for the paper

> G. Elkin\*, O. I. Shahar\*, N. Alali, M. C. Napolitano, A. Zambrano, O. Ben-Shahar.
> **Stylish Pieces: Advancing Deep Visual Style Recognition of Archaeological Fragments.**
> *ACM Journal on Computing and Cultural Heritage*, 2026. [doi:10.1145/3849479](https://doi.org/10.1145/3849479)
> (\* equal contribution)

Archaeologists often need to know which painting style an isolated fresco fragment belongs to, for example to separate fragments that cannot come from the same wall. We show that modern image classifiers can recognise the style of a fragment directly from its partial, irregularly shaped visual content, without first "completing" the fragment with a generative model.

This repository contains everything needed to reproduce the paper's experiments:

- one training script for **16 architectures**: CNNs, vision transformers, a hybrid, self-supervised foundation models (DINOv2, DINOv3, CLIP), and lightweight edge-device networks;
- **Random Forest / Gradient Boosting** baselines on frozen deep features and HOG;
- an **evaluation script** for trained models on any split, including the square-tile control test set;
- the paper's **result files** (`paper_results/`) for comparison with your own runs.

It also runs **on new datasets** of fragment images (see [Using your own dataset](#using-your-own-dataset)).

## Datasets

| Dataset | Classes | Fragments (train / val / test) | Where to get it |
|---|---|---|---|
| **Pompeian Styles** (ours) | 4 Pompeian styles (I–IV) | 2,874 / 463 / 549 (+ 693 square tiles) | [Hugging Face: OfirShahar/pompeian-styles](https://huggingface.co/datasets/OfirShahar/pompeian-styles) |
| CLEOPATRA ([Cascone et al.](https://doi.org/10.1007/s12652-022-04472-x)) | 11 art-historical periods | 18,458 / 4,186 / 5,713 | from the dataset authors |

**Pompeian Styles** fragments were made by cutting 508 wall-painting photographs from the Pompeii Archaeological Park with the shapes of real fresco fragments scanned by the [RePAIR project](https://doi.org/10.5281/zenodo.13993089). All fragments of one source image are in the same split. See the dataset card for details. The dataset is licensed under CC BY-NC-ND 4.0 (non-commercial use only).

Download it with:

```bash
huggingface-cli download OfirShahar/pompeian-styles --repo-type dataset --local-dir data/pompeian-styles
```

**CLEOPATRA** is expected as `CLEOPATRA/{train,valid,test}/*.png`, where the class (1–11) is the first number of each file name (e.g. `10.33.11.png` is class 10, Cubism).

## Installation

Tested with Python 3.10, PyTorch 2.6 and CUDA 12.4 on NVIDIA RTX 4090 GPUs.

```bash
git clone https://github.com/OfirShahar/stylish-pieces.git
cd stylish-pieces
conda create -n stylish-pieces python=3.10 -y && conda activate stylish-pieces
pip install -r requirements.txt
```

The DINOv3 weights are gated on Hugging Face. Accept the license on the model page (e.g. [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)) and set `export HF_TOKEN=<your token>` before training a `dinov3_*` model. All other weights download automatically.

## Reproducing the paper

```bash
# 1. All 16 deep models on Pompeian Styles (Table "Results on the Pompeian Styles Dataset")
bash scripts/reproduce_pompeian.sh data/pompeian-styles

# 2. All 16 deep models on CLEOPATRA
bash scripts/reproduce_cleopatra.sh /path/to/CLEOPATRA

# 3. Traditional-ML baselines, square-tile control set, result tables
bash scripts/reproduce_analysis.sh data/pompeian-styles
```

Each training run writes to `results/<dataset>/<arch>_<timestamp>/`. A SLURM example is in [`scripts/slurm_example.sbatch`](scripts/slurm_example.sbatch).

To compare your numbers with ours, build the same table from the paper's result files:

```bash
python collect_results.py --results_dir paper_results/pompeian --output_dir paper_table
```

Every run uses seed 42 and deterministic cuDNN, but results can still differ slightly across GPUs and library versions.

### Training protocol

All models start from pretrained weights and are fine-tuned end to end. The exception is CLIP, whose image encoder is frozen and only a classification head is trained (linear probe). Fragments are alpha-composited onto white and resized to each architecture's native input size (224 px; 256 px for MobileViT-S and FastViT-T8; 300 px for EfficientNet-B3), with ImageNet normalisation.

- **Augmentation:** resize to size + 32 and random crop, horizontal and vertical flips, ±15° rotation, colour jitter, Mixup (α = 0.2).
- **Optimisation:** AdamW, weight decay 1e-2, cosine schedule, gradient clipping at 1.0, up to 50 epochs.
- **Loss:** cross-entropy with inverse-frequency class weights and label smoothing 0.1.
- **Model selection:** best validation balanced accuracy, early stopping after 15 epochs without improvement.

Settings that differ per model (exactly as used in the paper):

| Model | `--arch` | Pompeian: batch / lr | CLEOPATRA: batch / lr |
|---|---|---|---|
| ResNet-50 | `resnet50` | 32 / 1e-4 | 32 / 1e-4 |
| EfficientNet-B3 | `efficientnet_b3` | 24 / 1e-4 | 32 / 1e-4 |
| ConvNeXt-Small | `convnext_small` | 32 / 5e-5 | 32 / 1e-4 |
| ViT-B/16 | `vit_b_16` | 32 / 5e-5 | 32 / 1e-4 |
| Swin-T | `swin_t` | 32 / 5e-5 | 32 / 1e-4 |
| MaxViT-T | `maxvit_t` | 24 / 5e-5 | 32 / 1e-4 |
| DINOv2-Small | `dinov2_small` | 32 / 1e-4 | 32 / 1e-4 |
| CLIP ViT-B/32 | `clip_vit_b32` | 64 / 1e-3 (wd 1e-3, no Mixup) | 32 / 1e-4 |
| DINOv3 ViT-S | `dinov3_small` | 32 / 1e-4 | 32 / 1e-4 |
| DINOv3 ViT-B | `dinov3_base` | 32 / 5e-5 | 32 / 5e-5 |
| DINOv3 ViT-L | `dinov3_large` | 16 / 5e-5 | 16 / 5e-5 |
| MobileNetV3-Small | `mobilenet_v3_small` | 64 / 1e-4 | 64 / 1e-4 |
| MobileNetV3-Large | `mobilenet_v3_large` | 64 / 1e-4 | 64 / 1e-4 |
| MobileViT-S | `mobilevit_s` | 32 / 5e-5 | 32 / 5e-5 |
| EfficientViT-B0 | `efficientvit_b0` | 64 / 5e-5 | 64 / 5e-5 |
| FastViT-T8 | `fastvit_t8` | 32 / 5e-5 | 32 / 5e-5 |

### Paper results

Test-set results, ranked by balanced accuracy on Pompeian Styles. Parameter counts are for the 4-class models.

| Model | `--arch` | Params (M) | Pompeian Acc. | Pompeian Bal. Acc. | Pompeian F1 | CLEOPATRA Acc. | CLEOPATRA Bal. Acc. | CLEOPATRA F1 |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Small | `convnext_small` | 49.5 | 0.767 | **0.804** | **0.781** | 0.524 | 0.525 | 0.504 |
| DINOv3 ViT-L | `dinov3_large` | 303.1 | 0.740 | 0.772 | 0.755 | **0.696** | **0.695** | **0.690** |
| Swin-T | `swin_t` | 27.5 | 0.694 | 0.746 | 0.708 | 0.515 | 0.521 | 0.501 |
| DINOv3 ViT-B | `dinov3_base` | 85.7 | 0.710 | 0.735 | 0.720 | 0.579 | 0.580 | 0.561 |
| DINOv3 ViT-S | `dinov3_small` | 21.6 | 0.703 | 0.729 | 0.719 | 0.568 | 0.565 | 0.551 |
| MaxViT-T | `maxvit_t` | 30.4 | 0.645 | 0.709 | 0.674 | 0.571 | 0.569 | 0.559 |
| ViT-B/16 | `vit_b_16` | 85.8 | 0.656 | 0.689 | 0.667 | 0.511 | 0.507 | 0.494 |
| EfficientNet-B3 | `efficientnet_b3` | 10.7 | 0.668 | 0.685 | 0.679 | 0.541 | 0.542 | 0.525 |
| ResNet-50 | `resnet50` | 23.5 | 0.654 | 0.682 | 0.658 | 0.505 | 0.500 | 0.489 |
| MobileNetV3-Large | `mobilenet_v3_large` | 4.2 | 0.623 | 0.680 | 0.640 | 0.497 | 0.499 | 0.486 |
| MobileViT-S | `mobilevit_s` | 4.9 | 0.583 | 0.631 | 0.583 | 0.505 | 0.497 | 0.483 |
| FastViT-T8 | `fastvit_t8` | 3.3 | 0.625 | 0.625 | 0.629 | 0.507 | 0.501 | 0.476 |
| EfficientViT-B0 | `efficientvit_b0` | 2.1 | 0.596 | 0.614 | 0.599 | 0.493 | 0.491 | 0.479 |
| CLIP ViT-B/32 | `clip_vit_b32` | 87.9 | 0.536 | 0.600 | 0.546 | 0.478 | 0.479 | 0.460 |
| MobileNetV3-Small | `mobilenet_v3_small` | 1.5 | 0.548 | 0.584 | 0.562 | 0.489 | 0.486 | 0.470 |
| DINOv2-Small | `dinov2_small` | 22.1 | 0.459 | 0.496 | 0.460 | 0.348 | 0.343 | 0.322 |

`paper_results/` also contains the Random Forest / Gradient Boosting results and the square-tile test results.

## Training a single model

```bash
python train.py --dataset pompeian --data_dir data/pompeian-styles --arch convnext_small --lr 5e-5
```

Main options (`python train.py -h` lists all of them):

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `pompeian` | `pompeian`, `cleopatra`, or `folder` (your own data) |
| `--data_dir` | (required) | dataset folder |
| `--arch` | (required) | one of the architectures above (more are listed in `models.py`) |
| `--epochs`, `--patience` | 50, 15 | training length and early-stopping patience |
| `--batch_size`, `--lr`, `--weight_decay` | 32, 1e-4, 1e-2 | optimisation |
| `--mixup_alpha`, `--label_smoothing` | 0.2, 0.1 | regularisation (`--mixup_alpha 0` disables Mixup) |
| `--output_dir` | `results/<dataset>` | where runs are saved |

Each run folder contains `args.json`, `best_model.pth`, `training_history.json`, `test_results.json` (accuracy, balanced accuracy, macro/weighted F1, precision, recall, per-class F1, confusion matrix) and `test_predictions.npz` (predictions, labels, class probabilities).

## Evaluating trained models

`train.py` evaluates each model on the test split when training ends. To evaluate trained models again, on another split, or on another dataset with the same classes, use `evaluate.py`:

```bash
# All trained Pompeian models (latest run per architecture) on the test split
python evaluate.py --dataset pompeian --data_dir data/pompeian-styles \
    --results_dir results/pompeian --split test --output_dir results/eval_test

# The square-tile control set
python evaluate.py --dataset pompeian --data_dir data/pompeian-styles \
    --results_dir results/pompeian --split testset_square --output_dir results/eval_testset_square

# Specific runs (run folders or best_model.pth files), e.g. on CLEOPATRA
python evaluate.py --dataset cleopatra --data_dir /path/to/CLEOPATRA \
    --checkpoint results/cleopatra/dinov3_large_<timestamp> --output_dir results/eval_cleopatra
```

Splits are `train`, `val`, `test` and `testset_square` for Pompeian Styles, `train`, `valid` and `test` for CLEOPATRA, and any sub-folder for `--dataset folder`. For every model the script writes `metrics.json` (the same metrics as above) and `predictions.npz`, plus a `summary.json` for all models.

## Using your own dataset

**Option 1: class folders.** Arrange fragment images as below. RGBA images with a transparent background are composited onto white; RGB images are used as they are.

```
my_dataset/
├── train/<class_a>/*.png   train/<class_b>/*.png ...
├── val/<class_a>/*.png ...      (or valid/)
└── test/<class_a>/*.png ...
```

```bash
python train.py --dataset folder --data_dir my_dataset --arch convnext_small --lr 5e-5
```

Classes are the folder names in `train/`, in alphabetical order. The number of output classes and the class weights are set automatically.

**Option 2: the npz format of Pompeian Styles.** Save `<split>_images.npz` (key `images`, uint8 array `N×H×W×4`) and `<split>_labels.npz` (key `labels`, integers starting at 1) for `train`, `val` and `test`, and use `--dataset pompeian`. `NpzFragmentDataset` in `data.py` assumes 4 classes; change `POMPEIAN_CLASSES` there for a different class list. Extra test sets in this format (e.g. `mytest_images.npz`) can be evaluated with `evaluate.py --split mytest`.

**Avoid leakage.** If several fragments come from the same artwork, put all of them in the same split. Otherwise the test set measures how well a model recognises artworks it has already seen rather than new ones.

## Other scripts

| Script | What it does |
|---|---|
| `train_random_forest.py` | RF and GB baselines on frozen ResNet-50 / DINOv2 / ConvNeXt features or HOG + colour histograms, with PCA to 256 dimensions |
| `collect_results.py` | Ranking table (Markdown + JSON) from a results folder |

## Repository layout

```
├── train.py                  training + test evaluation (all datasets and models)
├── data.py                   dataset loaders (Pompeian npz, CLEOPATRA, class folders) and transforms
├── models.py                 model factory
├── runs.py                   finds trained runs in a results folder
├── train_random_forest.py    traditional-ML baselines
├── evaluate.py               evaluation of trained models on any split
├── collect_results.py        result tables
├── scripts/                  reproduction scripts and SLURM example
├── paper_results/            the paper's result files (metrics, settings, training curves)
├── requirements.txt
└── LICENSE                   Apache-2.0
```

## License

The code is released under the [Apache License 2.0](LICENSE). The Pompeian Styles dataset is licensed separately under CC BY-NC-ND 4.0 (see its dataset card).

This repository does not include any pretrained weights. They are downloaded at run time from their original sources and keep their own licenses:

- torchvision models (ResNet-50, EfficientNet-B3, ConvNeXt-Small, ViT-B/16, Swin-T, MaxViT-T, MobileNetV3): [torchvision](https://github.com/pytorch/vision), ImageNet-1k weights
- DINOv2: [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
- DINOv3: [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Models fine-tuned from DINOv3 are covered by this license too, so if you share such a checkpoint you must do so under its terms.
- CLIP ViT-B/32 (LAION-2B): [open_clip, MIT](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)
- EfficientViT-B0: [Apache-2.0](https://huggingface.co/timm/efficientvit_b0.r224_in1k)
- MobileViT-S: [Apple ml-cvnets license](https://github.com/apple/ml-cvnets/blob/main/LICENSE)
- FastViT-T8: [Apple ml-fastvit license](https://github.com/apple/ml-fastvit/blob/main/LICENSE)

## Citation

If you use this code or the dataset, please cite:

```bibtex
@article{elkin2026stylish,
  title     = {Stylish Pieces: Advancing Deep Visual Style Recognition of Archaeological Fragments},
  author    = {Elkin, Gur and Shahar, Ofir Itzhak and Alali, Nadav and Napolitano, Maria Cristina and Zambrano, Alessandra and Ben-Shahar, Ohad},
  journal   = {ACM Journal on Computing and Cultural Heritage},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3849479},
  url       = {https://doi.org/10.1145/3849479}
}
```

## Acknowledgements

The wall-painting images come from the Pompeii Archaeological Park and the fragment shapes from the [RePAIR dataset](https://doi.org/10.5281/zenodo.13993089). We thank the authors of the pretrained models and libraries we build on.
