"""
Model factory for fragment style classification.

Every architecture is created with ImageNet (or self-supervised) pretrained weights and a
new classification head with `num_classes` outputs. Pretrained weights are downloaded at
run time from torchvision, torch.hub, timm / Hugging Face Hub, or open_clip, and are
covered by their own licenses (see the License section of the README).
"""

import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_ARCHS = [
    # Used in the paper
    "resnet50", "efficientnet_b3", "convnext_small",
    "vit_b_16", "swin_t", "maxvit_t",
    "dinov2_small", "clip_vit_b32", "dinov3_small", "dinov3_base", "dinov3_large",
    "mobilenet_v3_small", "mobilenet_v3_large", "mobilevit_s", "efficientvit_b0", "fastvit_t8",
    # Also available
    "resnet101", "efficientnet_b4", "convnext_base", "vit_l_16", "swin_s", "dinov2_base",
]

# Native input resolution of each architecture (pixels)
_INPUT_SIZES = {"efficientnet_b3": 300, "efficientnet_b4": 380, "mobilevit_s": 256, "fastvit_t8": 256}


def get_input_size(arch_name):
    """Input resolution used for `arch_name` (224 unless listed in _INPUT_SIZES)."""
    arch_name = arch_name.lower().strip()
    if arch_name not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: {', '.join(SUPPORTED_ARCHS)}")
    return _INPUT_SIZES.get(arch_name, 224)


def get_model(arch_name, num_classes=4, pretrained=True):
    """
    Create a classification model with a given architecture.

    Args:
        arch_name: Architecture name, one of SUPPORTED_ARCHS.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.

    Returns:
        model, img_size (int)
    """
    arch_name = arch_name.lower().strip()

    # ── ResNet ──────────────────────────────────────────────────────
    if arch_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 224

    elif arch_name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 224

    # ── EfficientNet ────────────────────────────────────────────────
    elif arch_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 300

    elif arch_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 380

    # ── ConvNeXt ────────────────────────────────────────────────────
    elif arch_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 224

    elif arch_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 224

    # ── Vision Transformer (ViT) ────────────────────────────────────
    elif arch_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, 224

    elif arch_name == "vit_l_16":
        weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
        model = models.vit_l_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, 224

    # ── Swin Transformer ────────────────────────────────────────────
    elif arch_name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model, 224

    elif arch_name == "swin_s":
        weights = models.Swin_S_Weights.DEFAULT if pretrained else None
        model = models.swin_s(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model, 224

    # ── DINOv2 (via torch.hub) ──────────────────────────────────────
    elif arch_name == "dinov2_small":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        embed_dim = model.embed_dim  # 384
        model.head = nn.Linear(embed_dim, num_classes)
        # Override forward to use cls token + head

        def forward_with_head(x):
            features = model.forward_features(x)
            cls_token = features["x_norm_clstoken"]
            return model.head(cls_token)

        model.forward = forward_with_head
        return model, 224

    elif arch_name == "dinov2_base":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained)
        embed_dim = model.embed_dim  # 768
        model.head = nn.Linear(embed_dim, num_classes)

        def forward_with_head(x):
            features = model.forward_features(x)
            cls_token = features["x_norm_clstoken"]
            return model.head(cls_token)

        model.forward = forward_with_head
        return model, 224

    # ── CLIP ViT (via open_clip) ────────────────────────────────────
    elif arch_name == "clip_vit_b32":
        try:
            import open_clip
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            # Use visual encoder only
            visual = clip_model.visual
            embed_dim = visual.output_dim  # 512

            class CLIPClassifier(nn.Module):
                def __init__(self, visual_encoder, embed_dim, num_classes):
                    super().__init__()
                    self.visual = visual_encoder
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Dropout(0.2),
                        nn.Linear(embed_dim, num_classes),
                    )

                def forward(self, x):
                    with torch.no_grad():
                        features = self.visual(x)
                    return self.classifier(features.float())

            model = CLIPClassifier(visual, embed_dim, num_classes)
            # Only train classifier head
            for param in model.visual.parameters():
                param.requires_grad = False
            return model, 224

        except ImportError:
            raise ImportError("open_clip is required for CLIP models: pip install open_clip_torch")

    # ── MaxViT ──────────────────────────────────────────────────────
    elif arch_name == "maxvit_t":
        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        model = models.maxvit_t(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 224

    # ── Edge-device models: MobileNetV3 (torchvision) ─────────────
    elif arch_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 224

    elif arch_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, 224

    # ── Edge-device models: MobileViT, EfficientViT, FastViT (timm) ─
    elif arch_name in ("mobilevit_s", "efficientvit_b0", "fastvit_t8"):
        import timm
        img_sizes = {"mobilevit_s": 256, "efficientvit_b0": 224, "fastvit_t8": 256}
        model = timm.create_model(arch_name, pretrained=pretrained, num_classes=num_classes)
        return model, img_sizes[arch_name]

    # ── DINOv3 (via HuggingFace transformers, gated) ────────────────
    elif arch_name in ("dinov3_small", "dinov3_base", "dinov3_large"):
        import os
        from transformers import AutoModel as HFAutoModel

        hf_token = os.environ.get("HF_TOKEN", None)
        model_ids = {
            "dinov3_small": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3_base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        }
        embed_dims = {"dinov3_small": 384, "dinov3_base": 768, "dinov3_large": 1024}

        model_id = model_ids[arch_name]
        embed_dim = embed_dims[arch_name]

        # Use "eager" attention to avoid SDPA CUDA kernel crashes during
        # backprop through RoPE on certain GPU/CUDA configurations.
        backbone = HFAutoModel.from_pretrained(
            model_id, token=hf_token, attn_implementation="eager"
        )

        class DINOv3Classifier(nn.Module):
            def __init__(self, backbone, embed_dim, num_classes):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim, num_classes),
                )

            def forward(self, x):
                outputs = self.backbone(pixel_values=x)
                cls_token = outputs.pooler_output  # (B, embed_dim)
                return self.head(cls_token)

        model = DINOv3Classifier(backbone, embed_dim, num_classes)
        return model, 224

    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: {', '.join(SUPPORTED_ARCHS)}")


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
