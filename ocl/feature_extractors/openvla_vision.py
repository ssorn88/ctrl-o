import math

import torch
import torch.nn as nn
import timm
import ocl.typing
from PIL import Image
from torchvision.transforms import Compose, Resize


class DinoSigLIPImageTransform:
    def __init__(self, dino_image_transform, siglip_image_transform):
        self.dino_image_transform = dino_image_transform
        self.siglip_image_transform = siglip_image_transform

    def __call__(self, img: Image, **kwargs):
        return {
            "dino": self.dino_image_transform(img, **kwargs),
            "siglip": self.siglip_image_transform(img, **kwargs),
        }


class OpenVLADinoSigLIPFeatureExtractor(nn.Module):
    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        default_image_size: int = 224,
        image_resize_strategy: str = "resize-naive",
        freeze: bool = True,
        out_dim: int = 256,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        if vision_backbone_id == "dinosiglip-vit-so-224px":
            dino_name = "vit_large_patch14_reg4_dinov2.lvd142m"
            siglip_name = "vit_so400m_patch14_siglip_224"
        elif vision_backbone_id == "dinosiglip-vit-so-384px":
            dino_name = "vit_large_patch14_reg4_dinov2.lvd142m"
            siglip_name = "vit_so400m_patch14_siglip_384"
        else:
            raise ValueError(f"Unsupported vision_backbone_id: {vision_backbone_id}")

        self.default_image_size = default_image_size
        self.image_resize_strategy = image_resize_strategy

        self.dino_featurizer = timm.create_model(
            dino_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=default_image_size,
        )
        self.siglip_featurizer = timm.create_model(
            siglip_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=default_image_size,
        )

        self.dino_featurizer.eval()
        self.siglip_featurizer.eval()

        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)

        self.dino_data_cfg["input_size"] = (3, default_image_size, default_image_size)
        self.siglip_data_cfg["input_size"] = (3, default_image_size, default_image_size)

        self.in_dim = self.dino_featurizer.num_features + self.siglip_featurizer.num_features
        self.out_dim = out_dim
        self.proj = nn.Linear(self.in_dim, self.out_dim)

        if freeze:
            for p in self.dino_featurizer.parameters():
                p.requires_grad = False
            for p in self.siglip_featurizer.parameters():
                p.requires_grad = False

        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        if image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose)
            assert isinstance(default_siglip_transform, Compose)
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (default_image_size, default_image_size)
            dino_transform = Compose(
                [Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation)]
                + list(default_dino_transform.transforms[1:])
            )
            siglip_transform = Compose(
                [Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation)]
                + list(default_siglip_transform.transforms[1:])
            )
            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)
        else:
            self.image_transform = DinoSigLIPImageTransform(default_dino_transform, default_siglip_transform)

    def get_transform(self):
        return self.image_transform

    @property
    def feature_dim(self):
        return self.out_dim

    def _make_positions(self, b, n, device, dtype):
        side = int(math.sqrt(n))
        if side * side != n:
            pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(-1).repeat(1, 2)
            return pos.unsqueeze(0).repeat(b, 1, 1)

        y, x = torch.meshgrid(
            torch.arange(side, device=device, dtype=dtype),
            torch.arange(side, device=device, dtype=dtype),
            indexing="ij",
        )
        pos = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
        return pos.unsqueeze(0).repeat(b, 1, 1)

    def _tensor_batch_to_dual_dict(self, batch_tensor: torch.Tensor):
        """
        batch_tensor: [B, 3, H, W] normalized with ImageNet stats
        -> dual input dict without PIL conversion
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch_tensor.device).view(1, 3, 1, 1)

        x = batch_tensor * std + mean
        x = x.clamp(0, 1)

        dino_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        dino_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        dino = (x - dino_mean) / dino_std

        siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        siglip_std = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        siglip = (x - siglip_mean) / siglip_std

        return {
            "dino": dino,
            "siglip": siglip,
        }

    def forward(self, inputs):
        x = inputs["input"]["image"]

        if isinstance(x, dict):
            dual_inputs = x
        elif isinstance(x, torch.Tensor):
            dual_inputs = self._tensor_batch_to_dual_dict(x)
        else:
            raise TypeError(f"Unsupported input image type: {type(x)}")

        dino_feats = self.dino_featurizer.forward_features(dual_inputs["dino"])
        siglip_feats = self.siglip_featurizer.forward_features(dual_inputs["siglip"])

        if isinstance(dino_feats, (list, tuple)):
            dino_feats = dino_feats[-1]
        if isinstance(siglip_feats, (list, tuple)):
            siglip_feats = siglip_feats[-1]

        if dino_feats.ndim != 3 or siglip_feats.ndim != 3:
            raise ValueError(f"Unexpected feature shape: {dino_feats.shape}, {siglip_feats.shape}")

        min_tokens = min(dino_feats.shape[1], siglip_feats.shape[1])
        dino_feats = dino_feats[:, -min_tokens:, :]
        siglip_feats = siglip_feats[:, -min_tokens:, :]

        feats = torch.cat([dino_feats, siglip_feats], dim=2)
        feats = self.proj(feats)

        b, n, _ = feats.shape
        positions = self._make_positions(b, n, feats.device, feats.dtype)

        return ocl.typing.FeatureExtractorOutput(
            features=feats,
            positions=positions,
        )