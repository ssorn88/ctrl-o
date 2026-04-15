# ocl/feature_extractors/openvla_dinosiglip_local.py

import math

import torch
import torch.nn as nn
import ocl.typing
from torchvision.transforms.functional import to_pil_image

from ._dinosiglip_vit_local import DinoSigLIPViTBackbone


class OpenVLADinoSigLIPFeatureExtractor(nn.Module):
    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        image_resize_strategy: str = "resize-naive",
        default_image_size: int = 224,
        freeze: bool = True,
        out_dim: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.backbone = DinoSigLIPViTBackbone(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy=image_resize_strategy,
            default_image_size=default_image_size,
        )

        self.in_dim = self.backbone.embed_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(self.in_dim, self.out_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_transform(self):
        return self.backbone.get_image_transform()

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
        batch_tensor: [B, C, H, W]
        -> {"dino": [B, 3, H', W'], "siglip": [B, 3, H', W']}
        """
        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B,C,H,W], got {batch_tensor.shape}")

        transform = self.get_transform()

        dino_list = []
        siglip_list = []

        # CPU/PIL 변환 후 backbone 전용 transform 적용
        for img in batch_tensor:
            img_cpu = img.detach().cpu()

            # 기존 파이프라인이 float tensor를 주는 경우를 대비
            # to_pil_image는 [0,1] float 또는 uint8 tensor를 기대
            if img_cpu.dtype != torch.uint8:
                img_cpu = img_cpu.clamp(0, 1)

            pil_img = to_pil_image(img_cpu)
            out = transform(pil_img)

            dino_list.append(out["dino"])
            siglip_list.append(out["siglip"])

        return {
            "dino": torch.stack(dino_list, dim=0).to(batch_tensor.device),
            "siglip": torch.stack(siglip_list, dim=0).to(batch_tensor.device),
        }

    def forward(self, inputs):
        pixel_values = inputs["input"]["image"]

        # 1) 이미 원하는 dict 형태로 들어온 경우
        if isinstance(pixel_values, dict):
            dual_inputs = pixel_values

        # 2) 현재 실제 케이스: [B,C,H,W] tensor
        elif isinstance(pixel_values, torch.Tensor):
            dual_inputs = self._tensor_batch_to_dual_dict(pixel_values)

        # 3) 혹시 list/tuple raw image라면 처리
        elif isinstance(pixel_values, (list, tuple)):
            transform = self.get_transform()
            transformed = [transform(img) for img in pixel_values]
            dual_inputs = {
                "dino": torch.stack([x["dino"] for x in transformed], dim=0),
                "siglip": torch.stack([x["siglip"] for x in transformed], dim=0),
            }
            dual_inputs = {k: v.to(self.proj.weight.device) for k, v in dual_inputs.items()}

        else:
            raise TypeError(f"Unsupported input image type: {type(pixel_values)}")

        feats = self.backbone(dual_inputs)  # [B, N, C_total]

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        if feats.ndim != 3:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        feats = self.proj(feats)
        b, n, _ = feats.shape
        positions = self._make_positions(b, n, feats.device, feats.dtype)

        return ocl.typing.FeatureExtractorOutput(
            features=feats,
            positions=positions,
        )