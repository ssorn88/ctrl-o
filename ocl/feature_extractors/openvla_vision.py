import torch
import torch.nn as nn
import timm
import ocl.typing


class OpenVLASigLIPFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        timm_model_name: str = None,
        img_size: int = 224,
        freeze: bool = True,
        dynamic_img_size: bool = False,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        resolved_model_name = model_name or timm_model_name or "vit_so400m_patch14_siglip_224"
        self.model_name = resolved_model_name

        self.featurizer = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=dynamic_img_size,
        )

        self.in_dim = self.featurizer.num_features
        self.out_dim = 256
        self.proj = nn.Linear(self.in_dim, self.out_dim)

        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)

        if freeze:
            for p in self.featurizer.parameters():
                p.requires_grad = False

    def get_transform(self):
        return timm.data.create_transform(**self.data_cfg, is_training=True)

    @property
    def feature_dim(self):
        return self.out_dim

    def forward(self, inputs):
        x = inputs["input"]["image"]
        feats = self.featurizer.forward_features(x)

        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        if feats.ndim != 3:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        # if feats.shape[1] > 1:
        #     feats = feats[:, 1:, :]

        feats = self.proj(feats)

        b, n, _ = feats.shape
        positions = torch.arange(n, device=feats.device, dtype=feats.dtype)
        positions = positions.unsqueeze(-1).repeat(1, 2)
        positions = positions.unsqueeze(0).repeat(b, 1, 1)

        return ocl.typing.FeatureExtractorOutput(
            features=feats,
            positions=positions,
        )