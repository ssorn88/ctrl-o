import torch
import torch.nn as nn
import logging

# 복사해온 prismatic 모듈에서 팩토리 함수를 임포트합니다.
from prismatic.models.materialize import get_vision_backbone_and_transform

logger = logging.getLogger(__name__)


class OpenVLAVisionExtractor(nn.Module):
    """
    OpenVLA (Prismatic VLM)의 Vision Encoder를 ctrl-o에서 사용하기 위한 Wrapper.
    """

    def __init__(
            self,
            # VLA가 사용하는 해상도와 모델 ID를 지정합니다.
            # (예: "siglip-vit-so400m", "dinosiglip-vit-so-224px", "dinosiglip-vit-so-384px")
            vision_backbone_id: str = "dinosiglip-vit-so-224px",
            image_resize_strategy: str = "resize-naive",  # 또는 "letterbox"
            freeze: bool = True,
            **kwargs
    ):
        super().__init__()
        self.vision_backbone_id = vision_backbone_id

        logger.info(f"OpenVLA Vision Encoder 로드 중: {vision_backbone_id}")

        # 1. materialize.py의 팩토리 함수를 사용하여 모델과 전처리 모듈 로드
        self.vision_encoder, self.image_transform = get_vision_backbone_and_transform(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy=image_resize_strategy
        )

        # 2. 파라미터 동결 (Freeze)
        # Vision Encoder를 사전 학습된 상태로 고정하고 OCL 모듈만 학습합니다.
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()  # Dropout, BatchNorm 등을 비활성화

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 형태의 입력 이미지 텐서
                    (image_transform 전처리가 완료된 상태여야 함)
        Returns:
            features: [B, Sequence_length, Feature_dimension] 형태의 특징 텐서
        """
        # 1. Prismatic Vision Backbone 통과
        # Prismatic의 VisionBackbone은 내부적으로 forward()를 통해 피처를 반환합니다.
        features = self.vision_encoder(images)

        # 2. 차원 변환 (Dimension Mapping)
        # ctrl-o는 [Batch, Sequence_length(Patch 수), Feature_dimension] 형태를 기대합니다.

        # [B, C, H, W] 형태인 경우 (공간 정보가 유지된 맵)
        if features.dim() == 4:
            b, c, h, w = features.shape
            # [B, C, H*W] 로 평탄화 후 [B, H*W, C] 로 전치
            features = features.flatten(2).transpose(1, 2)

        # 이미 [B, Seq_len, Dim] 형태인 경우 (일반적인 ViT 패치 시퀀스)
        elif features.dim() == 3:
            pass

        else:
            raise ValueError(f"예상치 못한 Vision Encoder 출력 차원입니다: {features.shape}")

        # ctrl-o 내부 컨벤션에 따라 텐서 또는 dict로 리턴
        return features

    def get_transform(self):
        """이 함수가 반환하는 transform을 데이터셋 전처리에 사용합니다."""
        return self.image_transform