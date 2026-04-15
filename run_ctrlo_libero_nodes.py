import os
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from llm2vec import LLM2Vec

from ocl.cli import train


# =========================================================
# 사용자 설정
# =========================================================
CHECKPOINT_PATH = "pretrained_models/ctrlo/pretrained_model.ckpt"
CONFIG_PATH = "pretrained_models/ctrlo/config.yaml"

# step_0000.png 들이 들어 있는 루트 폴더
IMAGE_ROOT = "/media/aivs-7/새 볼륨/es/openvla/openvla/libero_observation_image_only/libero_10_no_noops/episode_000000"

# 결과 저장 폴더
OUTPUT_ROOT = "./ctrlo_libero_outputs"

# 네가 준 instruction + nodes 정보 파일
NODES_JSON_PATH = "/media/aivs-7/새 볼륨/es/openvla/openvla/parser/libero_10_parser.json"

# 이미지당 최대 7개 phrase
MAX_PROMPTS = 7

# 보통 gripper는 잘 안 보이므로 기본 제외
EXCLUDE_GRIPPER = True

# 이미지 탐색 확장자
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 네가 준 JSON 예시를 파일로 저장해서 쓰면 됨
# =========================================================
# libero_nodes.json 예시:
# [
#   {
#     "instruction": "put both the alphabet soup and the tomato sauce in the basket",
#     "nodes": ["alphabet soup", "tomato sauce", "basket", "gripper"],
#     "edges": ...
#   },
#   ...
# ]


# =========================================================
# LLM2Vec / CTRL-O 로더
# =========================================================
print(f"[INFO] device = {DEVICE}")

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map=DEVICE,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)


class CtrloFeatureExtractor:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

        oclf_config = OmegaConf.load(config_path)
        self.model = train.build_model_from_config(
            oclf_config, checkpoint_path
        ).to(self.device)
        self.model.eval()

        self.base_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract_features_batch(self, images: List[Image.Image], prompts: List[List[str]]):
        images_t = torch.stack([self.base_transform(image) for image in images]).to(self.device)

        # prompts: List[List[str]] with shape [B, 7]
        name_embeddings = torch.stack([l2v.encode(prompt) for prompt in prompts]).to(self.device)

        bsz = images_t.shape[0]
        inputs = {
            "image": images_t,
            "bbox_centroids": torch.tensor([[-1, -1]] * MAX_PROMPTS, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
                .to(self.device),
            "contrastive_loss_mask": torch.stack([
                torch.tensor([int(p != "other") for p in prompt], dtype=torch.long)
                for prompt in prompts
            ]).to(self.device),
            "name_embedding": name_embeddings,
            "instance_bbox": torch.tensor(
                [[-1, -1, -1, -1]] * MAX_PROMPTS, dtype=torch.float32
            ).unsqueeze(0).repeat(bsz, 1, 1).to(self.device),
            "batch_size": bsz,
        }

        outputs = self.model(inputs)
        return outputs


# =========================================================
# 유틸
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 180) -> str:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in str(name))
    safe = safe.strip().replace(" ", "_")
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe


def load_nodes_db(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def build_instruction_to_nodes(nodes_db: List[Dict]) -> Dict[str, List[str]]:
    mapping = {}
    for item in nodes_db:
        instruction = normalize_text(item["instruction"])
        nodes = item["nodes"]

        if EXCLUDE_GRIPPER:
            nodes = [n for n in nodes if normalize_text(n) != "gripper"]

        mapping[instruction] = nodes
    return mapping


def pad_prompts(nodes: List[str], max_prompts: int = MAX_PROMPTS) -> List[str]:
    nodes = nodes[:max_prompts]
    if len(nodes) < max_prompts:
        nodes = nodes + ["other"] * (max_prompts - len(nodes))
    return nodes


def find_images(root: str) -> List[str]:
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in IMAGE_EXTS:
                image_paths.append(os.path.join(dirpath, fn))
    return sorted(image_paths)


def guess_instruction_from_path(image_path: str, instruction_to_nodes: Dict[str, List[str]]) -> Optional[str]:
    """
    경로 안에 instruction 문자열이 포함되어 있다고 가정하고 찾음.
    예:
      .../put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet/episode_000000/step_0000.png
    """
    path_norm = normalize_text(image_path.replace("_", " ").replace(os.sep, " "))

    best_match = None
    best_len = -1
    for instr in instruction_to_nodes.keys():
        if instr in path_norm and len(instr) > best_len:
            best_match = instr
            best_len = len(instr)

    return best_match


def save_overlay_grid(
    image: Image.Image,
    prompt_list: List[str],
    masks_as_image: torch.Tensor,
    save_path: str,
):
    """
    이미지 1장 + 7개 mask overlay 저장
    """
    ensure_dir(os.path.dirname(save_path))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    colors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0.75, 0.75, 0.75),
    ]

    fig, axes = plt.subplots(1, 8, figsize=(24, 4))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    axes[0].imshow(cv2.resize(image_np, (768, 768), interpolation=cv2.INTER_NEAREST))
    axes[0].axis("off")
    axes[0].set_title("image", fontsize=16, pad=8)

    image_shape = image_tensor.shape[1:]
    masks = masks_as_image.view(-1, 1, *image_shape)

    for j in range(min(masks.shape[0], MAX_PROMPTS)):
        curr_mask = masks[j].squeeze(0).detach().cpu().numpy()

        colored_mask = np.zeros((*curr_mask.shape, 3), dtype=np.float32)
        colored_mask[:, :, 0] = curr_mask * colors[j % len(colors)][0] * 1.5
        colored_mask[:, :, 1] = curr_mask * colors[j % len(colors)][1] * 1.5
        colored_mask[:, :, 2] = curr_mask * colors[j % len(colors)][2] * 1.5
        colored_mask = np.clip(colored_mask, 0, 1)

        alpha = 0.45
        blended = image_np * (1 - alpha) + colored_mask * alpha
        blended = cv2.resize(blended, (768, 768), interpolation=cv2.INTER_NEAREST)

        title = textwrap.fill(prompt_list[j], width=20) if prompt_list[j] != "other" else ""
        axes[j + 1].imshow(blended)
        axes[j + 1].set_title(title, fontsize=14, pad=8, fontweight="bold")
        axes[j + 1].axis("off")

    # plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.05, transparent=True)
    plt.savefig(
        save_path,
        bbox_inches='tight',
        dpi=300,
        pad_inches=0,
        transparent=False,  # ❗ 변경
        facecolor='white'  # ❗ 추가
    )
    plt.close(fig)


def save_raw_masks(
    masks_as_image: torch.Tensor,
    prompt_list: List[str],
    out_dir: str,
):
    """
    각 node별 grayscale mask도 따로 저장
    """
    ensure_dir(out_dir)

    masks = masks_as_image.detach().cpu().numpy()  # [7, H, W] 또는 비슷한 형태 가정
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    for idx, node in enumerate(prompt_list):
        if node == "other":
            continue
        mask = masks[idx]
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        node_name = sanitize_filename(node)
        Image.fromarray(mask_uint8).save(os.path.join(out_dir, f"{idx:02d}_{node_name}.png"))


# =========================================================
# 메인
# =========================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    nodes_db = load_nodes_db(NODES_JSON_PATH)
    instruction_to_nodes = build_instruction_to_nodes(nodes_db)

    image_paths = find_images(IMAGE_ROOT)
    print(f"[INFO] found {len(image_paths)} images")

    extractor = CtrloFeatureExtractor(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        device=DEVICE,
    )

    for idx, image_path in enumerate(image_paths):
        matched_instruction = guess_instruction_from_path(image_path, instruction_to_nodes)
        if matched_instruction is None:
            print(f"[SKIP] no instruction match: {image_path}")
            continue

        nodes = instruction_to_nodes[matched_instruction]
        prompt_list = pad_prompts(nodes, MAX_PROMPTS)

        rel_path = os.path.relpath(image_path, IMAGE_ROOT)
        stem = os.path.splitext(rel_path)[0]
        safe_stem = sanitize_filename(stem.replace(os.sep, "__"))

        image = Image.open(image_path).convert("RGB")
        outputs = extractor.extract_features_batch([image], [prompt_list])

        # CTRL-O inference.py 구조를 따라 masks_as_image 사용
        masks_as_image = outputs["object_decoder"].masks_as_image[0]

        sample_out_dir = os.path.join(OUTPUT_ROOT, safe_stem)
        ensure_dir(sample_out_dir)

        save_overlay_grid(
            image=image,
            prompt_list=prompt_list,
            masks_as_image=masks_as_image,
            save_path=os.path.join(sample_out_dir, "overlay.png"),
        )

        save_raw_masks(
            masks_as_image=masks_as_image,
            prompt_list=prompt_list,
            out_dir=os.path.join(sample_out_dir, "masks"),
        )

        with open(os.path.join(sample_out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_path": image_path,
                    "instruction": matched_instruction,
                    "nodes": nodes,
                    "prompts_used": prompt_list,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[DONE] {idx+1}/{len(image_paths)} -> {sample_out_dir}")


if __name__ == "__main__":
    main()