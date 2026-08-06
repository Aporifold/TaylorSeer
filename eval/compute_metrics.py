import sys
import types

import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms import functional as tvf


def to_tensor(images: list[Image.Image]) -> torch.Tensor:
    return torch.stack([tvf.to_tensor(image.convert("RGB")) for image in images])


@torch.inference_mode()
def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cuda",
) -> float:
    metric = PeakSignalNoiseRatio(
        data_range=1.0,
        reduction="elementwise_mean",
    ).to(device)
    score: torch.Tensor = metric(pred, target)
    return float(score.detach().cpu().item())


@torch.inference_mode()
def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cuda",
) -> float:
    metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        reduction="elementwise_mean",
    ).to(device)
    score: torch.Tensor = metric(pred, target)
    return float(score.detach().cpu().item())


@torch.inference_mode()
def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cuda",
) -> float:
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)
    score: torch.Tensor = metric(pred, target)
    return float(score.detach().cpu().item())


@torch.inference_mode()
def calculate_clip_score(
    pred: list[Image.Image],
    prompts: list[str],
    device: str = "cuda",
) -> float:
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    # CLIPScore's HF image processor always rescales by 1/255, so it expects
    # pixel values in [0, 255] rather than the [0, 1] range `to_tensor` gives.
    images = (to_tensor(pred) * 255).to(device)
    score: torch.Tensor = metric(images, prompts)
    return float(score.detach().cpu().item())


def _patch_for_image_reward() -> None:
    # `image-reward`'s vendored BLIP model imports these from
    # `transformers.modeling_utils`; newer `transformers` moved the first two to
    # `pytorch_utils` and dropped the third. Its `clip` dependency also does
    # `from pkg_resources import packaging`, which setuptools>=81 no longer ships.
    if "pkg_resources" not in sys.modules:
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            import packaging.version

            shim = types.ModuleType("pkg_resources")
            shim.packaging = packaging
            sys.modules["pkg_resources"] = shim

    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = (
            pytorch_utils.apply_chunking_to_forward
        )
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads, n_heads, head_size, already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )


@torch.inference_mode()
def calculate_image_reward(
    pred: list[Image.Image],
    prompts: list[str],
    device: str = "cuda",
) -> float:
    _patch_for_image_reward()
    import ImageReward as RM

    model = RM.load(name="ImageReward-v1.0", device=device)
    scores = [
        model.score(prompt, image.convert("RGB"))
        for image, prompt in zip(pred, prompts)
    ]
    return float(sum(scores) / len(scores))


@torch.inference_mode()
def calculate_fid(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cuda",
) -> float:
    metric = FrechetInceptionDistance(normalize=True).to(device)
    metric.update(target, real=True)
    metric.update(pred, real=False)
    score: torch.Tensor = metric.compute()
    return float(score.detach().cpu().item())


def compute_all_metrics(
    pred: list[Image.Image],
    target: list[Image.Image],
    prompts: list[str],
    device: str = "cuda",
):
    # transforms to tensors, ranging in [0, 1].
    pred_tensor = to_tensor(pred).to(device)
    target_tensor = to_tensor(target).to(device)

    # 1. compute PSNR, SSIM, LPIPS scores.
    psnr = calculate_psnr(pred_tensor, target_tensor, device)
    ssim = calculate_ssim(pred_tensor, target_tensor, device)
    lpips = calculate_lpips(pred_tensor, target_tensor, device)

    # 2. compute FID score.
    fid = calculate_fid(pred_tensor, target_tensor, device)

    # 3. compute CLIP score.
    clip_score = calculate_clip_score(pred, prompts, device)

    # 4. compute image reward score.
    image_reward_score = calculate_image_reward(pred, prompts, device)

    print("==================== METRICS ====================")
    print(f"[PSNR]: {psnr:.4f}")
    print(f"[SSIM]: {ssim:.4f}")
    print(f"[LPIPS]: {lpips:.4f}")
    print(f"[FID]: {fid:.4f}")
    print(f"[CLIP score]: {clip_score:.4f}")
    print(f"[ImageReward score]: {image_reward_score:.4f}")
    print("=================================================")

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "fid": fid,
        "clip_score": clip_score,
        "image_reward_score": image_reward_score,
    }
