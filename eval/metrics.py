"""Quality metrics for scripts/benchmark.py.

Compares TaylorSeer-accelerated outputs against a no-cache baseline:
- clip_score / image_reward: reference-free prompt-image alignment (need a text prompt).
- psnr / ssim / lpips: pairwise similarity between accelerated and baseline frames
  from the same seed (higher psnr/ssim and lower lpips means less drift from baseline).
- fid: distributional distance between the pooled baseline and accelerated frames.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as tvf
from PIL import Image

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
_IMAGE_REWARD_MODEL_NAME = "ImageReward-v1.0"


@dataclass
class QualityMetrics:
    """Averaged quality metrics for one benchmark run; `None` if not computed."""

    clip_score: float | None = None
    image_reward: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    fid: float | None = None


def _to_tensor_batch(images: Sequence[Image.Image]) -> torch.Tensor:
    """Stack PIL images into a [N, 3, H, W] float tensor in [0, 1]."""
    return torch.stack([tvf.to_tensor(image.convert("RGB")) for image in images])


def compute_clip_score(images: Sequence[Image.Image], prompts: Sequence[str], device: str) -> float:
    """Average CLIP cosine similarity (0-100) between each image and its prompt.

    Implemented directly against `transformers.CLIPModel` rather than
    `torchmetrics.multimodal.CLIPScore`: the latter's internal feature extraction
    assumes `get_image_features`/`get_text_features` return a bare tensor, which
    breaks on newer `transformers` releases that wrap the result in a
    `BaseModelOutputWithPooling`.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)

    inputs = processor(
        images=[image.convert("RGB") for image in images],
        text=list(prompts),
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    image_features = getattr(image_features, "pooler_output", image_features)
    text_features = getattr(text_features, "pooler_output", text_features)

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    score = 100 * (image_features * text_features).sum(dim=-1).clamp(min=0)
    return score.mean().item()


def _patch_for_image_reward() -> None:
    """Bridge `image-reward`'s dependency chain to a modern `transformers`/`setuptools`.

    `image-reward`'s vendored BLIP model imports `apply_chunking_to_forward`,
    `prune_linear_layer`, and `find_pruneable_heads_and_indices` from
    `transformers.modeling_utils`; newer `transformers` moved the first two to
    `pytorch_utils` and dropped the third entirely. Separately, its `clip`
    dependency does `from pkg_resources import packaging`, which no longer ships
    with `setuptools>=81`. Both are patched here rather than pinning older
    `transformers`/`setuptools`, since this project's diffusers adapters need the
    current ones. No-ops wherever the real module already provides the symbol.
    """
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
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def compute_image_reward(images: Sequence[Image.Image], prompts: Sequence[str], device: str) -> float:
    """Average ImageReward score (higher is better) between each image and its prompt."""
    _patch_for_image_reward()
    import ImageReward as image_reward

    model = image_reward.load(_IMAGE_REWARD_MODEL_NAME, device=device)
    scores = [model.score(prompt, image.convert("RGB")) for image, prompt in zip(prompts, images)]
    return sum(scores) / len(scores)


def compute_similarity_metrics(
    accelerated: Sequence[Image.Image], baseline: Sequence[Image.Image]
) -> tuple[float, float, float]:
    """Average PSNR, SSIM, and LPIPS between paired accelerated/baseline frames."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    accelerated_batch = _to_tensor_batch(accelerated)
    baseline_batch = _to_tensor_batch(baseline)
    if accelerated_batch.shape != baseline_batch.shape:
        raise ValueError(
            f"Accelerated frames {tuple(accelerated_batch.shape)} and baseline frames "
            f"{tuple(baseline_batch.shape)} must have the same shape to compare pairwise."
        )

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    return (
        psnr(accelerated_batch, baseline_batch).item(),
        ssim(accelerated_batch, baseline_batch).item(),
        lpips(accelerated_batch, baseline_batch).item(),
    )


def compute_fid(accelerated: Sequence[Image.Image], baseline: Sequence[Image.Image]) -> float:
    """FID between the pooled accelerated and baseline frames.

    FID is a distributional metric designed for tens of thousands of samples; with
    only a handful of prompts/frames (as in a quick benchmark run) this number is
    noisy — read it as a rough magnitude, not a precise value.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(_to_tensor_batch(baseline), real=True)
    fid.update(_to_tensor_batch(accelerated), real=False)
    return fid.compute().item()


def compute_quality_metrics(
    accelerated: Sequence[Image.Image],
    baseline: Sequence[Image.Image],
    prompts: Sequence[str] | None,
    device: str,
) -> QualityMetrics:
    """Compute every metric that can be computed; report and skip ones that fail."""
    metrics = QualityMetrics()

    if prompts is not None:
        try:
            metrics.clip_score = compute_clip_score(accelerated, prompts, device)
        except Exception as error:
            print(f"[metrics] skipping CLIP score: {error}")
        try:
            metrics.image_reward = compute_image_reward(accelerated, prompts, device)
        except Exception as error:
            print(f"[metrics] skipping ImageReward: {error}")

    try:
        metrics.psnr, metrics.ssim, metrics.lpips = compute_similarity_metrics(accelerated, baseline)
    except Exception as error:
        print(f"[metrics] skipping PSNR/SSIM/LPIPS: {error}")

    try:
        metrics.fid = compute_fid(accelerated, baseline)
    except Exception as error:
        print(f"[metrics] skipping FID: {error}")

    return metrics
