import os
import random
import pickle
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import lpips
import pyiqa
from PIL import Image  # (currently unused)
from torchvision import transforms  # (currently unused)

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from reward_loss import DWTLoss, CLIPLoss


CHECKPOINT_PATH = "./FaceRewardModel.pth"


def load_inference_function(reward_config):
    """
    Load required models according to the given configuration and return an
    inference function for computing a composite reward score.

    This function is intended to be called once at service startup.

    Args:
        reward_config (dict): A mapping from metric name to its weight.

    Returns:
        function: A configured function that computes the composite reward.
    """
    # 1) --- Weight validation ---
    # Ensure the sum of selected metric weights equals 1.0.
    total_weight = sum(reward_config.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"The sum of weights must be 1.0, but got: {total_weight}")

    print("Weight configuration validated.")
    print("Selected metrics and weights:", reward_config)

    # 2) --- Common settings and model container ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32
    models = {}  # Store all lazily loaded models in a dict.

    # 3) --- Load models on demand ---

    # a) Face Reward Model (if configured)
    if "face" in reward_config:
        print("Loading Face Reward model...")
        model_name = "ViT-H-14"
        reward_model, _, _ = create_model_and_transforms(
            model_name,
            "laion2B-s32B-b79K",
            precision=weight_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            reward_model.load_state_dict(checkpoint["state_dict"], strict=True)
            print("Face Reward model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading Face Reward model weights: {e}")
            raise

        reward_model = reward_model.to(device).eval()
        for param in reward_model.parameters():
            param.requires_grad = False

        models["face_model"] = reward_model
        models["tokenizer"] = get_tokenizer(model_name)
        models["face_transform"] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        print(f"Face Reward model is loaded on device: {device}.")

    # b) LPIPS model (if configured)
    if "lpips" in reward_config:
        print("Loading LPIPS model...")
        net_lpips = lpips.LPIPS(net="vgg").to(device).eval()
        net_lpips.requires_grad_(False)
        models["lpips_net"] = net_lpips
        print("LPIPS model loaded.")

    # c) SSIM metric (if configured)
    if "ssim" in reward_config:
        print("Loading SSIM (MS-SSIM) metric...")
        models["ssim_metric"] = pyiqa.create_metric("ms_ssim", device=device, as_loss=False)
        print("SSIM metric loaded.")

    # d) CLIPIQA metric (if configured)
    if "clipiqa" in reward_config:
        print("Loading CLIPIQA metric...")
        models["clipiqa_metric"] = pyiqa.create_metric("clipiqa", as_loss=False).to(device, dtype=weight_dtype)
        print("CLIPIQA metric loaded.")

    # e) DWT loss (if configured)
    if "dwt" in reward_config:
        print("Loading DWT loss...")
        models["dwt_loss"] = DWTLoss(loss_weight=1.0, lowq=True, highq=False).to(device, dtype=torch.float32)
        print("DWT loss loaded.")

    # f) CLIP score (if configured)  # Note: key kept as 'clip_socre' to preserve original behavior.
    if "clipscore" in reward_config:
        print("Loading CLIP Score model...")
        models["clip_socre"] = CLIPLoss(clip_loss_model="ViT-B-16-SigLIP-256").to(device, dtype=torch.float32)
        print("CLIP Score model loaded.")

    @torch.no_grad()
    def compute_comprehensive_reward(image_tensors, gt_image_tensors, captions):
        """
        Compute the composite reward score according to startup configuration.

        Args:
            image_tensors (torch.Tensor): Generated images, shape (B, C, H, W), range [-1, 1].
            gt_image_tensors (torch.Tensor): Ground truth images, shape (B, C, H, W), range [-1, 1].
            captions (list[str]): List of text captions corresponding to the images.

        Returns:
            dict: A dictionary containing the total score per sample and per-metric details.
        """
        batch_size = image_tensors.shape[0]
        total_scores = torch.zeros(batch_size, device=device, dtype=weight_dtype)
        results = {"details": {}}

        # Convert to [0, 1] range for models that expect it.
        tensors_0_1 = ((image_tensors.detach().to(device) + 1) / 2).clamp(0, 1)
        gt_tensors_0_1 = ((gt_image_tensors.detach().to(device) + 1) / 2).clamp(0, 1)

        # --- Compute and combine metrics as configured ---

        if "face" in reward_config:
            preprocessed = models["face_transform"](tensors_0_1).to(weight_dtype)
            tokens = models["tokenizer"](captions).to(device)
            outputs = models["face_model"](preprocessed, tokens)
            scores = torch.diagonal(outputs["image_features"] @ outputs["text_features"].T)
            # Note: score range is not normalized here; used as-is.
            results["details"]["face_scores"] = scores
            total_scores += reward_config["face"] * scores

        if "lpips" in reward_config:
            # LPIPS expects inputs in [-1, 1]
            dist = models["lpips_net"](image_tensors.to(device), gt_image_tensors.to(device)).squeeze()
            scores = 1.0 - dist  # Convert distance to similarity-like score
            results["details"]["lpips_scores"] = scores
            total_scores += reward_config["lpips"] * scores

        if "l2" in reward_config:
            # Use inputs in [-1, 1]
            loss = F.mse_loss(
                image_tensors.to(device),
                gt_image_tensors.to(device),
                reduction="none",
            ).mean(dim=[1, 2, 3])
            # Map loss to score via exponential decay
            scores = torch.exp(-15 * loss)
            results["details"]["l2_scores"] = scores
            total_scores += reward_config["l2"] * scores

        if "ssim" in reward_config:
            # pyiqa expects inputs in [0, 1]
            scores = models["ssim_metric"](tensors_0_1, gt_tensors_0_1)
            results["details"]["ssim_scores"] = scores
            total_scores += reward_config["ssim"] * scores

        if "clipiqa" in reward_config:
            # pyiqa expects inputs in [0, 1]
            scores = models["clipiqa_metric"](tensors_0_1).squeeze()
            results["details"]["clipiqa_scores"] = scores
            total_scores += reward_config["clipiqa"] * scores

        if "dwt" in reward_config:
            # DWT loss expects inputs in [0, 1]
            loss = models["dwt_loss"](tensors_0_1, gt_tensors_0_1)
            scores = torch.exp(-15 * loss)  # Map loss to score
            results["details"]["dwt_scores"] = scores
            total_scores += reward_config["dwt"] * scores

        if "clipscore" in reward_config:
            loss = models["clip_socre"](tensors_0_1, gt_tensors_0_1)
            scores = torch.exp(-15 * loss)
            results["details"]["clip_socre_scores"] = scores  # key kept as original
            total_scores += reward_config["clipscore"] * scores

        # --- Pack final results ---
        results["total_scores"] = total_scores.cpu().tolist()
        for key, tensor in results["details"].items():
            results["details"][key] = tensor.cpu().tolist()

        return results

    return compute_comprehensive_reward
