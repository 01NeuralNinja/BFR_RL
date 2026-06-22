from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import os
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_gaussian_noise_grpo,
    random_add_jpg_compression,
    random_add_jpg_compression_grpo
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset_grpo(data.Dataset):

    def __init__(
            self,
            file_list: str,
            file_backend_cfg: Mapping[str, Any],
            out_size: int,
            crop_type: str,
            blur_kernel_size: int,
            kernel_list: Sequence[str],
            kernel_prob: Sequence[float],
            blur_sigma: Sequence[float],
            downsample_range: Sequence[float],
            noise_range: Sequence[float],
            jpeg_range: Sequence[int],
    ) -> "CodeformerDataset_grpo":
        super(CodeformerDataset_grpo, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

        # Cache the per-sample degradation parameters.
        self.degradation_cache = {}

    def set_epoch(self, epoch: int):
        """
        Call at the start of each new epoch to clear the cache.
        This ensures a fresh set of random degradations every epoch.
        """
        self.degradation_cache.clear()

    def load_gt_image(self, image_path: str, max_retry: int = 5):
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # Check whether the degradation parameters for this index are cached.
        if index in self.degradation_cache:
            # If cached, reuse the stored parameters directly.
            degradation_params = self.degradation_cache[index]
            gt_path = degradation_params['gt_path']
            prompt = degradation_params['prompt']
            caption = degradation_params['caption']
        else:
            # Not cached (first time this index is seen this epoch): generate and cache.
            degradation_params = {}
            img_gt_temp = None
            original_index = index
            while img_gt_temp is None:
                image_file = self.image_files[index]
                gt_path = image_file["image_path"]
                prompt = image_file["prompt"]
                img_gt_temp = self.load_gt_image(gt_path)
                if img_gt_temp is None:
                    print(f"filed to load {gt_path}, try another image")
                    index = random.randint(0, len(self) - 1)

            # Cache under the original index even if loading fell back to another image.
            index = original_index

            caption_path = gt_path.replace('.png', '.txt')
            if not os.path.exists(caption_path):
                raise FileNotFoundError(f"Caption file does not exist: {caption_path}")
            with open(caption_path, 'r') as f:
                caption = f.read().strip()

            # Generate all random degradation parameters.
            degradation_params['gt_path'] = gt_path
            degradation_params['prompt'] = prompt
            degradation_params['caption'] = caption
            degradation_params['use_empty_prompt'] = (np.random.uniform() < 0.5)
            degradation_params['kernel'] = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, self.blur_kernel_size,
                self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], noise_range=None
            )
            degradation_params['scale'] = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            if self.noise_range is not None:
                degradation_params['noise_sigma'] = np.random.uniform(self.noise_range[0], self.noise_range[1])
            if self.jpeg_range is not None:
                degradation_params['jpeg_quality'] = np.random.randint(self.jpeg_range[0], self.jpeg_range[1])

            # Store the generated parameters in the cache.
            self.degradation_cache[index] = degradation_params

        # --- Process the image with the fixed parameters ---

        # Load the GT image.
        img_gt = self.load_gt_image(gt_path)
        # Simple fallback if loading fails for some reason.
        if img_gt is None:
            # Should have been handled earlier, but retry for safety.
            # Return the first sample to avoid crashing.
            print(f"ERROR: Failed to load {gt_path} during processing step. Returning a fallback item.")
            return self.__getitem__(0)

        # Use the prompt from the cached parameters.
        prompt = degradation_params['prompt']
        caption = degradation_params['caption']
        if degradation_params['use_empty_prompt']:
            prompt = ""

        img_gt_float = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt_float.shape

        # Apply degradations using the cached parameters.
        # 1. blur
        img_lq = cv2.filter2D(img_gt_float, -1, degradation_params['kernel'])
        # 2. downsample
        img_lq = cv2.resize(img_lq, (int(w // degradation_params['scale']), int(h // degradation_params['scale'])),
                            interpolation=cv2.INTER_LINEAR)
        # 3. noise
        if self.noise_range is not None and 'noise_sigma' in degradation_params:
            img_lq = random_add_gaussian_noise_grpo(img_lq, sigma_range=None,
                                               sigma=degradation_params['noise_sigma'])  # pass fixed sigma
        # 4. jpeg compression
        if self.jpeg_range is not None and 'jpeg_quality' in degradation_params:
            img_lq = random_add_jpg_compression_grpo(img_lq, quality_range=None,
                                                quality=degradation_params['jpeg_quality'])  # pass fixed quality

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # format conversion
        gt = (img_gt_float[..., ::-1] * 2 - 1).astype(np.float32)
        lq = img_lq[..., ::-1].astype(np.float32)

        return gt, lq, prompt, caption

    def __len__(self) -> int:
        return len(self.image_files)

