from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

import requests
import pickle
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def face_reward_remote(device):
    """
    Create a callable that sends a whole batch of images + captions to the
    self-hosted Face Reward server via HTTP and returns the reward scores.

    Note:
        Ensure the URL/port matches the server you launched.

    Args:
        device: Reserved argument for API compatibility (not used here).

    Returns:
        Callable(images, prompts, metadata) -> (scores_dict, empty_metadata_dict)
    """
    # TODO: keep this in sync with your running server address
    url = "http://127.0.0.1:18085"

    # Configure a Session with retry logic to improve network robustness.
    sess = requests.Session()
    # sess.proxies = {"http": None, "https": None}
    retries = Retry(
        total=5,                  # reasonable retry count
        backoff_factor=0.5,       # exponential backoff factor
        status_forcelist=[500, 502, 503, 504],  # retry on these server errors
        allowed_methods=frozenset(["POST"]),
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        """
        Inner function that performs the actual call to the reward server and
        parses the dynamic scores from the response.

        Args:
            images (torch.Tensor): Generated images in [-1, 1], shape (B, C, H, W).
            prompts (List[str]): Text captions corresponding to images.
            metadata (dict): Should contain reference images under key "gt_images".

        Returns:
            Tuple[dict, dict]: (all_scores_dict, empty_metadata_dict)
                - all_scores_dict includes "total_scores" and all per-metric scores.
        """
        try:
            # --- Retrieve ground-truth images from metadata ---
            if metadata is None or "gt_images" not in metadata:
                raise ValueError("Missing required 'gt_images' in metadata.")

            gt_images = metadata["gt_images"]

            if not isinstance(images, torch.Tensor) or not isinstance(gt_images, torch.Tensor):
                raise TypeError("Expected both 'images' and 'gt_images' to be torch.Tensor.")

            # --- Pack the request payload (must match the server's expected schema) ---
            data = {
                "image_tensors": images.cpu(),
                "gt_image_tensors": gt_images.cpu(),
                "captions": prompts,
            }

            data_bytes = pickle.dumps(data)

            # Send POST request to the composite reward server.
            response = sess.post(url, data=data_bytes, timeout=120)
            response.raise_for_status()  # raise for any 4xx/5xx

            # Parse the server response (pickled dict).
            response_data = pickle.loads(response.content)

            # --- Dynamically parse returned scores ---
            all_scores = {}

            # 1) Total scores (required)
            if "total_scores" in response_data:
                all_scores["total_scores"] = response_data["total_scores"]
            else:
                raise ValueError("Key 'total_scores' not found in server response.")

            # 2) Per-metric details (optional, under 'details' dict)
            details = response_data.get("details", {})
            if isinstance(details, dict):
                all_scores.update(details)

            # 3) Return all scores and an empty metadata dict
            return all_scores, {}

        except Exception as e:
            # Strengthened error logging for easier debugging
            print(f"Fatal error in reward function _fn: {e}")
            # In training, you may choose to return a default penalty instead of raising:
            # return {"total_scores": [-10.0] * len(prompts)}, {}
            raise

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        'face_reward_remote': face_reward_remote  # This now points to the client _fn
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        # This part correctly initializes the scoring functions
        # The check for 'device' handles functions that need it
        if 'device' in score_functions[score_name].__code__.co_varnames:
            score_fns[score_name] = score_functions[score_name](device)
        else:
            score_fns[score_name] = score_functions[score_name]()

    # only_strict is only for geneval.
    def _fn(images, prompts, metadata, only_strict=True):
        # This will be the main dictionary holding all scores
        score_details = {}
        # This list will hold the weighted average score across all metrics
        total_scores = []

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images,
                                                                                                             prompts,
                                                                                                             metadata,
                                                                                                             only_strict)
                # The primary score for weighting is the main 'scores'
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            else:
                # --- START: CORE MODIFICATION ---
                # This branch is now aligned with your remote reward server logic.

                # 1. Call the remote server function. It returns a dictionary of all scores.
                all_scores_dict, _ = score_fns[score_name](images, prompts, metadata)

                # 2. The primary score for weighting is the 'total_scores' from the remote server.
                scores = all_scores_dict.get('total_scores')
                if scores is None:
                    raise ValueError(f"The reward server for '{score_name}' did not return a 'total_scores' key.")

                # 3. Add all scores (total and detailed) from the server to our main details dictionary.
                score_details.update(all_scores_dict)
                # --- END: CORE MODIFICATION ---

            # For consistency, store the primary score for this metric under its name
            score_details[score_name] = scores

            # Calculate the weighted score for this metric
            weighted_scores = [weight * score for score in scores]

            # Aggregate the weighted scores to get the final average
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        # Add the final weighted average to the results
        score_details['avg'] = total_scores

        # Return the comprehensive dictionary. This is cleaner than returning many loose variables.
        return score_details, {}

    return _fn