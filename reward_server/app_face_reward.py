import os
import pickle
import traceback

import torch
from flask import Flask, request, Blueprint

from reward_server.reward_inference import load_inference_function


root = Blueprint("root", __name__)

# This is the recommended (best) weighting configuration for the reward.
REWARD_CONFIG = {
    "face": 0.3,
    "lpips": 0.3,
    "clipiqa": 0.1,
    "dwt": 0.3,
}

# Alternative config example:
# REWARD_CONFIG = {
#     "face": 0.4,
#     "lpips": 0.2,
#     "clipiqa": 0.2,
#     "dwt": 0.2,
# }

# Global variable to store the loaded inference function.
INFERENCE_FN = None


def create_app(REWARD_CONFIG: dict = REWARD_CONFIG) -> Flask:
    """
    Create and configure the Flask application.

    This function will be called by Gunicorn or when running directly.
    It loads the model once at startup and stores an inference function globally.
    """
    global INFERENCE_FN

    # Load the model and get the inference function at app startup.
    print("Loading Face Reward Model...")
    INFERENCE_FN = load_inference_function(REWARD_CONFIG)
    print("Model loaded. Server is ready.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/", methods=["POST"])
def inference():
    """
    Handle POST requests, run model inference, and return results.

    Expected pickled request payload (bytes) is a dict with:
      - "image_tensors": torch.Tensor
      - "gt_image_tensors": torch.Tensor
      - "captions": any (e.g., list[str])
    """
    print(f"Received POST request from {request.remote_addr}.")
    raw_data = request.get_data()

    try:
        # 1) Deserialize the received data.
        data = pickle.loads(raw_data)
        image_tensors = data["image_tensors"]
        captions = data["captions"]
        gt_image_tensors = data["gt_image_tensors"]

        # 2) Validate types.
        if not isinstance(image_tensors, torch.Tensor) or not isinstance(gt_image_tensors, torch.Tensor):
            raise TypeError("Expected both 'image_tensors' and 'gt_image_tensors' to be torch.Tensor.")

        # 3) Call the already-loaded inference function with three arguments.
        results = INFERENCE_FN(image_tensors, gt_image_tensors, captions)

        # 4) The combined reward result is a dict; serialize it directly.
        response_bytes = pickle.dumps(results)
        returncode = 200

    except Exception:
        # Error handling: return the traceback text.
        error_message = traceback.format_exc()
        print(error_message)
        response_bytes = error_message.encode("utf-8")
        returncode = 500

    return response_bytes, returncode


# For local debugging only. Use Gunicorn in production.
if __name__ == "__main__":
    print("Warning: Running Flask app in debug mode.")
    print('For production, start with: gunicorn "app_face_reward:create_app()"')
    app = create_app(REWARD_CONFIG)
    app.run(host="127.0.0.1", port=18085)
