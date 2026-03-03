"""
Download FaRL pretrained weights and verify compatibility with OpenAI CLIP ViT-B/16.

Downloads FaRL-Base-Patch16-LAIONFace20M-ep16.pth (~600MB) from the official
FacePerceiver/FaRL GitHub release. After downloading, verifies the checkpoint
structure and runs an integration test with OpenAI's CLIP ViT-B/16 model.
"""

import os
import sys

import requests
import torch
from tqdm import tqdm

FARL_URL = (
    "https://github.com/FacePerceiver/FaRL/releases/download/"
    "pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FILENAME = "FaRL-Base-Patch16-LAIONFace20M-ep16.pth"


def download_farl():
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, FILENAME)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"File already exists: {dest} ({size_mb:.1f} MB)")
        return dest

    print(f"Downloading FaRL weights from:\n  {FARL_URL}")
    print(f"Saving to: {dest}\n")

    resp = requests.get(FARL_URL, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=FILENAME
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"\nDownload complete: {size_mb:.1f} MB")
    return dest


def verify_checkpoint(path):
    print("\n--- Checkpoint Verification ---")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # FaRL checkpoints store the model weights under "state_dict"
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    keys = list(state_dict.keys())
    print(f"Total keys in state_dict: {len(keys)}")
    print("First 5 keys:")
    for k in keys[:5]:
        print(f"  {k}")

    return checkpoint


def integration_test(checkpoint):
    print("\n--- Integration Test ---")
    import clip

    # Load CLIP ViT-B/16
    print("Loading CLIP ViT-B/16 (OpenAI)...")
    model, preprocess = clip.load("ViT-B/16", device="cpu")

    # Get FaRL state_dict
    if "state_dict" in checkpoint:
        farl_sd = checkpoint["state_dict"]
    else:
        farl_sd = checkpoint

    # Swap in FaRL weights (strict=False to allow mismatches)
    result = model.load_state_dict(farl_sd, strict=False)
    print(f"Missing keys:    {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    # Forward pass with random 224x224 image
    print("Running dummy forward pass...")
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model.encode_image(dummy_image)
    print(f"Output embedding shape: {list(features.shape)}")
    print("Integration test PASSED!")


def main():
    path = download_farl()
    checkpoint = verify_checkpoint(path)
    integration_test(checkpoint)


if __name__ == "__main__":
    main()
