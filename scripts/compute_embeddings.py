"""
Compute two embedding vectors for every painting in the dataset:
  1. Vanilla CLIP ViT-B/32 (artistic style / vibe)
  2. FaRL ViT-B/16 (facial resemblance)

Outputs two .npy files in data/:
  - clip_embeddings.npy  — shape [N, 512]
  - farl_embeddings.npy  — shape [N, 512]

Row order matches the metadata CSV exactly.
"""

import csv
import gc
import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PORTRAITS_DIR = os.path.join(PROJECT_ROOT, "data", "portraits")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "portraits_metadata.csv")
FARL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
CLIP_OUT = os.path.join(PROJECT_ROOT, "data", "clip_embeddings.npy")
FARL_OUT = os.path.join(PROJECT_ROOT, "data", "farl_embeddings.npy")

BATCH_SIZE = 64


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_clip_embeddings(metadata, device):
    """Pass 1: Vanilla CLIP ViT-B/32 via open_clip."""
    import open_clip

    print("\n=== Pass 1: Vanilla CLIP ViT-B/32 ===")
    print(f"Device: {device}")

    model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    n = len(metadata)
    embeddings = np.zeros((n, 512), dtype=np.float32)
    missing = 0

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="CLIP ViT-B/32"):
        batch_rows = metadata[batch_start : batch_start + BATCH_SIZE]
        tensors = []
        indices = []

        for i, row in enumerate(batch_rows):
            idx = batch_start + i
            path = os.path.join(PORTRAITS_DIR, row["filename"])
            if not os.path.exists(path):
                print(f"  WARNING: missing file {row['filename']}, using zero vector")
                missing += 1
                continue
            img = Image.open(path).convert("RGB")
            tensors.append(clip_preprocess(img))
            indices.append(idx)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        feats = feats.cpu().numpy().astype(np.float32)

        for j, idx in enumerate(indices):
            embeddings[idx] = feats[j]

    if missing:
        print(f"  Total missing files: {missing}")

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero for missing files
    embeddings = embeddings / norms

    np.save(CLIP_OUT, embeddings)
    print(f"Saved CLIP embeddings: {CLIP_OUT} — shape {embeddings.shape}")

    # Unload model
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return embeddings


def compute_farl_embeddings(metadata, device):
    """Pass 2: FaRL ViT-B/16 via OpenAI clip."""
    import clip

    print("\n=== Pass 2: FaRL ViT-B/16 ===")
    print(f"Device: {device}")

    model, farl_preprocess = clip.load("ViT-B/16", device="cpu")
    farl_state = torch.load(FARL_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(farl_state["state_dict"], strict=False)
    model = model.to(device).eval()

    n = len(metadata)
    embeddings = np.zeros((n, 512), dtype=np.float32)
    missing = 0

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="FaRL ViT-B/16"):
        batch_rows = metadata[batch_start : batch_start + BATCH_SIZE]
        tensors = []
        indices = []

        for i, row in enumerate(batch_rows):
            idx = batch_start + i
            path = os.path.join(PORTRAITS_DIR, row["filename"])
            if not os.path.exists(path):
                print(f"  WARNING: missing file {row['filename']}, using zero vector")
                missing += 1
                continue
            img = Image.open(path).convert("RGB")
            tensors.append(farl_preprocess(img))
            indices.append(idx)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        feats = feats.cpu().numpy().astype(np.float32)

        for j, idx in enumerate(indices):
            embeddings[idx] = feats[j]

    if missing:
        print(f"  Total missing files: {missing}")

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    np.save(FARL_OUT, embeddings)
    print(f"Saved FaRL embeddings: {FARL_OUT} — shape {embeddings.shape}")

    # Unload model
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return embeddings


def sanity_checks(clip_emb, farl_emb, metadata):
    print("\n=== Sanity Checks ===")
    print(f"CLIP embeddings shape: {clip_emb.shape}")
    print(f"FaRL embeddings shape: {farl_emb.shape}")

    clip_norms = np.linalg.norm(clip_emb, axis=1)
    farl_norms = np.linalg.norm(farl_emb, axis=1)
    # Exclude zero vectors (missing files) from norm stats
    clip_valid = clip_norms[clip_norms > 0.5]
    farl_valid = farl_norms[farl_norms > 0.5]
    print(f"CLIP L2 norms — mean: {clip_valid.mean():.6f}, std: {farl_valid.std():.6f}")
    print(f"FaRL L2 norms — mean: {farl_valid.mean():.6f}, std: {farl_valid.std():.6f}")

    # Pick 3 random paintings for comparison
    n = len(metadata)
    random.seed(42)
    sample_indices = random.sample(range(n), 3)

    for idx in sample_indices:
        title = metadata[idx].get("title", "Untitled")
        artist = metadata[idx].get("artist", "Unknown")
        print(f"\n--- [{idx}] \"{title}\" by {artist} ---")

        clip_sims = clip_emb @ clip_emb[idx]
        farl_sims = farl_emb @ farl_emb[idx]
        blend_sims = 0.5 * clip_sims + 0.5 * farl_sims

        # Exclude self
        clip_sims[idx] = -1
        farl_sims[idx] = -1
        blend_sims[idx] = -1

        clip_top5 = np.argsort(clip_sims)[-5:][::-1]
        farl_top5 = np.argsort(farl_sims)[-5:][::-1]
        blend_top5 = np.argsort(blend_sims)[-5:][::-1]

        print("  CLIP top-5 (style/vibe):")
        for rank, j in enumerate(clip_top5, 1):
            t = metadata[j].get("title", "Untitled")
            a = metadata[j].get("artist", "Unknown")
            print(f"    {rank}. [{j}] \"{t}\" by {a}  (sim={clip_sims[j]:.4f})")

        print("  FaRL top-5 (face):")
        for rank, j in enumerate(farl_top5, 1):
            t = metadata[j].get("title", "Untitled")
            a = metadata[j].get("artist", "Unknown")
            print(f"    {rank}. [{j}] \"{t}\" by {a}  (sim={farl_sims[j]:.4f})")

        print("  Blend top-5 (50/50):")
        for rank, j in enumerate(blend_top5, 1):
            t = metadata[j].get("title", "Untitled")
            a = metadata[j].get("artist", "Unknown")
            print(f"    {rank}. [{j}] \"{t}\" by {a}  (sim={blend_sims[j]:.4f})")

        # Show overlap
        clip_set = set(clip_top5.tolist())
        farl_set = set(farl_top5.tolist())
        overlap = clip_set & farl_set
        print(f"  CLIP vs FaRL top-5 overlap: {len(overlap)}/5")


def main():
    metadata = load_metadata()
    print(f"Loaded {len(metadata)} rows from metadata CSV")

    device = get_device()

    clip_emb = compute_clip_embeddings(metadata, device)
    farl_emb = compute_farl_embeddings(metadata, device)
    sanity_checks(clip_emb, farl_emb, metadata)


if __name__ == "__main__":
    main()
