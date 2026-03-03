"""
Shared matching engine for Face-to-Art.

Loads two embedding models (vanilla CLIP for vibe, FaRL for face),
precomputed painting embeddings, and metadata. Provides find_matches()
which crops a selfie, encodes it through both models, z-score normalizes,
blends, and returns the top-K most similar paintings.

Imported by both the Gradio app and Telegram bot.
"""

import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, "data")
_PORTRAITS_DIR = os.path.join(_DATA_DIR, "portraits")
_CSV_PATH = os.path.join(_DATA_DIR, "portraits_metadata.csv")
_CLIP_EMB_PATH = os.path.join(_DATA_DIR, "clip_embeddings.npy")
_FARL_EMB_PATH = os.path.join(_DATA_DIR, "farl_embeddings.npy")
_FARL_WEIGHTS_PATH = os.path.join(_DIR, "models", "FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_env_device = os.environ.get("DEVICE", "").lower()
if _env_device:
    DEVICE = torch.device(_env_device)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Load vanilla CLIP ViT-B/32 (vibe engine)
# ---------------------------------------------------------------------------
import open_clip as _open_clip

clip_model, _, clip_preprocess = _open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Load FaRL ViT-B/16 (face engine)
# ---------------------------------------------------------------------------
import clip as _clip

farl_model, farl_preprocess = _clip.load("ViT-B/16", device="cpu")
_farl_state = torch.load(_FARL_WEIGHTS_PATH, map_location="cpu", weights_only=False)
farl_model.load_state_dict(_farl_state["state_dict"], strict=False)
del _farl_state
farl_model = farl_model.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Load precomputed embeddings & metadata
# ---------------------------------------------------------------------------
clip_embeddings: np.ndarray = np.load(_CLIP_EMB_PATH)   # (N, 512)
farl_embeddings: np.ndarray = np.load(_FARL_EMB_PATH)   # (N, 512)

metadata: list[dict] = []
with open(_CSV_PATH, newline="", encoding="utf-8") as _f:
    for _row in csv.DictReader(_f):
        metadata.append(_row)

assert len(metadata) == clip_embeddings.shape[0] == farl_embeddings.shape[0]

print(f"Matcher loaded: {len(metadata)} paintings, device={DEVICE}")

# ---------------------------------------------------------------------------
# Haar cascade for face detection
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


# ---------------------------------------------------------------------------
# Face cropping
# ---------------------------------------------------------------------------
def crop_face(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Detect the largest face and return (tight_crop, loose_crop).

    tight_crop: ~15% padding around the face box  (for FaRL)
    loose_crop: ~40% padding including surroundings (for CLIP vibe)

    If no face is found, both crops are the original image.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return image, image

    # Pick the largest face by area
    areas = [w * h for (_, _, w, h) in faces]
    x, y, w, h = faces[np.argmax(areas)]

    img_h, img_w = img_array.shape[:2]

    def _padded_crop(pad_frac: float) -> Image.Image:
        pad_x = int(w * pad_frac)
        pad_y = int(h * pad_frac)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)
        return image.crop((x1, y1, x2, y2))

    tight = _padded_crop(0.15)
    loose = _padded_crop(0.40)
    return tight, loose


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _encode_clip(img: Image.Image) -> np.ndarray:
    """Encode a PIL image through vanilla CLIP -> L2-normalized 512-d vector."""
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
    emb = emb.cpu().numpy().astype(np.float32).squeeze()
    emb /= np.linalg.norm(emb) + 1e-10
    return emb


def _encode_farl(img: Image.Image) -> np.ndarray:
    """Encode a PIL image through FaRL -> L2-normalized 512-d vector."""
    tensor = farl_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = farl_model.encode_image(tensor)
    emb = emb.cpu().numpy().astype(np.float32).squeeze()
    emb /= np.linalg.norm(emb) + 1e-10
    return emb


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------
def find_matches(
    image: Image.Image,
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[dict]:
    """Find the top-K painting matches for a selfie.

    Args:
        image:  PIL RGB image (a selfie / photo of a face).
        alpha:  Blend weight.  0 = pure CLIP vibe, 1 = pure FaRL face.
        top_k:  Number of results to return.

    Returns:
        List of dicts, each with all CSV metadata fields plus:
            blended_score, clip_score, farl_score, painting_index, painting_path
    """
    image = image.convert("RGB")

    # A — crop
    tight_crop, loose_crop = crop_face(image)

    # B — encode loose crop through vanilla CLIP
    clip_vec = _encode_clip(loose_crop)

    # C — encode tight crop through FaRL
    farl_vec = _encode_farl(tight_crop)

    # D — cosine similarities (embeddings are already L2-normed)
    clip_sims = clip_embeddings @ clip_vec    # (N,)
    farl_sims = farl_embeddings @ farl_vec    # (N,)

    # E — z-score normalize before blending
    clip_z = (clip_sims - clip_sims.mean()) / (clip_sims.std() + 1e-8)
    farl_z = (farl_sims - farl_sims.mean()) / (farl_sims.std() + 1e-8)

    # F — blend
    final_scores = alpha * farl_z + (1 - alpha) * clip_z

    # G — top-K
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    # H — build results
    results = []
    for idx in top_indices:
        row = dict(metadata[idx])
        row["blended_score"] = float(final_scores[idx])
        row["clip_score"] = float(clip_sims[idx])
        row["farl_score"] = float(farl_sims[idx])
        row["painting_index"] = int(idx)
        row["painting_path"] = os.path.join(_PORTRAITS_DIR, metadata[idx]["filename"])
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_painting_image(index: int) -> Image.Image:
    """Load and return a painting by its metadata index."""
    path = os.path.join(_PORTRAITS_DIR, metadata[index]["filename"])
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    random.seed(7)
    test_idx = random.randint(0, len(metadata) - 1)
    test_img = get_painting_image(test_idx)
    test_title = metadata[test_idx].get("title", "Untitled")
    test_artist = metadata[test_idx].get("artist", "Unknown")
    print(f'\nSelf-test using painting [{test_idx}] "{test_title}" by {test_artist}')

    for alpha, label in [(0.0, "pure CLIP vibe"), (0.5, "50/50 blend"), (1.0, "pure FaRL face")]:
        matches = find_matches(test_img, alpha=alpha, top_k=3)
        print(f"\n  alpha={alpha:.1f} ({label}):")
        for rank, m in enumerate(matches, 1):
            print(
                f'    {rank}. "{m["title"]}" by {m["artist"]}  '
                f'(blend={m["blended_score"]:.3f}, '
                f'clip={m["clip_score"]:.4f}, farl={m["farl_score"]:.4f})'
            )
