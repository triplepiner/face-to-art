"""
Download portrait paintings from Artificio/WikiArt dataset.

Streams the dataset (~103K images) to avoid loading everything into memory.
Filters by genre == "portrait", resizes to 512x512, saves as JPEG quality 85.
Produces a CSV with human-readable metadata including painting titles.
"""

import csv
import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "portraits")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "portraits_metadata.csv")
MAX_IMAGES = 3000
TARGET_SIZE = (512, 512)
JPEG_QUALITY = 85


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Artificio/WikiArt dataset (streaming)...")
    dataset = load_dataset("Artificio/WikiArt", split="train", streaming=True)

    metadata_rows = []
    count = 0

    print(f"Filtering portraits and saving (max {MAX_IMAGES})...")
    progress = tqdm(dataset, desc="Scanning dataset", unit="img")

    for sample in progress:
        # Genre is already a human-readable string in this dataset
        if sample["genre"] != "portrait":
            continue

        count += 1
        filename = f"{count:05d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Convert and resize image
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(filepath, "JPEG", quality=JPEG_QUALITY)

        metadata_rows.append({
            "filename": filename,
            "title": sample.get("title", "Untitled"),
            "artist": sample.get("artist", "Unknown Artist"),
            "genre": sample.get("genre", "portrait"),
            "style": sample.get("style", "Unknown"),
        })

        progress.set_postfix(portraits=count)

        if count >= MAX_IMAGES:
            print(f"\nReached max of {MAX_IMAGES} portraits.")
            break

    progress.close()

    # Write CSV
    print(f"Writing metadata CSV ({len(metadata_rows)} rows)...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "title", "artist", "genre", "style"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Done! {count} portraits saved to {OUTPUT_DIR}")
    print(f"Metadata CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
