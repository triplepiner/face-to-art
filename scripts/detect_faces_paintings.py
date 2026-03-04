"""Detect faces in all paintings and add has_face / face_confidence columns to the metadata CSV."""

import csv
import os
from collections import Counter

import mediapipe as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_DIR, "data")
_PORTRAITS_DIR = os.path.join(_DATA_DIR, "portraits")
_CSV_PATH = os.path.join(_DATA_DIR, "portraits_metadata.csv")
_MODEL_PATH = os.path.join(_DIR, "models", "blaze_face_short_range.tflite")

# Read existing metadata
with open(_CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

print(f"Loaded {len(rows)} paintings from metadata CSV")

# Initialize MediaPipe face detection (tasks API)
detector = mp.tasks.vision.FaceDetector.create_from_options(
    mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        min_detection_confidence=0.5,
    )
)

faces_found = 0
faces_missing = 0
style_counts: dict[str, Counter] = {}  # style -> Counter(has_face=T/F)

for row in tqdm(rows, desc="Detecting faces"):
    img_path = os.path.join(_PORTRAITS_DIR, row["filename"])
    try:
        img = Image.open(img_path).convert("RGB")
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
        results = detector.detect(mp_img)
    except Exception as e:
        print(f"  Error processing {row['filename']}: {e}")
        row["has_face"] = "False"
        row["face_confidence"] = "0.0"
        faces_missing += 1
        continue

    if results.detections:
        best = max(results.detections, key=lambda d: d.categories[0].score)
        confidence = float(best.categories[0].score)
        row["has_face"] = "True"
        row["face_confidence"] = f"{confidence:.4f}"
        faces_found += 1
    else:
        row["has_face"] = "False"
        row["face_confidence"] = "0.0"
        faces_missing += 1

    # Track by style
    style = row.get("style", "Unknown") or "Unknown"
    if style not in style_counts:
        style_counts[style] = Counter()
    style_counts[style][row["has_face"]] += 1

detector.close()

# Write updated CSV
if "has_face" not in fieldnames:
    fieldnames.append("has_face")
if "face_confidence" not in fieldnames:
    fieldnames.append("face_confidence")

with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Summary
print(f"\n{'='*60}")
print(f"FACE DETECTION SUMMARY")
print(f"{'='*60}")
print(f"Total paintings:       {len(rows)}")
print(f"With detected face:    {faces_found} ({100*faces_found/len(rows):.1f}%)")
print(f"Without detected face: {faces_missing} ({100*faces_missing/len(rows):.1f}%)")
print(f"\nBreakdown by style:")
print(f"{'Style':<30} {'Total':>6} {'Face':>6} {'No Face':>8} {'% Face':>8}")
print(f"{'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
for style in sorted(style_counts.keys()):
    counts = style_counts[style]
    total = counts["True"] + counts["False"]
    with_face = counts["True"]
    without_face = counts["False"]
    pct = 100 * with_face / total if total > 0 else 0
    print(f"{style:<30} {total:>6} {with_face:>6} {without_face:>8} {pct:>7.1f}%")
