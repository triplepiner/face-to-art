"""Benchmark 4 — Face Detection Coverage Analysis.

Runs MediaPipe face detection on every painting in the dataset,
reports coverage overall and by art movement/style, identifies
faceless famous paintings, and generates a bar chart.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mediapipe as mp

# ── paths ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PORTRAITS_DIR = os.path.join(DATA_DIR, "portraits")
CSV_PATH = os.path.join(DATA_DIR, "portraits_metadata.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "blaze_face_short_range.tflite")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── matplotlib styling ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#d1d5db",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

# ── MediaPipe face detector (same config as matcher.py) ──────────────
detector = mp.tasks.vision.FaceDetector.create_from_options(
    mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        min_detection_confidence=0.5,
    )
)


def detect_faces(img_path):
    """Run face detection on a painting. Returns (has_face, confidence, num_faces)."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        results = detector.detect(mp_img)

        if not results.detections:
            return False, 0.0, 0

        best_conf = max(d.categories[0].score for d in results.detections)
        return True, round(best_conf, 4), len(results.detections)
    except Exception as e:
        print(f"    WARNING: {img_path}: {e}")
        return False, 0.0, 0


def main():
    # ── Load metadata ─────────────────────────────────────────────────
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    has_face_col_exists = "has_face" in fieldnames
    print(f"Loaded {len(rows)} paintings from metadata CSV")
    print(f"  has_face column exists: {has_face_col_exists}")

    # ── Run face detection on every painting ──────────────────────────
    print(f"\nRunning face detection on {len(rows)} paintings...")
    records = []

    for i, row in enumerate(rows):
        img_path = os.path.join(PORTRAITS_DIR, row["filename"])
        has_face, confidence, num_faces = detect_faces(img_path)

        records.append({
            "index": i,
            "filename": row["filename"],
            "title": row.get("title", "Unknown"),
            "artist": row.get("artist", "Unknown"),
            "style": row.get("style", "Unknown"),
            "is_famous": row.get("is_famous", "").lower() == "true",
            "has_face": has_face,
            "confidence": confidence,
            "num_faces": num_faces,
        })

        # Update row for CSV export
        row["has_face"] = str(has_face)
        row["face_confidence"] = str(confidence)

        if (i + 1) % 100 == 0 or i == len(rows) - 1:
            done = i + 1
            faces_so_far = sum(1 for r in records if r["has_face"])
            print(f"  [{done}/{len(rows)}] {faces_so_far} faces detected so far")

    # ── Compute statistics ────────────────────────────────────────────
    total = len(records)
    with_face = sum(1 for r in records if r["has_face"])
    without_face = total - with_face
    pct = 100 * with_face / total if total else 0

    # By style
    style_stats = defaultdict(lambda: {"total": 0, "with_face": 0})
    for r in records:
        s = r["style"] or "Unknown"
        style_stats[s]["total"] += 1
        if r["has_face"]:
            style_stats[s]["with_face"] += 1

    style_table = []
    for style, stats in style_stats.items():
        rate = 100 * stats["with_face"] / stats["total"] if stats["total"] else 0
        style_table.append((style, stats["total"], stats["with_face"], rate))
    style_table.sort(key=lambda x: x[3])  # worst to best

    # Faceless famous paintings
    faceless_famous = [
        r for r in records if r["is_famous"] and not r["has_face"]
    ]

    # ── Generate markdown report ──────────────────────────────────────
    md_lines = [
        "# Face Detection Coverage Report\n",
        "## Overall Coverage\n",
        f"**{with_face}/{total}** paintings have detected faces (**{pct:.1f}%**)\n",
        f"- With face: {with_face}",
        f"- Without face: {without_face}\n",
        "## Detection Rate by Art Movement/Style\n",
        "| Style | Total | With Face | Detection Rate |",
        "|-------|------:|----------:|---------------:|",
    ]
    for style, cnt, faces, rate in style_table:
        md_lines.append(f"| {style} | {cnt} | {faces} | {rate:.1f}% |")

    md_lines.append("")
    md_lines.append(f"## Top {min(20, len(faceless_famous))} Faceless Famous Paintings\n")
    if faceless_famous:
        md_lines.append("These famous paintings have no detected face — they may produce unexpected FaRL matches.\n")
        md_lines.append("| # | Title | Artist | Style |")
        md_lines.append("|--:|-------|--------|-------|")
        for j, r in enumerate(faceless_famous[:20], 1):
            md_lines.append(f"| {j} | {r['title']} | {r['artist']} | {r['style']} |")
    else:
        md_lines.append("All famous paintings have detected faces.")

    md_path = os.path.join(RESULTS_DIR, "face_coverage.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved report: {md_path}")

    # ── Bar chart ─────────────────────────────────────────────────────
    # Filter styles with at least 5 paintings for readability
    chart_data = [(s, c, r) for s, c, _, r in style_table if c >= 5]
    if chart_data:
        styles_plot = [d[0] for d in chart_data]
        counts_plot = [d[1] for d in chart_data]
        rates_plot = [d[2] for d in chart_data]

        fig, ax = plt.subplots(figsize=(10, max(5, len(chart_data) * 0.4 + 1)))
        y_pos = np.arange(len(styles_plot))

        # Color gradient: red (low rate) → green (high rate)
        colors = []
        for rate in rates_plot:
            t = rate / 100.0
            r = int(220 * (1 - t) + 34 * t)
            g = int(38 * (1 - t) + 163 * t)
            b = int(38 * (1 - t) + 34 * t)
            colors.append(f"#{r:02x}{g:02x}{b:02x}")

        bars = ax.barh(y_pos, rates_plot, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{s} (n={c})" for s, c in zip(styles_plot, counts_plot)],
                           fontsize=9)
        ax.set_xlabel("Face Detection Rate (%)")
        ax.set_title("Face Detection Coverage by Art Style")
        ax.set_xlim(0, 105)
        ax.invert_yaxis()

        # Value labels
        for bar, rate in zip(bars, rates_plot):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.0f}%", va="center", fontsize=8)

        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "face_coverage_chart.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart: {chart_path}")

    # ── Update CSV if columns were missing ────────────────────────────
    if not has_face_col_exists:
        if "has_face" not in fieldnames:
            fieldnames.append("has_face")
        if "face_confidence" not in fieldnames:
            fieldnames.append("face_confidence")
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated CSV with has_face/face_confidence columns: {CSV_PATH}")
    else:
        print("CSV already has has_face column — skipped CSV update")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"OVERALL: {with_face}/{total} paintings have faces ({pct:.1f}%)")
    print(f"\nBest styles:")
    for s, c, f, r in style_table[-5:]:
        print(f"  {s}: {r:.0f}% ({f}/{c})")
    print(f"\nWorst styles:")
    for s, c, f, r in style_table[:5]:
        print(f"  {s}: {r:.0f}% ({f}/{c})")
    if faceless_famous:
        print(f"\nFaceless famous paintings: {len(faceless_famous)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
