"""Benchmark 2 — Alpha Sensitivity Analysis.

Sweeps alpha from 0.0 to 1.0, computes Jaccard distance between adjacent
result sets, and generates sensitivity + heatmap plots.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matcher import find_matches  # noqa: E402

# ── directories ───────────────────────────────────────────────────────
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BENCH_DIR, "test_faces")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── matplotlib styling ────────────────────────────────────────────────
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6"]

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


def load_test_faces():
    faces = []
    for fname in sorted(os.listdir(FACES_DIR)):
        if fname.endswith(".jpg"):
            path = os.path.join(FACES_DIR, fname)
            label = os.path.splitext(fname)[0]
            faces.append((label, path))
    return faces


def jaccard_distance(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(union)


def main():
    faces = load_test_faces()
    if not faces:
        print("ERROR: No test faces found in benchmarks/test_faces/. Run bench_ablation.py first.")
        return

    alphas = np.arange(0.0, 1.01, 0.1)
    alphas = np.round(alphas, 1)
    n_faces = len(faces)

    print(f"Alpha sensitivity sweep: {len(alphas)} values x {n_faces} faces")

    # ── Collect all results ───────────────────────────────────────────
    # results[face_idx][alpha_idx] = set of painting indices (top-5)
    results = []
    # Also track top-1 painting index at each alpha
    top1_indices = []
    # Reference top-1 at alpha=0.5
    ref_alpha_idx = list(alphas).index(0.5)

    for fi, (label, path) in enumerate(faces):
        img = Image.open(path).convert("RGB")
        face_results = []
        face_top1 = []
        for ai, alpha in enumerate(alphas):
            matches = find_matches(img, alpha=float(alpha), top_k=5)
            idx_set = {m["painting_index"] for m in matches}
            face_results.append(idx_set)
            face_top1.append(matches[0]["painting_index"])
            print(f"  [{fi+1}/{n_faces}] {label} alpha={alpha:.1f} done")
        results.append(face_results)
        top1_indices.append(face_top1)

    # ── Jaccard distance between adjacent alphas ──────────────────────
    n_pairs = len(alphas) - 1
    jaccard_matrix = np.zeros((n_faces, n_pairs))
    for fi in range(n_faces):
        for pi in range(n_pairs):
            jaccard_matrix[fi, pi] = jaccard_distance(
                results[fi][pi], results[fi][pi + 1]
            )

    midpoints = [(alphas[i] + alphas[i + 1]) / 2 for i in range(n_pairs)]
    mean_jaccard = jaccard_matrix.mean(axis=0)
    std_jaccard = jaccard_matrix.std(axis=0)

    # ── Plot 1: Sensitivity curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        midpoints, mean_jaccard, yerr=std_jaccard,
        fmt="o-", color=COLORS[0], ecolor=COLORS[1],
        capsize=5, capthick=1.5, linewidth=2, markersize=7,
        label="Mean Jaccard distance",
    )
    ax.fill_between(
        midpoints,
        mean_jaccard - std_jaccard,
        mean_jaccard + std_jaccard,
        alpha=0.15, color=COLORS[0],
    )
    ax.set_xlabel("Alpha midpoint")
    ax.set_ylabel("Jaccard Distance (higher = more change)")
    ax.set_title("Result Set Sensitivity to Alpha")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()

    sens_path = os.path.join(RESULTS_DIR, "alpha_sensitivity.png")
    fig.savefig(sens_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved sensitivity plot: {sens_path}")

    # ── Plot 2: Stability heatmap ─────────────────────────────────────
    # For each face, find rank of the alpha=0.5 top-1 painting at every alpha
    rank_matrix = np.full((n_faces, len(alphas)), np.nan)
    for fi in range(n_faces):
        ref_painting = top1_indices[fi][ref_alpha_idx]
        for ai, alpha in enumerate(alphas):
            img = Image.open(faces[fi][1]).convert("RGB")
            matches = find_matches(img, alpha=float(alpha), top_k=10)
            indices = [m["painting_index"] for m in matches]
            if ref_painting in indices:
                rank_matrix[fi, ai] = indices.index(ref_painting) + 1
            else:
                rank_matrix[fi, ai] = 11  # not in top-10

    fig, ax = plt.subplots(figsize=(10, max(4, n_faces * 0.45 + 1)))
    im = ax.imshow(
        rank_matrix, aspect="auto",
        cmap="RdYlGn_r", vmin=1, vmax=11,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(n_faces))
    ax.set_yticklabels([f[0].split("_", 1)[-1].replace("_", " ").title() for f in faces],
                       fontsize=9)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Test Face")
    ax.set_title("Stability: Rank of Best Match (alpha=0.5) Across All Alphas")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank (1=best, 11=not in top-10)")

    # Annotate cells with rank values
    for fi in range(n_faces):
        for ai in range(len(alphas)):
            val = int(rank_matrix[fi, ai])
            txt = str(val) if val <= 10 else ">10"
            color = "white" if val >= 6 else "black"
            ax.text(ai, fi, txt, ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    hmap_path = os.path.join(RESULTS_DIR, "alpha_heatmap.png")
    fig.savefig(hmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {hmap_path}")

    print("\nAlpha sensitivity benchmark complete!")


if __name__ == "__main__":
    main()
