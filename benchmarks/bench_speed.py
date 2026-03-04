"""Benchmark 3 — Pipeline Speed Profiling.

Times each stage of the matching pipeline (face detection, CLIP encoding,
FaRL encoding, similarity+fusion) across 50 iterations per face, comparing
MPS vs CPU when available.
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matcher  # noqa: E402

# ── directories ───────────────────────────────────────────────────────
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BENCH_DIR, "test_faces")
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

COLORS = {
    "face_det":  "#2563eb",
    "clip_enc":  "#dc2626",
    "farl_enc":  "#16a34a",
    "sim_fuse":  "#f59e0b",
    "total":     "#8b5cf6",
}

STAGE_NAMES = {
    "face_det":  "Face Detection",
    "clip_enc":  "CLIP Encoding",
    "farl_enc":  "FaRL Encoding",
    "sim_fuse":  "Similarity+Fusion",
    "total":     "Total Pipeline",
}

ITERS = 50
NUM_FACES = 5


def load_test_faces(n=NUM_FACES):
    faces = []
    for fname in sorted(os.listdir(FACES_DIR)):
        if fname.endswith(".jpg") and len(faces) < n:
            path = os.path.join(FACES_DIR, fname)
            img = Image.open(path).convert("RGB")
            faces.append((fname, img))
    return faces


def time_pipeline(img, device_label="default"):
    """Time individual stages of the matching pipeline. Returns dict of durations."""
    img = img.convert("RGB")

    # Stage 1: Face detection
    t0 = time.perf_counter()
    tight_crop, loose_crop = matcher.crop_face(img)
    t1 = time.perf_counter()

    # Stage 2: CLIP encoding
    clip_vec = matcher._encode_clip(loose_crop)
    t2 = time.perf_counter()

    # Stage 3: FaRL encoding
    farl_vec = matcher._encode_farl(tight_crop)
    t3 = time.perf_counter()

    # Stage 4: Similarity + fusion
    clip_sims = matcher.clip_embeddings @ clip_vec
    farl_sims = matcher.farl_embeddings @ farl_vec
    clip_z = (clip_sims - clip_sims.mean()) / (clip_sims.std() + 1e-8)
    farl_z = (farl_sims - farl_sims.mean()) / (farl_sims.std() + 1e-8)
    farl_z[~matcher._has_face] = 0.0
    final = 0.5 * farl_z + 0.5 * clip_z
    _top = np.argsort(final)[-5:][::-1]
    t4 = time.perf_counter()

    return {
        "face_det":  t1 - t0,
        "clip_enc":  t2 - t1,
        "farl_enc":  t3 - t2,
        "sim_fuse":  t4 - t3,
        "total":     t4 - t0,
    }


def run_benchmark(faces, device_label):
    """Run ITERS iterations for each face, collect timing data."""
    all_times = {k: [] for k in STAGE_NAMES}

    for fi, (fname, img) in enumerate(faces):
        print(f"  [{device_label}] {fname}: ", end="", flush=True)
        for i in range(ITERS):
            times = time_pipeline(img, device_label)
            for k, v in times.items():
                all_times[k].append(v)
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
        print(f" {ITERS} iters done")

    return {k: np.array(v) for k, v in all_times.items()}


def stats_table(times_dict, label):
    lines = [
        f"### {label}\n",
        "| Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |",
        "|-------|----------|------------|---------|---------|",
    ]
    for key in ["face_det", "clip_enc", "farl_enc", "sim_fuse", "total"]:
        arr = times_dict[key] * 1000  # to ms
        lines.append(
            f"| {STAGE_NAMES[key]} "
            f"| {arr.mean():.1f} "
            f"| {np.median(arr):.1f} "
            f"| {np.percentile(arr, 95):.1f} "
            f"| {np.percentile(arr, 99):.1f} |"
        )
    return "\n".join(lines)


def make_bar_chart(results_by_device):
    """Grouped bar chart: stages on x, devices as groups."""
    stages = ["face_det", "clip_enc", "farl_enc", "sim_fuse", "total"]
    stage_labels = [STAGE_NAMES[s] for s in stages]
    devices = list(results_by_device.keys())
    n_devices = len(devices)
    n_stages = len(stages)

    device_colors = [COLORS.get(d, "#6b7280") for d in stages]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_w = 0.35
    x = np.arange(n_stages)

    for di, device in enumerate(devices):
        times = results_by_device[device]
        means = [times[s].mean() * 1000 for s in stages]
        stds = [times[s].std() * 1000 for s in stages]
        offset = (di - (n_devices - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=4,
            label=device.upper(),
            color=[c if di == 0 else _lighten(c) for c in device_colors],
            edgecolor="white", linewidth=0.5,
        )
        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.0f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Pipeline Speed ({ITERS} iterations x {NUM_FACES} faces)")
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def _lighten(hex_color, factor=0.4):
    """Lighten a hex color."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    faces = load_test_faces()
    if not faces:
        print("ERROR: No test faces found. Run bench_ablation.py first.")
        return

    print(f"Speed benchmark: {len(faces)} faces x {ITERS} iterations each\n")
    results = {}

    # Run on default device (MPS if available)
    default_device = str(matcher.DEVICE)
    print(f"Running on {default_device.upper()}...")
    results[default_device] = run_benchmark(faces, default_device)

    # If MPS, also run on CPU for comparison
    if default_device == "mps":
        print("\nMoving models to CPU for comparison...")
        matcher.clip_model = matcher.clip_model.to("cpu")
        matcher.farl_model = matcher.farl_model.to("cpu")
        original_device = matcher.DEVICE
        matcher.DEVICE = torch.device("cpu")

        print("Running on CPU...")
        results["cpu"] = run_benchmark(faces, "cpu")

        # Restore MPS
        matcher.clip_model = matcher.clip_model.to(original_device)
        matcher.farl_model = matcher.farl_model.to(original_device)
        matcher.DEVICE = original_device

    # ── Results table ─────────────────────────────────────────────────
    md_lines = ["# Speed Benchmark Results\n"]
    for device, times in results.items():
        md_lines.append(stats_table(times, f"Device: {device.upper()}"))
        md_lines.append("")

    md_path = os.path.join(RESULTS_DIR, "speed_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved results table: {md_path}")

    # ── Bar chart ─────────────────────────────────────────────────────
    fig = make_bar_chart(results)
    chart_path = os.path.join(RESULTS_DIR, "speed_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart: {chart_path}")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for device, times in results.items():
        total = times["total"] * 1000
        print(f"{device.upper()}: mean={total.mean():.1f}ms, "
              f"median={np.median(total):.1f}ms, p95={np.percentile(total, 95):.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
