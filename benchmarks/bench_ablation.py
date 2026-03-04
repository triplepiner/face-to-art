"""Benchmark 1 — Ablation Study: Dual vs Single Model Comparison.

Downloads 15 diverse public-domain portrait photos from Wikimedia Commons,
runs find_matches at three alpha settings, generates comparison grids,
summary table, shareable cards, and a full CSV of results.
"""

import csv
import io
import os
import sys
import time
import urllib.parse

import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from card import create_card, scale_score  # noqa: E402
from matcher import find_matches, get_painting_image  # noqa: E402

# ── directories ───────────────────────────────────────────────────────
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BENCH_DIR, "test_faces")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
COMP_DIR = os.path.join(RESULTS_DIR, "ablation_comparisons")
CARDS_DIR = os.path.join(BENCH_DIR, "cards")
for d in [FACES_DIR, RESULTS_DIR, COMP_DIR, CARDS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── font ──────────────────────────────────────────────────────────────
def _load_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

FONT = _load_font(13)
FONT_LG = _load_font(16)
FONT_TITLE = _load_font(20)

# ── test face sources ─────────────────────────────────────────────────
# Diverse portrait photos from free image sources (Pixabay License / CC0).
# Mix of genders, ages, ethnicities.
# (id, url, display_label)
_CANDIDATES = [
    ("01_man_beard",      "https://cdn.pixabay.com/photo/2016/11/21/12/42/beard-1845166_640.jpg",          "Man with Beard"),
    ("02_woman_hat",      "https://cdn.pixabay.com/photo/2017/08/01/01/33/beanie-2562646_640.jpg",         "Woman in Beanie"),
    ("03_woman_smile",    "https://cdn.pixabay.com/photo/2018/01/15/07/51/woman-3083383_640.jpg",          "Smiling Woman"),
    ("04_elder_man",      "https://cdn.pixabay.com/photo/2016/11/29/03/53/old-man-1867632_640.jpg",        "Elderly Man"),
    ("05_young_girl",     "https://cdn.pixabay.com/photo/2016/11/29/09/38/adult-1868750_640.jpg",          "Young Woman"),
    ("06_boy_portrait",   "https://cdn.pixabay.com/photo/2015/06/22/08/37/children-817365_640.jpg",        "Boy Portrait"),
    ("07_african_woman",  "https://cdn.pixabay.com/photo/2017/04/01/21/06/portrait-2194457_640.jpg",       "African Woman"),
    ("08_asian_man",      "https://cdn.pixabay.com/photo/2016/11/18/19/07/happy-1836445_640.jpg",          "Asian Man"),
    ("09_woman_curly",    "https://cdn.pixabay.com/photo/2017/02/16/23/10/smile-2072908_640.jpg",          "Woman Curly Hair"),
    ("10_man_glasses",    "https://cdn.pixabay.com/photo/2017/11/02/14/26/model-2911330_640.jpg",          "Man with Glasses"),
    ("11_elder_woman",    "https://cdn.pixabay.com/photo/2017/09/01/21/53/old-woman-2705803_640.jpg",      "Elderly Woman"),
    ("12_child_african",  "https://cdn.pixabay.com/photo/2016/01/10/17/37/child-1132087_640.jpg",          "African Child"),
    ("13_man_turban",     "https://cdn.pixabay.com/photo/2016/11/21/16/01/man-1846050_640.jpg",            "Man in Turban"),
    ("14_woman_veil",     "https://cdn.pixabay.com/photo/2017/04/20/20/00/woman-2245984_640.jpg",          "Woman Portrait"),
    ("15_man_suit",       "https://cdn.pixabay.com/photo/2016/11/21/15/54/man-1845814_640.jpg",            "Man in Suit"),
    ("16_woman_elder2",   "https://cdn.pixabay.com/photo/2015/01/08/18/11/laptops-593296_640.jpg",         "Woman at Desk"),
    ("17_man_young",      "https://cdn.pixabay.com/photo/2018/04/27/03/50/portrait-3353699_640.jpg",       "Young Man"),
    ("18_woman_red",      "https://cdn.pixabay.com/photo/2017/08/06/12/06/people-2591874_640.jpg",         "Woman Outdoors"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def download_face(face_id, url, label):
    path = os.path.join(FACES_DIR, f"{face_id}.jpg")
    if os.path.exists(path):
        print(f"  [cached] {label}")
        return path
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        img.save(path, quality=95)
        print(f"  + {label}")
        return path
    except Exception as e:
        print(f"  x {label}: {e}")
        return None


def download_all_faces(target=15):
    print("Downloading test portrait photos...")
    faces = []
    for i, (fid, url, lbl) in enumerate(_CANDIDATES):
        if len(faces) >= target:
            break
        p = download_face(fid, url, lbl)
        if p:
            faces.append((fid, p, lbl))
        if i < len(_CANDIDATES) - 1:
            time.sleep(0.5)
    print(f"  Got {len(faces)}/{target} faces\n")
    return faces


# ── comparison grid ───────────────────────────────────────────────────
THUMB = 200
GAP = 8
LABEL_H = 40
ROW_H = THUMB + LABEL_H + GAP
COLS = 4
BG = (255, 255, 255)
BORDER = (200, 200, 200)
TEXT_COLOR = (30, 30, 30)
ACCENT = (0, 102, 204)

ROW_LABELS = [
    "CLIP Only (Vibe)   alpha=0.0",
    "FaRL Only (Face)   alpha=1.0",
    "Blended (Ours)     alpha=0.5",
]


def _thumb(img):
    """Resize to THUMB x THUMB with center crop."""
    img = img.copy()
    img.thumbnail((THUMB * 2, THUMB * 2), Image.LANCZOS)
    w, h = img.size
    left = (w - THUMB) // 2
    top = (h - THUMB) // 2
    return img.crop((left, top, left + THUMB, top + THUMB))


def make_comparison(selfie, face_label, matches_by_alpha):
    """Create a 4-col x 3-row comparison grid image."""
    left_margin = 240
    width = left_margin + COLS * (THUMB + GAP) + GAP
    title_h = 50
    height = title_h + 3 * ROW_H + GAP

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text((left_margin, 12), f"Ablation: {face_label}", fill=TEXT_COLOR, font=FONT_TITLE)

    alphas_order = [0.0, 1.0, 0.5]  # CLIP, FaRL, Blended
    selfie_thumb = _thumb(selfie)

    for row_i, (alpha, row_label) in enumerate(zip(alphas_order, ROW_LABELS)):
        y_base = title_h + row_i * ROW_H

        # Row label
        draw.text((8, y_base + THUMB // 2 - 10), row_label, fill=ACCENT, font=FONT)

        # Selfie in column 0
        canvas.paste(selfie_thumb, (left_margin + GAP, y_base))
        draw.text((left_margin + GAP, y_base + THUMB + 2), "Input", fill=TEXT_COLOR, font=FONT)

        # Top-3 matches
        matches = matches_by_alpha[alpha]
        for col_i, m in enumerate(matches[:3]):
            x = left_margin + (col_i + 1) * (THUMB + GAP) + GAP
            painting = get_painting_image(m["painting_index"])
            canvas.paste(_thumb(painting), (x, y_base))

            title = m.get("title", "?")[:22]
            score = scale_score(m["blended_score"])
            lbl = f"{title} ({score}%)"
            draw.text((x, y_base + THUMB + 2), lbl, fill=TEXT_COLOR, font=FONT)

    return canvas


# ── main ──────────────────────────────────────────────────────────────
def main():
    faces = download_all_faces(target=15)
    if not faces:
        print("ERROR: No test faces downloaded. Aborting.")
        return

    alphas = {0.0: "CLIP Only", 1.0: "FaRL Only", 0.5: "Blended"}
    all_results = {}  # face_id -> {alpha: [matches]}
    csv_rows = []

    print("Running ablation study...")
    for face_id, face_path, label in faces:
        selfie = Image.open(face_path).convert("RGB")
        matches_by_alpha = {}

        for alpha in [0.0, 1.0, 0.5]:
            matches = find_matches(selfie, alpha=alpha, top_k=3)
            matches_by_alpha[alpha] = matches
            mode = alphas[alpha]
            print(f"  {label} | {mode}: top={matches[0]['title'][:30]}")

            for rank, m in enumerate(matches, 1):
                csv_rows.append({
                    "benchmark": f"ablation_{mode.lower().replace(' ', '_')}",
                    "test_photo": label,
                    "rank": rank,
                    "painting": m.get("title", "Unknown"),
                    "artist": m.get("artist", "Unknown"),
                    "clip_similarity": f"{m['clip_score']:.4f}",
                    "farl_similarity": f"{m['farl_score']:.4f}",
                    "blended_score": f"{m['blended_score']:.4f}",
                })

        all_results[face_id] = matches_by_alpha

        # Save comparison grid
        comp = make_comparison(selfie, label, matches_by_alpha)
        comp.save(os.path.join(COMP_DIR, f"{face_id}_comparison.jpg"), quality=95)

        # Generate shareable card (blended top-1)
        top = matches_by_alpha[0.5][0]
        painting = get_painting_image(top["painting_index"])
        overall_pct = scale_score(top["blended_score"])
        clip_pct = scale_score(top["clip_score"])
        farl_pct = scale_score(top["farl_score"])
        card = create_card(
            selfie=selfie, painting=painting,
            title=top.get("title", "Unknown"),
            artist=top.get("artist", "Unknown"),
            year="", style=top.get("style", ""),
            overall_pct=overall_pct, clip_pct=clip_pct, farl_pct=farl_pct,
        )
        card.save(os.path.join(CARDS_DIR, f"{face_id}_card.jpg"), quality=95)

    # ── Summary markdown table ────────────────────────────────────────
    print("\nGenerating ablation_results.md...")
    md_lines = [
        "# Ablation Results: CLIP vs FaRL vs Blended\n",
        "| Test Face | CLIP Top-1 | FaRL Top-1 | Blend Top-1 | Agreement? |",
        "|-----------|-----------|-----------|------------|------------|",
    ]
    for face_id, face_path, label in faces:
        m = all_results[face_id]
        clip_top = m[0.0][0]["title"][:25]
        farl_top = m[1.0][0]["title"][:25]
        blend_top = m[0.5][0]["title"][:25]

        # Agreement: any painting in top-3 of ALL three methods
        sets = [
            {r["painting_index"] for r in m[a][:3]}
            for a in [0.0, 1.0, 0.5]
        ]
        common = sets[0] & sets[1] & sets[2]
        agree = "Yes" if common else "No"

        md_lines.append(f"| {label} | {clip_top} | {farl_top} | {blend_top} | {agree} |")

    md_path = os.path.join(RESULTS_DIR, "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  Saved {md_path}")

    # ── CSV with all results ──────────────────────────────────────────
    csv_path = os.path.join(CARDS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark", "test_photo", "rank", "painting", "artist",
            "clip_similarity", "farl_similarity", "blended_score",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    print(f"\nAblation complete! {len(faces)} faces processed.")
    print(f"  Comparison grids: {COMP_DIR}/")
    print(f"  Cards:            {CARDS_DIR}/")
    print(f"  Results table:    {md_path}")
    print(f"  Full CSV:         {csv_path}")


if __name__ == "__main__":
    main()
