"""
Generate shareable comparison card images (1080x1080) for Face-to-Art.

Usage:
    from card import create_card, scale_score

    pct = scale_score(0.28)            # -> ~76
    card = create_card(selfie, painting, "Mona Lisa", "Leonardo da Vinci",
                       "1503", "Renaissance", overall_pct=pct, ...)
    card.save("card.png")
"""

import os

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Font loading — try macOS system fonts, fall back to Pillow default
# ---------------------------------------------------------------------------
_FONT_PATHS = {
    "bold": [
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/ArialHB.ttc",
    ],
    "regular": [
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
    ],
    "italic": [
        "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        "/System/Library/Fonts/Avenir Next.ttc",
    ],
}


def _load_font(style: str, size: int, bold_index: int = 0) -> ImageFont.FreeTypeFont:
    """Try system font paths, fall back to Pillow default."""
    for path in _FONT_PATHS.get(style, []):
        if os.path.exists(path):
            try:
                if path.endswith(".ttc"):
                    # For .ttc: index 0 = regular, higher indices = bold/italic
                    idx = bold_index if style == "bold" else 0
                    return ImageFont.truetype(path, size, index=idx)
                return ImageFont.truetype(path, size)
            except (OSError, IndexError):
                continue
    return ImageFont.load_default(size)


# ---------------------------------------------------------------------------
# Score scaling
# ---------------------------------------------------------------------------
def scale_score(raw: float, low: float = 0.15, high: float = 0.40) -> int:
    """Map a raw cosine similarity from [low, high] to [50, 99].

    Values outside the range are clamped. Returns an integer percentage.
    """
    t = (raw - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return int(50 + t * 49)


# ---------------------------------------------------------------------------
# Card layout constants
# ---------------------------------------------------------------------------
CARD_W, CARD_H = 1080, 1080
BG_COLOR = (26, 26, 46)        # #1a1a2e deep navy
ACCENT_GOLD = (255, 183, 77)   # warm gold/orange
WHITE = (255, 255, 255)
LIGHT_GRAY = (180, 180, 195)
DIM_GRAY = (100, 100, 120)
BORDER_COLOR = (220, 220, 230)
BORDER_WIDTH = 3

IMG_SIZE = 440
IMG_GAP = 40
IMG_Y = 170


def _fit_and_border(img: Image.Image, size: int) -> Image.Image:
    """Resize to square, add thin white border."""
    img = img.convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    bordered = Image.new("RGB", (size + BORDER_WIDTH * 2, size + BORDER_WIDTH * 2), BORDER_COLOR)
    bordered.paste(img, (BORDER_WIDTH, BORDER_WIDTH))
    return bordered


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
) -> int:
    """Draw text centered horizontally. Returns the bottom y coordinate."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (CARD_W - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)
    return y + th


def _truncate_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Truncate text with ellipsis if it exceeds max_width."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    while len(text) > 0:
        truncated = text + "..."
        if draw.textbbox((0, 0), truncated, font=font)[2] <= max_width:
            return truncated
        text = text[:-1]
    return "..."


# ---------------------------------------------------------------------------
# Main card creation
# ---------------------------------------------------------------------------
def create_card(
    selfie: Image.Image,
    painting: Image.Image,
    title: str,
    artist: str,
    year: str,
    style: str,
    overall_pct: int,
    clip_pct: int,
    farl_pct: int,
    fun_fact: str | None = None,
) -> Image.Image:
    """Create a 1080x1080 comparison card image.

    Args:
        selfie:      PIL Image of the user's photo.
        painting:    PIL Image of the matched painting.
        title:       Painting title.
        artist:      Artist name.
        year:        Year string (can be empty).
        style:       Art style / movement.
        overall_pct: Overall match percentage (already scaled).
        clip_pct:    Visual vibe percentage (already scaled).
        farl_pct:    Facial match percentage (already scaled).
        fun_fact:    Optional fun fact text.

    Returns:
        PIL Image (1080x1080 RGB).
    """
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # --- Fonts ---
    font_title = _load_font("bold", 40, bold_index=1)
    font_label = _load_font("regular", 24)
    font_pct = _load_font("bold", 72, bold_index=1)
    font_info = _load_font("regular", 28)
    font_detail = _load_font("regular", 22)
    font_fact = _load_font("italic", 20)
    font_watermark = _load_font("regular", 18)

    # --- Top title ---
    _draw_centered_text(draw, 40, "What Painting Are You?", font_title, WHITE)

    # --- Side-by-side images ---
    selfie_img = _fit_and_border(selfie, IMG_SIZE)
    painting_img = _fit_and_border(painting, IMG_SIZE)

    total_w = selfie_img.width + IMG_GAP + painting_img.width
    start_x = (CARD_W - total_w) // 2

    card.paste(selfie_img, (start_x, IMG_Y))
    card.paste(painting_img, (start_x + selfie_img.width + IMG_GAP, IMG_Y))

    # --- Labels below images ---
    label_y = IMG_Y + selfie_img.height + 10
    selfie_center_x = start_x + selfie_img.width // 2
    painting_center_x = start_x + selfie_img.width + IMG_GAP + painting_img.width // 2

    you_bbox = draw.textbbox((0, 0), "You", font=font_label)
    you_w = you_bbox[2] - you_bbox[0]
    draw.text((selfie_center_x - you_w // 2, label_y), "You", font=font_label, fill=LIGHT_GRAY)

    painting_label = _truncate_text(title, font_label, painting_img.width)
    pl_bbox = draw.textbbox((0, 0), painting_label, font=font_label)
    pl_w = pl_bbox[2] - pl_bbox[0]
    draw.text(
        (painting_center_x - pl_w // 2, label_y),
        painting_label,
        font=font_label,
        fill=LIGHT_GRAY,
    )

    # --- Match percentage ---
    pct_y = label_y + 55
    pct_text = f"{overall_pct}% Match"
    bottom = _draw_centered_text(draw, pct_y, pct_text, font_pct, ACCENT_GOLD)

    # --- Painting info ---
    info_y = bottom + 18
    year_part = f", {year}" if year else ""
    info_text = f"{title} by {artist}{year_part}"
    info_text = _truncate_text(info_text, font_info, CARD_W - 80)
    bottom = _draw_centered_text(draw, info_y, info_text, font_info, WHITE)

    # --- Detail scores ---
    detail_y = bottom + 12
    detail_text = f"Visual Vibe: {clip_pct}%  \u00b7  Facial Match: {farl_pct}%"
    bottom = _draw_centered_text(draw, detail_y, detail_text, font_detail, LIGHT_GRAY)

    # --- Style ---
    if style:
        style_y = bottom + 8
        bottom = _draw_centered_text(draw, style_y, style, font_detail, DIM_GRAY)

    # --- Fun fact ---
    if fun_fact:
        fact_y = bottom + 16
        wrapped = _truncate_text(fun_fact, font_fact, CARD_W - 100)
        _draw_centered_text(draw, fact_y, wrapped, font_fact, DIM_GRAY)

    # --- Watermark ---
    _draw_centered_text(draw, CARD_H - 45, "face-to-art", font_watermark, DIM_GRAY)

    return card


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy selfie (warm tone) and painting (cool tone)
    selfie = Image.new("RGB", (600, 800), (210, 160, 130))
    painting = Image.new("RGB", (500, 700), (60, 90, 140))

    card = create_card(
        selfie=selfie,
        painting=painting,
        title="Girl with a Pearl Earring",
        artist="Johannes Vermeer",
        year="1665",
        style="Dutch Golden Age",
        overall_pct=scale_score(0.32),
        clip_pct=scale_score(0.28),
        farl_pct=scale_score(0.35),
        fun_fact="Vermeer used ultramarine blue pigment worth more than gold.",
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_card.png")
    card.save(out_path)
    print(f"Test card saved: {out_path}")
    print(f"Card size: {card.size}")
    print(f"Score examples: raw 0.15 -> {scale_score(0.15)}%, "
          f"0.28 -> {scale_score(0.28)}%, 0.40 -> {scale_score(0.40)}%")
