# Configurable environment variables:
#   DEVICE=mps|cpu              — force compute device (auto-detects MPS by default)
#   PAINTINGS_SOURCE=local|huggingface  — where to load painting images from

"""
Gradio web app for Face-to-Art.

Upload a selfie, adjust the vibe/face slider, and discover which famous
painting you resemble most.
"""

import gradio as gr
from PIL import Image

from card import create_card, scale_score
from matcher import crop_face, find_matches, get_painting_image


def _face_detected(image: Image.Image) -> bool:
    """Check whether crop_face found a real face (vs returning the original)."""
    tight, _ = crop_face(image)
    # crop_face returns the original image object itself when no face is found
    return tight is not image


def _process(image: Image.Image | None, alpha: float) -> tuple:
    """Main submit handler. Returns (card, gallery, details_md)."""
    # --- Edge cases ---
    if image is None:
        raise gr.Error("Please upload a photo first.")

    try:
        image = image.convert("RGB")
    except Exception:
        raise gr.Error("Couldn't read that file. Please upload a JPEG or PNG image.")

    w, h = image.size
    if w < 64 or h < 64:
        raise gr.Error("Image is too small — please upload a photo at least 64x64 pixels.")

    face_found = _face_detected(image)

    # --- Run matching ---
    matches = find_matches(image, alpha=alpha, top_k=5)

    # --- Build comparison card for #1 ---
    top = matches[0]
    painting_img = get_painting_image(top["painting_index"])
    card = create_card(
        selfie=image,
        painting=painting_img,
        title=top["title"],
        artist=top["artist"],
        year="",
        style=top.get("style", ""),
        overall_pct=scale_score(top["blended_score"], low=-1.0, high=4.0),
        clip_pct=scale_score(top["clip_score"]),
        farl_pct=scale_score(top["farl_score"]),
    )

    # --- Gallery images ---
    gallery = []
    for m in matches:
        p_img = get_painting_image(m["painting_index"])
        caption = f'{m["title"]} by {m["artist"]}'
        gallery.append((p_img, caption))

    # --- Details markdown ---
    lines = []

    if not face_found:
        lines.append(
            "> **No face detected** — matching on overall visual style. "
            "Try a clearer photo with your face visible for better results.\n"
        )

    for rank, m in enumerate(matches, 1):
        overall = scale_score(m["blended_score"], low=-1.0, high=4.0)
        clip_pct = scale_score(m["clip_score"])
        farl_pct = scale_score(m["farl_score"])
        style = m.get("style", "")

        lines.append(f"### #{rank} — {m['title']}")
        lines.append(f"**{m['artist']}**" + (f" · *{style}*" if style else ""))
        lines.append(
            f"**{overall}% Match** · "
            f"Visual Vibe: {clip_pct}% · Facial Match: {farl_pct}%"
        )
        lines.append("")

    details_md = "\n".join(lines)

    return card, gallery, details_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="What Painting Are You?") as demo:
    gr.Markdown(
        "# 🎨 What Painting Are You?\n"
        "Upload a selfie and our AI will match you with your artistic twin "
        "from **3,200+ famous paintings** using two AI models — one for "
        "artistic vibe, one for facial resemblance.\n\n"
        "📸 *We don't store your photos.*"
    )

    with gr.Row():
        with gr.Column(scale=2):
            selfie_input = gr.Image(
                type="pil",
                label="Upload a selfie",
                sources=["upload", "webcam"],
            )
        with gr.Column(scale=1):
            alpha_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="← Artistic Vibe ... Facial Resemblance →",
            )
            submit_btn = gr.Button("🔍 Find My Painting Twin", variant="primary")

    card_output = gr.Image(label="Your Comparison Card", type="pil")
    gallery_output = gr.Gallery(
        label="Top 5 Matches",
        columns=5,
        object_fit="contain",
        height=300,
    )
    details_output = gr.Markdown(label="Match Details")

    submit_btn.click(
        fn=_process,
        inputs=[selfie_input, alpha_slider],
        outputs=[card_output, gallery_output, details_output],
    )

demo.launch(theme=gr.themes.Soft())
