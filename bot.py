"""Telegram bot for Face-to-Art — find your painting twin."""

import asyncio
import io
import os
from collections import OrderedDict

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from PIL import Image

from card import create_card, scale_score
from matcher import find_matches, get_painting_image

BOT_TOKEN = os.environ["BOT_TOKEN"]

router = Router()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
dp.include_router(router)

# ── per-user state (LRU-capped) ──────────────────────────────────────
MAX_USERS = 1000
user_states: OrderedDict[int, dict] = OrderedDict()

# limit concurrent model inference
inference_sem = asyncio.Semaphore(3)

ALPHA_PRESETS = [
    ("🎨 Pure Vibe",   0.0),
    ("🎨 Mostly Vibe", 0.2),
    ("⚖️ Balanced",    0.5),
    ("👤 Mostly Face", 0.8),
    ("👤 Pure Face",   1.0),
]

WELCOME = (
    "🎨 <b>What Painting Are You?</b>\n\n"
    "Send me a selfie and I'll find your artistic twin "
    "from 3,200+ masterpieces!\n\n"
    "<b>Commands:</b>\n"
    "/vibe — artistic style emphasis\n"
    "/face — facial resemblance emphasis\n"
    "/balance — adjust style↔face ratio\n"
    "/help — this message"
)


# ── helpers ───────────────────────────────────────────────────────────

def get_user(user_id: int) -> dict:
    """Get or create user state, evicting oldest if over MAX_USERS."""
    if user_id in user_states:
        user_states.move_to_end(user_id)
        return user_states[user_id]
    state = {"selfie": None, "alpha": 0.5}
    user_states[user_id] = state
    if len(user_states) > MAX_USERS:
        user_states.popitem(last=False)
    return state


def pil_to_buffered(img: Image.Image, filename: str = "card.jpg") -> BufferedInputFile:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return BufferedInputFile(file=buf.getvalue(), filename=filename)


def alpha_keyboard(current: float) -> InlineKeyboardMarkup:
    rows = []
    for label, val in ALPHA_PRESETS:
        mark = " ✓" if abs(val - current) < 0.01 else ""
        rows.append([InlineKeyboardButton(
            text=f"{label}{mark}",
            callback_data=f"alpha:{val}",
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_caption(match: dict, rank: int, overall_pct: int,
                   clip_pct: int, farl_pct: int) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    title = match.get("title", "Unknown")
    artist = match.get("artist", "Unknown")
    style = match.get("style", "")

    if rank == 1:
        lines = [
            f"{medals[1]} You are <b>{overall_pct}% {title}</b>!",
            f"🎨 {artist} · {style}" if style else f"🎨 {artist}",
            f"👁 Visual Vibe: {clip_pct}% · 👤 Facial Match: {farl_pct}%",
        ]
    else:
        lines = [
            f"{medals.get(rank, '🏅')} <b>{overall_pct}%</b> — {title}",
            f"🎨 {artist}" + (f" · {style}" if style else ""),
        ]
    return "\n".join(lines)


async def run_matching(selfie: Image.Image, alpha: float, top_k: int = 3):
    async with inference_sem:
        return await asyncio.to_thread(find_matches, selfie, alpha, top_k)


async def send_results(message: Message, selfie: Image.Image,
                       matches: list[dict], alpha: float):
    for rank, m in enumerate(matches, 1):
        overall_pct = scale_score(m["blended_score"])
        clip_pct = scale_score(m["clip_score"])
        farl_pct = scale_score(m["farl_score"])

        painting = get_painting_image(m["painting_index"])
        caption = format_caption(m, rank, overall_pct, clip_pct, farl_pct)

        if rank == 1:
            card = create_card(
                selfie=selfie,
                painting=painting,
                title=m.get("title", "Unknown"),
                artist=m.get("artist", "Unknown"),
                year="",
                style=m.get("style", ""),
                overall_pct=overall_pct,
                clip_pct=clip_pct,
                farl_pct=farl_pct,
            )
            card_file = pil_to_buffered(card, "card.jpg")
            # send as photo (preview) then as document (full-res download)
            await message.answer_photo(photo=card_file, caption=caption,
                                       parse_mode="HTML")
            doc_file = pil_to_buffered(card, "your_painting_twin.jpg")
            await message.answer_document(
                document=doc_file,
                caption="📥 Full-res card for sharing",
            )
        else:
            painting_file = pil_to_buffered(painting, f"match_{rank}.jpg")
            await message.answer_photo(photo=painting_file, caption=caption,
                                       parse_mode="HTML")

    mode_label = (
        "🎨 Vibe-focused" if alpha < 0.3
        else "👤 Face-focused" if alpha > 0.7
        else "⚖️ Balanced"
    )
    await message.answer(
        f"Current mode: <b>{mode_label}</b> (α={alpha:.1f})\n\n"
        "Try /vibe for artistic matching, /face for facial matching,\n"
        "or /balance to fine-tune the style↔face ratio!",
        parse_mode="HTML",
    )


async def rerun_with_alpha(message: Message, alpha: float, mode_name: str):
    state = get_user(message.from_user.id)
    if state["selfie"] is None:
        await message.answer("📸 Send me a selfie first!")
        return
    state["alpha"] = alpha
    status = await message.answer(f"🔍 Re-analyzing with <b>{mode_name}</b> mode...",
                                  parse_mode="HTML")
    matches = await run_matching(state["selfie"], alpha)
    await status.delete()
    await send_results(message, state["selfie"], matches, alpha)


# ── handlers ──────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message):
    get_user(message.from_user.id)
    await message.answer(WELCOME, parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(WELCOME, parse_mode="HTML")


@router.message(Command("vibe"))
async def cmd_vibe(message: Message):
    await rerun_with_alpha(message, alpha=0.2, mode_name="Artistic Vibe")


@router.message(Command("face"))
async def cmd_face(message: Message):
    await rerun_with_alpha(message, alpha=0.8, mode_name="Facial Match")


@router.message(Command("balance"))
async def cmd_balance(message: Message):
    state = get_user(message.from_user.id)
    await message.answer(
        "⚖️ <b>Adjust the matching balance</b>\n\n"
        "🎨 <i>Vibe</i> = artistic style & mood\n"
        "👤 <i>Face</i> = facial resemblance\n\n"
        "Pick a preset, then send a photo (or I'll re-run your last selfie):",
        parse_mode="HTML",
        reply_markup=alpha_keyboard(state["alpha"]),
    )


@router.callback_query(F.data.startswith("alpha:"))
async def cb_alpha(callback: CallbackQuery):
    alpha = float(callback.data.split(":")[1])
    state = get_user(callback.from_user.id)
    state["alpha"] = alpha

    label = next((l for l, v in ALPHA_PRESETS if abs(v - alpha) < 0.01), f"α={alpha}")
    await callback.message.edit_reply_markup(reply_markup=alpha_keyboard(alpha))
    await callback.answer(f"Set to {label}")

    # auto re-run if we have a stored selfie
    if state["selfie"] is not None:
        status = await callback.message.answer(
            f"🔍 Re-analyzing with <b>{label}</b>...", parse_mode="HTML"
        )
        matches = await run_matching(state["selfie"], alpha)
        await status.delete()
        await send_results(callback.message, state["selfie"], matches, alpha)


@router.message(F.photo)
async def on_photo(message: Message):
    state = get_user(message.from_user.id)
    status = await message.answer("🔍 Analyzing your portrait...")

    # download highest-res photo
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    data = await bot.download_file(file.file_path)
    selfie = Image.open(io.BytesIO(data.read())).convert("RGB")

    # store for /vibe, /face, /balance reuse
    state["selfie"] = selfie

    matches = await run_matching(selfie, state["alpha"])
    await status.delete()
    await send_results(message, selfie, matches, state["alpha"])


@router.message()
async def on_text(message: Message):
    await message.answer("📸 Send me a photo to find your painting twin!")


# ── entry point ───────────────────────────────────────────────────────

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
