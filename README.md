# 🎨 Face-to-Art

Find your painting twin from 3,200+ masterpieces. Upload a selfie and the app matches you to classical portraits using a two-model AI approach — one for artistic style, one for facial resemblance.

![sample card](test_card.png)

## How it works

Two vision models score every painting against your selfie:

| Model | Role | What it captures |
|-------|------|-----------------|
| **CLIP** ViT-B/32 | Vibe engine | Artistic style, mood, color palette, composition |
| **FaRL** ViT-B/16 | Face engine | Facial structure, expression, resemblance |

An **alpha slider** (0–1) blends the two:
- `α = 0` → pure artistic vibe (CLIP only)
- `α = 0.5` → balanced (default)
- `α = 1` → pure facial match (FaRL only)

Results are presented as shareable 1080×1080 comparison cards.

## Interfaces

### Gradio web app

```bash
python app.py
```

Opens a local web UI with image upload/webcam, alpha slider, and a gallery of top 5 matches.

### Telegram bot

```bash
export BOT_TOKEN="your_token_from_botfather"
python bot.py
```

**Commands:**
| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/vibe` | Re-run with artistic style emphasis (α=0.2) |
| `/face` | Re-run with facial resemblance emphasis (α=0.8) |
| `/balance` | Inline keyboard to pick any preset (Pure Vibe → Pure Face) |
| `/help` | Show commands |

Send a selfie → get top 3 matches with comparison cards + full-res download.

## Project structure

```
face_to_art/
├── app.py              # Gradio web interface
├── bot.py              # Telegram bot (aiogram 3.x)
├── matcher.py          # Matching engine (CLIP + FaRL inference)
├── card.py             # 1080×1080 comparison card generator
├── requirements.txt    # Python dependencies
├── data/
│   ├── portraits/              # 3,140+ painting images (512×512)
│   ├── portraits_metadata.csv  # Title, artist, genre, style per painting
│   ├── clip_embeddings.npy     # Precomputed CLIP vectors (N×512)
│   └── farl_embeddings.npy     # Precomputed FaRL vectors (N×512)
├── models/
│   └── FaRL-Base-Patch16-LAIONFace20M-ep16.pth
└── scripts/
    ├── download_portraits.py   # Fetch WikiArt portraits
    ├── download_famous.py      # Fetch ~200 famous paintings from Wikimedia
    ├── download_farl.py        # Download FaRL weights
    └── compute_embeddings.py   # Precompute CLIP & FaRL embeddings
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**2. Download data & models**

```bash
python scripts/download_portraits.py
python scripts/download_famous.py
python scripts/download_farl.py
```

**3. Compute embeddings**

```bash
python scripts/compute_embeddings.py
```

This precomputes CLIP and FaRL vectors for all paintings (runs once, takes a few minutes).

**4. Run**

```bash
# Web UI
python app.py

# Telegram bot
export BOT_TOKEN="your_token"
python bot.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | auto (MPS → CPU) | Force compute device (`mps`, `cpu`, `cuda`) |
| `BOT_TOKEN` | — | Telegram bot token (required for `bot.py`) |

## Tech stack

- **PyTorch** — model inference
- **OpenAI CLIP** — artistic style embeddings
- **FaRL** — face-aware representation learning
- **OpenCV** — Haar cascade face detection
- **Pillow** — image processing & card rendering
- **Gradio** — web interface
- **aiogram 3.x** — Telegram bot framework
