import os
import io
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

router = APIRouter()

# ── Constants ──────────────────────────────────────────────────────────────────
W, H = 1280, 720  # 16:9, standard video resolution
PAD_X = 90

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Inter-Bold.ttf")

BG              = (250, 250, 252)
ACCENT          = (79, 70, 229)    # indigo
TEXT_DARK       = (24, 24, 27)
TEXT_GRAY       = (82, 82, 91)
TEXT_LIGHT_GRAY = (161, 161, 170)


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateSlideImageRequest(BaseModel):
    title:         str
    bullets:       list[str]
    module_label:  str   # e.g. "Module 1 · Sales Mastery" — shown as a small label at top
    slide_number:  int   # 1-indexed
    total_slides:  int


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Loads the Inter variable font at a given weight and size."""
    f = ImageFont.truetype(FONT_PATH, size)
    f.set_variation_by_name(weight)
    return f


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Pixel-accurate word wrap — measures actual rendered width, not character count."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def base_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Creates the background: subtle decorative circles + top accent bar."""
    img = Image.new("RGB", (W, H), BG)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.ellipse([W - 260, -180, W + 220, 280], fill=(79, 70, 229, 18))
    odraw.ellipse([-150, H - 220, 220, H + 150], fill=(79, 70, 229, 12))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    return img, draw


def progress_bar(draw: ImageDraw.ImageDraw, current: int, total: int) -> None:
    bar_y = H - 14
    draw.rectangle([0, bar_y, W, H], fill=(235, 235, 240))
    fill_w = int(W * (current / total)) if total > 0 else 0
    draw.rectangle([0, bar_y, fill_w, H], fill=ACCENT)


def render_slide(title: str, bullets: list[str], module_label: str, slide_number: int, total_slides: int) -> Image.Image:
    """
    Renders one slide as a PIL Image (1280x720).

    Layout: module label (top-left) → title (auto-wraps to max 2 lines,
    shrinks font if needed) → accent underline → bullet list (each bullet
    wraps independently) → slide counter (bottom-right) → progress bar.
    """
    img, draw = base_slide()

    label_font = load_font("SemiBold", 20)
    draw.text((PAD_X, 50), module_label.upper(), font=label_font, fill=ACCENT)

    # Title — wrap to max 2 lines, shrinking font size if it doesn't fit
    title_size = 54
    title_font = load_font("Bold", title_size)
    max_title_width = W - 2 * PAD_X
    title_lines = wrap_text(draw, title, title_font, max_title_width)
    while len(title_lines) > 2 and title_size > 36:
        title_size -= 4
        title_font = load_font("Bold", title_size)
        title_lines = wrap_text(draw, title, title_font, max_title_width)
    title_lines = title_lines[:2]

    y = 110
    for line in title_lines:
        draw.text((PAD_X, y), line, font=title_font, fill=TEXT_DARK)
        y += int(title_size * 1.2)

    y += 14
    draw.rectangle([PAD_X, y, PAD_X + 90, y + 6], fill=ACCENT)
    y += 50

    # Bullets — each wraps independently, font size fixed
    bullet_font = load_font("Regular", 30)
    line_height = 46
    max_bullet_width = W - 2 * PAD_X - 38
    for b in bullets:
        draw.ellipse([PAD_X, y + 13, PAD_X + 13, y + 26], fill=ACCENT)
        for line in wrap_text(draw, b, bullet_font, max_bullet_width):
            draw.text((PAD_X + 38, y), line, font=bullet_font, fill=TEXT_GRAY)
            y += line_height
        y += 18

    slide_font = load_font("Medium", 22)
    draw.text((W - 150, H - 50), f"{slide_number} / {total_slides}", font=slide_font, fill=TEXT_LIGHT_GRAY)
    progress_bar(draw, slide_number, total_slides)

    return img


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-slide-image")
async def generate_slide_image(request: GenerateSlideImageRequest):
    """
    Renders ONE slide as a PNG image (1280x720) and returns it directly.

    This is Step 2 of the video pipeline: confirm the visual design works
    well with real AI-generated content before assembling into video (Step 3).
    """
    try:
        if not request.title.strip():
            raise HTTPException(status_code=400, detail="title cannot be empty")
        if request.total_slides <= 0:
            raise HTTPException(status_code=400, detail="total_slides must be greater than 0")

        img = render_slide(
            title=request.title,
            bullets=request.bullets,
            module_label=request.module_label,
            slide_number=request.slide_number,
            total_slides=request.total_slides,
        )

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"SLIDE-IMAGE ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Slide image generation failed: {str(e)}")
