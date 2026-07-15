import os
import io
import uuid
import json
import httpx
import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.chart import XL_CHART_TYPE
from pptx.chart.data import ChartData
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
UNSPLASH_KEY     = os.getenv("UNSPLASH_ACCESS_KEY", "")
R2_ACCOUNT_ID    = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY    = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET        = os.getenv("R2_BUCKET_NAME", "knowledge-videos")
R2_PUBLIC_URL    = os.getenv("R2_PUBLIC_URL", "https://pub-222a48f277db4cb6b9fbdda27a672cc5.r2.dev")

r2 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto",
    config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
)

# ── Color palettes ─────────────────────────────────────────────────────────────
PALETTES = {
    "midnight_executive": {
        "primary": "1E2761", "secondary": "CADCFC", "accent": "FFFFFF",
        "dark_text": "1A1A2E", "light_text": "F5F5F5",
    },
    "ocean_gradient": {
        "primary": "065A82", "secondary": "1C7293", "accent": "FFFFFF",
        "dark_text": "0D2137", "light_text": "F0F8FF",
    },
    "charcoal_minimal": {
        "primary": "36454F", "secondary": "F2F2F2", "accent": "212121",
        "dark_text": "1C1C1C", "light_text": "FAFAFA",
    },
}


def hex_to_rgb(hex_str: str) -> RGBColor:
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class BrandingSettings(BaseModel):
    palette:       Optional[str] = "midnight_executive"
    primary_color: Optional[str] = None
    company_name:  Optional[str] = ""
    font:          Optional[str] = "Calibri"


class GeneratePresentationRequest(BaseModel):
    query:          str
    chunks:         list[str]
    tone:           Optional[str]  = "professional"
    num_slides:     Optional[int]  = 8
    enable_charts:  Optional[bool] = True
    enable_images:  Optional[bool] = True
    branding:       Optional[BrandingSettings] = BrandingSettings()


class EditSlideRequest(BaseModel):
    slide:       dict
    instruction: str
    context:     Optional[str] = ""


class ExportPptxRequest(BaseModel):
    slides:    list[dict]
    title:     str
    branding:  Optional[BrandingSettings] = BrandingSettings()


def add_text_box(slide, text, left, top, width, height,
                 font_name="Calibri", font_size=14, bold=False,
                 color="1A1A2E", align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = hex_to_rgb(color)
    return txBox


def set_slide_background(slide, hex_color: str):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(hex_color)


def upload_to_r2(data: bytes, key: str, content_type: str) -> str:
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"{R2_PUBLIC_URL}/{key}"


def headline_fit(headline: str, base: int = 32):
    """Returns (font_size, box_height_inches, num_lines) sized to avoid overflow."""
    n = len(headline)
    if n < 50:
        return base, 1, Inches(0.65)
    elif n < 90:
        return base - 7, 2, Inches(1.15)
    else:
        return base - 12, 3, Inches(1.6)


SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def render_title_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    set_slide_background(slide, palette["primary"])

    title    = slide_data.get("title", "")
    subtitle = slide_data.get("subtitle", "")
    company  = slide_data.get("company", "")

    add_text_box(slide, title, Inches(1), Inches(2.2), Inches(11.3), Inches(1.8),
                 font_name=font, font_size=42, bold=True,
                 color=palette["accent"], align=PP_ALIGN.CENTER)

    if subtitle:
        add_text_box(slide, subtitle, Inches(1), Inches(4.1), Inches(11.3), Inches(0.9),
                     font_name=font, font_size=20,
                     color=palette["secondary"], align=PP_ALIGN.CENTER)

    if company:
        add_text_box(slide, company, Inches(1), Inches(5.5), Inches(11.3), Inches(0.5),
                     font_name=font, font_size=14,
                     color=palette["secondary"], align=PP_ALIGN.CENTER)


def render_executive_summary(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)

    headline = slide_data.get("headline", "Executive Summary")
    bullets  = slide_data.get("bullets", [])[:3]

    h_size, h_lines, h_height = headline_fit(headline, base=32)

    add_text_box(slide, headline, Inches(0.6), Inches(0.4), Inches(12.1), h_height,
                 font_name=font, font_size=h_size, bold=True, color=palette["dark_text"])

    y = Inches(0.4) + h_height + Inches(0.45)

    for bullet in bullets:
        circle = slide.shapes.add_shape(9, Inches(0.6), y + Inches(0.1), Inches(0.26), Inches(0.26))
        circle.fill.solid()
        circle.fill.fore_color.rgb = hex_to_rgb(palette["primary"])
        circle.line.fill.background()

        add_text_box(slide, bullet, Inches(1.05), y, Inches(11.3), Inches(0.85),
                     font_name=font, font_size=15, color=palette["dark_text"])
        y += Inches(1.25)


def render_content_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)

    headline = slide_data.get("headline", "")
    bullets  = slide_data.get("bullets", [])

    h_size, h_lines, h_height = headline_fit(headline, base=30)

    add_text_box(slide, headline, Inches(0.6), Inches(0.35), Inches(12.1), h_height,
                 font_name=font, font_size=h_size, bold=True, color=palette["dark_text"])

    y = Inches(0.35) + h_height + Inches(0.35)
    for bullet in bullets[:5]:
        add_text_box(slide, f"•  {bullet}", Inches(0.8), y, Inches(11.5), Inches(0.8),
                     font_name=font, font_size=15, color=palette["dark_text"])
        y += Inches(0.95)


def render_chart_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)

    headline   = slide_data.get("headline", "")
    chart_data = slide_data.get("chart", {})
    insight    = slide_data.get("insight", "")

    h_size, h_lines, h_height = headline_fit(headline, base=28)

    add_text_box(slide, headline, Inches(0.6), Inches(0.3), Inches(12.1), h_height,
                 font_name=font, font_size=h_size, bold=True, color=palette["dark_text"])

    chart_top = Inches(0.3) + h_height + Inches(0.2)

    cd = ChartData()
    labels   = chart_data.get("labels", ["Q1", "Q2", "Q3", "Q4"])
    datasets = chart_data.get("datasets", [{"label": "Data", "data": [0, 0, 0, 0]}])
    cd.categories = labels
    for ds in datasets:
        data_points = [v if v is not None else 0 for v in ds.get("data", [])]
        cd.add_series(ds.get("label", "Series"), data_points)

    chart_type_map = {
        "bar": XL_CHART_TYPE.BAR_CLUSTERED,
        "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
        "line": XL_CHART_TYPE.LINE,
        "pie": XL_CHART_TYPE.PIE,
    }
    xl_type = chart_type_map.get(chart_data.get("type", "column"), XL_CHART_TYPE.COLUMN_CLUSTERED)

    available_h = SLIDE_H - chart_top - Inches(0.3)
    slide.shapes.add_chart(xl_type, Inches(0.6), chart_top, Inches(8.3), available_h, cd)

    if insight:
        add_text_box(slide, f"Insight: {insight}", Inches(9.2), chart_top + Inches(0.3),
                     Inches(3.6), Inches(2.5), font_name=font, font_size=14, italic=True,
                     color=palette["primary"])


def render_big_stat_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)

    headline = slide_data.get("headline", "")
    stats    = slide_data.get("stats", [])[:3]

    h_size, h_lines, h_height = headline_fit(headline, base=28)

    add_text_box(slide, headline, Inches(0.6), Inches(0.3), Inches(12.1), h_height,
                 font_name=font, font_size=h_size, bold=True, color=palette["dark_text"])

    col_w = Inches(4)
    positions = [Inches(0.5), Inches(4.7), Inches(8.9)]
    stat_top = Inches(0.3) + h_height + Inches(0.7)

    for i, stat in enumerate(stats):
        x = positions[i]
        add_text_box(slide, stat.get("value", ""), x, stat_top, col_w, Inches(1.8),
                     font_name=font, font_size=56, bold=True,
                     color=palette["primary"], align=PP_ALIGN.CENTER)
        add_text_box(slide, stat.get("label", ""), x, stat_top + Inches(1.9), col_w, Inches(0.6),
                     font_name=font, font_size=15, color=palette["dark_text"], align=PP_ALIGN.CENTER)
        if stat.get("sublabel"):
            add_text_box(slide, stat.get("sublabel"), x, stat_top + Inches(2.55), col_w, Inches(0.5),
                         font_name=font, font_size=12, italic=True, color="888888", align=PP_ALIGN.CENTER)


def render_two_column_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)

    headline    = slide_data.get("headline", "")
    left_points = slide_data.get("left", [])
    right_text  = slide_data.get("right", "")

    h_size, h_lines, h_height = headline_fit(headline, base=28)

    add_text_box(slide, headline, Inches(0.6), Inches(0.3), Inches(12.1), h_height,
                 font_name=font, font_size=h_size, bold=True, color=palette["dark_text"])

    y = Inches(0.3) + h_height + Inches(0.35)
    for pt in left_points[:5]:
        add_text_box(slide, f"•  {pt}", Inches(0.6), y, Inches(5.9), Inches(0.8),
                     font_name=font, font_size=14, color=palette["dark_text"])
        y += Inches(0.95)

    if right_text:
        box_top = Inches(0.3) + h_height + Inches(0.35)
        box = slide.shapes.add_shape(1, Inches(7.2), box_top, Inches(5.7),
                                     SLIDE_H - box_top - Inches(0.4))
        box.fill.solid()
        box.fill.fore_color.rgb = hex_to_rgb(palette["secondary"])
        box.line.fill.background()

        add_text_box(slide, right_text, Inches(7.5), box_top + Inches(0.3), Inches(5.2), Inches(4.5),
                     font_name=font, font_size=14, color=palette["dark_text"])


def render_closing_slide(prs, slide_data, palette, font):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    set_slide_background(slide, palette["primary"])

    add_text_box(slide, slide_data.get("title", "Thank You"), Inches(1), Inches(2.2),
                 Inches(11.3), Inches(1.5), font_name=font, font_size=40, bold=True,
                 color=palette["accent"], align=PP_ALIGN.CENTER)

    sources = slide_data.get("sources", [])
    if sources:
        src_text = "Sources: " + " · ".join(sources)
        add_text_box(slide, src_text, Inches(1), Inches(5.8), Inches(11.3), Inches(0.6),
                     font_name=font, font_size=12, italic=True,
                     color=palette["secondary"], align=PP_ALIGN.CENTER)


SLIDE_RENDERERS = {
    "title": render_title_slide,
    "executive_summary": render_executive_summary,
    "content": render_content_slide,
    "bullets": render_content_slide,
    "chart": render_chart_slide,
    "big_stat": render_big_stat_slide,
    "two_column": render_two_column_slide,
    "closing": render_closing_slide,
}


@router.post("/generate-presentation")
async def generate_presentation(request: GeneratePresentationRequest):
    try:
        context = "\n\n---\n\n".join(request.chunks[:12])
        slide_types_note = ""
        if request.enable_charts:
            slide_types_note += ' Include at least 1-2 "chart" slides where data exists.'
        if not request.enable_images:
            slide_types_note += ' Do not include has_image fields.'

        system_prompt = f"""You are an expert presentation designer creating executive-level slides.
Tone: {request.tone}. Target audience: CEOs and senior managers.
Create {request.num_slides} slides maximum.{slide_types_note}

CRITICAL RULES:
- Headlines must be SHORT — maximum 8-10 words. Long headlines break slide layouts.
- Put the detailed insight in bullets/content, not in the headline.
- Vary slide types — do not repeat the same layout twice in a row.
- Use ONLY data found in the provided context — mark estimated data with "(est.)"
- Charts must have realistic data arrays matching the labels length.

Respond ONLY with valid JSON, no markdown fences:
{{
  "title": "presentation title",
  "slides": [
    {{"index": 0, "type": "title", "title": "string", "subtitle": "string"}},
    {{"index": 1, "type": "executive_summary", "headline": "short headline, max 8 words", "bullets": ["string", "string", "string"]}},
    {{"index": 2, "type": "big_stat", "headline": "short headline", "stats": [{{"value": "23%", "label": "Revenue Growth", "sublabel": "vs last quarter"}}]}},
    {{"index": 3, "type": "chart", "headline": "short headline", "chart": {{"type": "column", "labels": ["Q1","Q2"], "datasets": [{{"label": "2025", "data": [10,20]}}]}}, "insight": "one sentence"}},
    {{"index": 4, "type": "two_column", "headline": "short headline", "left": ["point 1"], "right": "supporting paragraph"}},
    {{"index": 5, "type": "content", "headline": "short headline", "bullets": ["string"]}},
    {{"index": 6, "type": "closing", "title": "Thank You", "sources": ["doc.pdf"]}}
  ]
}}"""

        user_prompt = f"""Create a {request.tone} presentation about: "{request.query}"

Source content:
{context}

Generate {request.num_slides} slides. Headlines must be short (max 8-10 words). Respond with JSON only."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            max_tokens=3000,
            temperature=0.3
        )

        data = json.loads(response.choices[0].message.content)
        return {"title": data.get("title", request.query), "slides": data.get("slides", []), "query": request.query}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        print(f"GENERATE-PRESENTATION ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Presentation generation failed: {str(e)}")


@router.post("/edit-slide")
async def edit_slide(request: EditSlideRequest):
    try:
        system_prompt = """You are editing a single presentation slide based on a user instruction.
Return ONLY the updated slide JSON with the same structure. Keep the same index and type unless asked to change it.
IMPORTANT: Keep headlines short — max 8-10 words — to avoid layout overflow.
Respond with valid JSON only, no markdown."""

        user_prompt = f"""Current slide:
{json.dumps(request.slide, indent=2)}

User instruction: {request.instruction}
Context: {request.context}

Return the updated slide JSON only."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.4
        )
        return json.loads(response.choices[0].message.content)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Slide edit failed: {str(e)}")


@router.post("/export-pptx")
async def export_pptx(request: ExportPptxRequest):
    try:
        branding = request.branding or BrandingSettings()
        palette = PALETTES.get(branding.palette or "midnight_executive", PALETTES["midnight_executive"]).copy()
        if branding.primary_color:
            palette["primary"] = branding.primary_color.lstrip("#")
        font = branding.font or "Calibri"

        prs = Presentation()
        prs.slide_width  = SLIDE_W
        prs.slide_height = SLIDE_H

        for slide_data in sorted(request.slides, key=lambda s: s.get("index", 0)):
            renderer = SLIDE_RENDERERS.get(slide_data.get("type", "content"), render_content_slide)
            renderer(prs, slide_data, palette, font)

        buf = io.BytesIO()
        prs.save(buf)
        buf.seek(0)
        pptx_bytes = buf.getvalue()

        safe_title = request.title.replace(" ", "_")[:50]
        key = f"presentations/{uuid.uuid4()}/{safe_title}.pptx"
        url = upload_to_r2(pptx_bytes, key,
                          "application/vnd.openxmlformats-officedocument.presentationml.presentation")

        print(f"[export-pptx] Uploaded: {key} ({len(pptx_bytes)/1024:.1f}KB)")

        return {"success": True, "pptx_url": url, "slide_count": len(request.slides),
                "size_kb": round(len(pptx_bytes) / 1024, 1)}

    except Exception as e:
        import traceback
        print(f"EXPORT-PPTX ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
