import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydub import AudioSegment
import numpy as np


CANVAS_SIZE = (1080, 1920)
IMAGE_SIZE = (1080, 1350)
IMAGE_RATIO = 4 / 5
IMAGE_TOP = 570
TEXT_BOX_BOTTOM = 570
GRADIENT_TOP = 570
GRADIENT_BOTTOM = 1020
FPS = 30
TITLE_SECONDS = 2.4

TEXT_SAFE_LEFT = 74
TEXT_SAFE_RIGHT = 74
TEXT_SAFE_TOP = 20
TEXT_SAFE_BOTTOM = CANVAS_SIZE[1] - TEXT_BOX_BOTTOM

BACKGROUND_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
DEFAULT_BG_MUSIC_GAIN_DB = -22.0
DEFAULT_BG_MUSIC_FADE_MS = 2500

REQUEST_TIMEOUT = 18
MIN_PUBLIC_DOMAIN_SCORE = 7.0

TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL = (
    "Traditional Catholic visual guardrails: use pre-1962 Catholic sacred art references, "
    "traditional vestments, Latin Mass-era devotional imagery, modest reverence, and timeless church interiors. "
    "Avoid modern liturgical settings, modern vestments, celebrity-like clergy portraits, contemporary church architecture, "
    "political symbols, caricature, satire, text, logos, and watermarks."
)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slugify(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return value or fallback


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_folder(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_timestamp(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))

    if millis >= 1000:
        whole += 1
        millis -= 1000

    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    sep = "," if srt else "."

    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def resolve_video_clips_folder(shorts_analysis_path: str | Path) -> Path:
    path = Path(shorts_analysis_path).expanduser().resolve()

    if path.is_dir():
        return path

    return path.parent


def resolve_output_paths(shorts_analysis_path: str | Path) -> Dict[str, Path]:
    video_clips_folder = resolve_video_clips_folder(shorts_analysis_path)

    paths = {
        "video_clips_folder": video_clips_folder,
        "shorts_analysis": video_clips_folder / "shorts_analysis.json",
        "shorts_manifest": video_clips_folder / "shorts_manifest.json",
        "image_candidates": video_clips_folder / "image_candidates.json",
        "upload_metadata": video_clips_folder / "upload_metadata.json",
        "images_dir": video_clips_folder / "images",
        "audio_dir": video_clips_folder / "audio",
        "videos_dir": video_clips_folder / "videos",
        "captions_dir": video_clips_folder / "captions",
        "artwork_credits": video_clips_folder / "artwork_credits.json",
    }

    for key in ["images_dir", "audio_dir", "videos_dir", "captions_dir"]:
        ensure_folder(paths[key])

    return paths


def get_selected_clips(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = [clip for clip in analysis.get("clips", []) if clip.get("selected")]

    if not selected:
        raise ValueError("No clips selected. Run Step #3 first and set selected clips to true.")

    selected.sort(key=lambda clip: int(clip.get("id", 0)))
    return selected


def get_clip_paths(clip: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Path]:
    clip_id = f"clip-{int(clip['id']):02d}"
    slug = slugify(clip.get("title", ""), clip_id)

    return {
        "image": paths["images_dir"] / f"{clip_id}-{slug}.jpg",
        "audio": paths["audio_dir"] / f"{clip_id}.mp3",
        "video": paths["videos_dir"] / f"{clip_id}-{slug}.mp4",
        "captions": paths["captions_dir"] / f"{clip_id}.srt",
        "image_meta": paths["images_dir"] / f"{clip_id}-{slug}.image.json",
    }


def build_search_terms(clip: Dict[str, Any]) -> List[str]:
    terms = []

    for value in clip.get("image_search_terms", []):
        value = clean_text(value)
        if value:
            terms.append(value)

    title = clean_text(clip.get("title", ""))
    image_idea = clean_text(clip.get("image_idea", ""))
    power_quote = clean_text(clip.get("power_quote", ""))

    for value in [title, image_idea, power_quote]:
        if value:
            terms.append(value)

    catholic_fallbacks = [
        f"{title} sacred art",
        f"{title} Renaissance painting",
        f"{title} Biblical painting",
        "Christ carrying the cross",
        "Crucifixion Renaissance painting",
        "Last Supper chalice",
        "Mass of Saint Gregory",
        "saint in prayer",
        "Good Samaritan painting",
    ]

    terms.extend(catholic_fallbacks)

    unique = []
    seen = set()

    for term in terms:
        term = clean_text(term)
        key = term.lower()

        if key and key not in seen:
            unique.append(term)
            seen.add(key)

    return unique[:12]


def score_artwork(candidate: Dict[str, Any], clip: Dict[str, Any], search_term: str) -> float:
    title = clean_text(candidate.get("title", "")).lower()
    artist = clean_text(candidate.get("artist", "")).lower()
    source = clean_text(candidate.get("source", "")).lower()
    term = clean_text(search_term).lower()

    clip_title = clean_text(clip.get("title", "")).lower()
    image_idea = clean_text(clip.get("image_idea", "")).lower()
    power_quote = clean_text(clip.get("power_quote", "")).lower()

    sacred_terms = [
        "christ",
        "jesus",
        "cross",
        "crucifixion",
        "virgin",
        "mary",
        "saint",
        "apostle",
        "mass",
        "eucharist",
        "chalice",
        "altar",
        "prayer",
        "angel",
        "samaritan",
        "martyr",
        "passion",
        "last supper",
        "sacrifice",
    ]

    bad_terms = [
        "modern",
        "poster",
        "photograph",
        "abstract",
        "installation",
        "fashion",
        "advertisement",
    ]

    score = 0.0

    if candidate.get("image_url"):
        score += 2.0

    if candidate.get("public_domain"):
        score += 3.0

    if source in {"the met", "art institute of chicago", "rijksmuseum", "national gallery of art"}:
        score += 0.7

    for word in sacred_terms:
        if word in title:
            score += 1.2
        if word in term:
            score += 0.5
        if word in image_idea:
            score += 0.6
        if word in power_quote:
            score += 0.4

    for word in re.findall(r"[a-zA-Z]+", clip_title):
        if len(word) > 4 and word in title:
            score += 0.5

    for word in re.findall(r"[a-zA-Z]+", term):
        if len(word) > 4 and word in title:
            score += 0.7

    if "unknown" not in artist and artist:
        score += 0.3

    for word in bad_terms:
        if word in title:
            score -= 2.0

    return round(score, 2)


def requests_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def search_met(term: str, clip: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
    results = []

    search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    search_data = requests_get_json(
        search_url,
        params={
            "q": term,
            "hasImages": "true",
        },
    )

    object_ids = (search_data or {}).get("objectIDs") or []

    for object_id in object_ids[:max_results]:
        object_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        obj = requests_get_json(object_url)

        if not obj:
            continue

        image_url = obj.get("primaryImage") or obj.get("primaryImageSmall")

        if not image_url:
            continue

        public_domain = bool(obj.get("isPublicDomain"))

        if not public_domain:
            continue

        candidate = {
            "source": "The Met",
            "title": clean_text(obj.get("title", "Untitled")),
            "artist": clean_text(obj.get("artistDisplayName", "Unknown artist")),
            "date": clean_text(obj.get("objectDate", "")),
            "source_url": clean_text(obj.get("objectURL", "")),
            "image_url": image_url,
            "license": "Public Domain / Open Access",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_artic(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    results = []

    search_url = "https://api.artic.edu/api/v1/artworks/search"
    data = requests_get_json(
        search_url,
        params={
            "q": term,
            "limit": max_results,
            "fields": "id,title,artist_display,image_id,is_public_domain,date_display,thumbnail",
            "query[term][is_public_domain]": "true",
        },
    )

    for item in (data or {}).get("data", []):
        image_id = item.get("image_id")

        if not image_id:
            continue

        if not item.get("is_public_domain"):
            continue

        image_url = f"https://www.artic.edu/iiif/2/{image_id}/full/1600,/0/default.jpg"

        candidate = {
            "source": "Art Institute of Chicago",
            "title": clean_text(item.get("title", "Untitled")),
            "artist": clean_text(item.get("artist_display", "Unknown artist")),
            "date": clean_text(item.get("date_display", "")),
            "source_url": f"https://www.artic.edu/artworks/{item.get('id')}",
            "image_url": image_url,
            "license": "Public Domain / CC0 when marked public domain",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_rijksmuseum(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Requires RIJKSMUSEUM_API_KEY in .env.

    Example:
    RIJKSMUSEUM_API_KEY=your_key_here
    """

    api_key = os.getenv("RIJKSMUSEUM_API_KEY", "").strip()

    if not api_key:
        return []

    results = []

    search_url = "https://www.rijksmuseum.nl/api/en/collection"
    data = requests_get_json(
        search_url,
        params={
            "key": api_key,
            "q": term,
            "imgonly": "True",
            "ps": max_results,
            "format": "json",
        },
    )

    for item in (data or {}).get("artObjects", []):
        web_image = item.get("webImage") or {}
        image_url = web_image.get("url")

        if not image_url:
            continue

        candidate = {
            "source": "Rijksmuseum",
            "title": clean_text(item.get("title", "Untitled")),
            "artist": clean_text(item.get("principalOrFirstMaker", "Unknown artist")),
            "date": "",
            "source_url": clean_text(item.get("links", {}).get("web", "")),
            "image_url": image_url,
            "license": "Rijksmuseum image/open data. Verify object rights if needed.",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_nga(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Placeholder hook for National Gallery of Art.

    Recommended Codex task:
    - Add NGA Open Data CSV / local dataset connector here.
    - Return the same candidate shape as the other sources.

    Keeping this as a hook prevents the workflow from breaking.
    """

    return []


def search_public_domain_art_for_clip(clip: Dict[str, Any]) -> Dict[str, Any]:
    all_candidates = []
    search_terms = build_search_terms(clip)

    for term in search_terms:
        all_candidates.extend(search_met(term, clip))
        all_candidates.extend(search_nga(term, clip))
        all_candidates.extend(search_artic(term, clip))
        all_candidates.extend(search_rijksmuseum(term, clip))

        strong_candidates = [c for c in all_candidates if c.get("score", 0) >= MIN_PUBLIC_DOMAIN_SCORE]

        if strong_candidates:
            break

    all_candidates.sort(key=lambda item: item.get("score", 0), reverse=True)

    return {
        "clip_id": clip.get("id"),
        "clip_title": clip.get("title"),
        "search_terms": search_terms,
        "candidates": all_candidates[:20],
        "best_candidate": all_candidates[0] if all_candidates else None,
    }


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    try:
        image = Image.open(BytesIO(response.content))
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Downloaded image was not readable: {url}") from exc

    return ImageOps.exif_transpose(image).convert("RGB")


def center_crop_to_ratio(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    current_ratio = width / height

    if current_ratio > ratio:
        new_width = int(height * ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))

    new_height = int(width / ratio)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def save_cropped_image(image: Image.Image, output_path: Path) -> Path:
    image = center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92, optimize=True, progressive=True)

    return output_path


def find_font(bold: bool = True, serif: bool = False) -> str:
    serif_candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]

    sans_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    candidates = serif_candidates if serif else sans_candidates

    if not bold:
        candidates = [p for p in candidates if "Bold" not in p] + candidates

    for path in candidates:
        if os.path.exists(path):
            return path

    raise RuntimeError("Could not find a usable TrueType font.")


def load_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(find_font(bold=bold, serif=serif), size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = re.split(r"\s+", str(text or "").strip())
    lines = []
    line = ""

    for word in words:
        trial = f"{line} {word}".strip()
        bbox = font.getbbox(trial)

        if bbox[2] - bbox[0] <= max_width:
            line = trial
            continue

        if line:
            lines.append(line)

        line = word

    if line:
        lines.append(line)

    return lines


def make_placeholder_image(clip: Dict[str, Any], output_path: Path) -> Path:
    image = Image.new("RGB", IMAGE_SIZE, (18, 18, 18))
    draw = ImageDraw.Draw(image)

    title_font = load_font(58, bold=True, serif=True)
    small_font = load_font(30, bold=False, serif=False)

    title = clean_text(clip.get("title", "Catholic Short")).upper()
    image_idea = clean_text(clip.get("image_idea", ""))

    lines = wrap_text(title, title_font, IMAGE_SIZE[0] - 120)[:4]

    y = 220

    for line in lines:
        bbox = title_font.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x + 2, y + 2), line, font=title_font, fill=(0, 0, 0))
        draw.text((x, y), line, font=title_font, fill=(235, 235, 235))
        y += 80

    small_lines = wrap_text(image_idea or "No image found", small_font, IMAGE_SIZE[0] - 160)[:4]
    y = IMAGE_SIZE[1] - 310

    for line in small_lines:
        bbox = small_font.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=small_font, fill=(180, 180, 180))
        y += 42

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)

    return output_path


def generate_ai_image_for_clip(clip: Dict[str, Any], output_path: Path) -> Optional[Path]:
    """
    Optional AI image fallback.

    Requires:
    OPENAI_API_KEY

    This function uses the current OpenAI Python client image API.
    If generation fails, it returns None and the app creates a placeholder.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = (
            "Create a 4:5 vertical image for a Catholic YouTube Short.\n\n"
            f"Title: {clip.get('title', '')}\n"
            f"Image idea: {clip.get('image_idea', '')}\n"
            f"Power quote: {clip.get('power_quote', '')}\n\n"
            "Use light watercolor with ink details, reverent sacred art tone, cinematic composition. "
            f"{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL} "
            "No text, no typography, no subtitles, no logo, no watermark."
        )

        model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=os.getenv("OPENAI_SHORTS_IMAGE_SIZE", "1024x1536"),
            n=1,
        )

        item = response.data[0]

        if getattr(item, "b64_json", None):
            import base64

            image_bytes = base64.b64decode(item.b64_json)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return save_cropped_image(image, output_path)

        if getattr(item, "url", None):
            image = download_image(item.url)
            return save_cropped_image(image, output_path)

        return None

    except Exception as exc:
        print(f"AI image generation failed for clip {clip.get('id')}: {exc}")
        return None


def ensure_image_for_clip(
    clip: Dict[str, Any],
    paths: Dict[str, Path],
    force_image: bool = False,
    allow_ai_fallback: bool = True,
) -> Dict[str, Any]:
    clip_paths = get_clip_paths(clip, paths)
    image_path = clip_paths["image"]
    image_meta_path = clip_paths["image_meta"]

    if image_path.exists() and image_meta_path.exists() and not force_image:
        return load_json(image_meta_path)

    print(f"Finding image for clip {clip.get('id')}: {clip.get('title')}")

    image_search_result = search_public_domain_art_for_clip(clip)
    best = image_search_result.get("best_candidate")

    if best and best.get("score", 0) >= MIN_PUBLIC_DOMAIN_SCORE:
        try:
            image = download_image(best["image_url"])
            save_cropped_image(image, image_path)

            meta = {
                "clip_id": clip.get("id"),
                "clip_title": clip.get("title"),
                "image_path": str(image_path),
                "image_source_type": "public_domain",
                "artwork": best,
                "all_candidates": image_search_result.get("candidates", []),
                "created_at": time.time(),
            }

            save_json(image_meta_path, meta)
            return meta

        except Exception as exc:
            print(f"Public-domain image failed for clip {clip.get('id')}: {exc}")

    if allow_ai_fallback:
        generated = generate_ai_image_for_clip(clip, image_path)

        if generated:
            meta = {
                "clip_id": clip.get("id"),
                "clip_title": clip.get("title"),
                "image_path": str(generated),
                "image_source_type": "ai_generated",
                "artwork": None,
                "all_candidates": image_search_result.get("candidates", []),
                "created_at": time.time(),
            }

            save_json(image_meta_path, meta)
            return meta

    make_placeholder_image(clip, image_path)

    meta = {
        "clip_id": clip.get("id"),
        "clip_title": clip.get("title"),
        "image_path": str(image_path),
        "image_source_type": "placeholder",
        "artwork": None,
        "all_candidates": image_search_result.get("candidates", []),
        "created_at": time.time(),
    }

    save_json(image_meta_path, meta)
    return meta


def make_background(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", CANVAS_SIZE, "black")
    canvas.paste(image, (0, IMAGE_TOP))

    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gradient_height = GRADIENT_BOTTOM - GRADIENT_TOP

    for offset in range(gradient_height):
        alpha = int(255 * (1 - ((offset + 1) / gradient_height)))
        y = GRADIENT_TOP + offset
        draw.line([(0, y), (CANVAS_SIZE[0], y)], fill=(0, 0, 0, alpha))

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def text_overlay(text: str, title: bool = False) -> Image.Image:
    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if title:
        text = str(text or "").upper()

    font_size = 68 if title else 62
    font = load_font(font_size, bold=True, serif=title)

    safe_left = TEXT_SAFE_LEFT
    safe_right = CANVAS_SIZE[0] - TEXT_SAFE_RIGHT
    safe_top = TEXT_SAFE_TOP
    safe_bottom = CANVAS_SIZE[1] - TEXT_SAFE_BOTTOM
    safe_width = safe_right - safe_left
    safe_height = safe_bottom - safe_top

    max_lines = 3
    lines = wrap_text(text, font, safe_width)[:max_lines]

    def metrics() -> Tuple[int, List[int], List[int]]:
        gap = int(font_size * 0.32)
        widths = []
        heights = []

        for line in lines:
            bbox = font.getbbox(line)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])

        return sum(heights) + gap * max(0, len(lines) - 1), widths, heights

    total_height, widths, heights = metrics()

    while (len(wrap_text(text, font, safe_width)) > max_lines or total_height > safe_height) and font_size > 38:
        font_size -= 4
        font = load_font(font_size, bold=True, serif=title)
        lines = wrap_text(text, font, safe_width)[:max_lines]
        total_height, widths, heights = metrics()

    line_gap = int(font_size * 0.32)
    y = safe_top + max(0, (safe_height - total_height) // 2)

    fill = (206, 24, 32, 255) if title else (255, 255, 255, 255)
    shadow = (0, 0, 0, 220)

    for line, width, height in zip(lines, widths, heights):
        x = safe_left + (safe_width - width) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_gap

    return overlay


def cut_audio_clip(source_audio: str | Path, start: float, end: float, output_path: Path) -> Path:
    source_audio = Path(source_audio).expanduser().resolve()

    if not source_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {source_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(source_audio)
    start_ms = int(float(start) * 1000)
    end_ms = int(float(end) * 1000)

    audio[start_ms:end_ms].export(output_path, format="mp3")

    return output_path


def list_background_audio_files(bg_audio_dir: str | Path) -> List[Path]:
    bg_audio_dir = Path(bg_audio_dir).expanduser().resolve()

    if not bg_audio_dir.exists() or not bg_audio_dir.is_dir():
        return []

    files = []

    for path in bg_audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() in BACKGROUND_AUDIO_EXTENSIONS:
            files.append(path)

    return sorted(files)


def loop_audio_to_duration(audio: AudioSegment, duration_ms: int) -> AudioSegment:
    if len(audio) <= 0:
        raise RuntimeError("Background audio file has no duration.")

    if len(audio) >= duration_ms:
        return audio[:duration_ms]

    repeats = (duration_ms // len(audio)) + 1
    return (audio * repeats)[:duration_ms]


def mix_background_music(
    speech_path: Path,
    bg_audio_files: List[Path],
    gain_db: float,
    fade_ms: int,
) -> Optional[Path]:
    if not bg_audio_files:
        return None

    bg_path = random.choice(bg_audio_files)

    speech = AudioSegment.from_file(speech_path)
    duration_ms = len(speech)

    if duration_ms <= 0:
        return None

    music = AudioSegment.from_file(bg_path)
    music = loop_audio_to_duration(music, duration_ms)
    music = music.set_frame_rate(speech.frame_rate).set_channels(speech.channels)
    music = music + float(gain_db)

    fade_ms = max(0, min(int(fade_ms), duration_ms // 2))

    if fade_ms:
        music = music.fade_in(fade_ms).fade_out(fade_ms)

    mixed = speech.overlay(music)
    mixed.export(speech_path, format="mp3")

    return bg_path


def create_basic_caption_groups(clip: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simple captions using source_text.
    Later Codex can replace this with word-level captions.
    """

    start = float(clip["start"])
    end = float(clip["end"])
    duration = end - start

    text = clean_text(clip.get("source_text", ""))

    if not text:
        text = clean_text(clip.get("power_quote", ""))

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [clean_text(s) for s in sentences if clean_text(s)]

    if not sentences:
        return []

    groups = []
    current_time = start
    seconds_per_group = max(2.0, min(4.0, duration / max(1, len(sentences))))

    for sentence in sentences:
        group_end = min(end, current_time + seconds_per_group)

        groups.append(
            {
                "start": round(current_time, 3),
                "end": round(group_end, 3),
                "text": sentence[:120],
            }
        )

        current_time = group_end

        if current_time >= end:
            break

    return groups


def write_srt(clip: Dict[str, Any], output_path: Path) -> Path:
    groups = clip.get("caption_groups") or create_basic_caption_groups(clip)

    lines = []
    clip_start = float(clip["start"])

    for index, group in enumerate(groups, 1):
        text = clean_text(group.get("text", ""))

        if not text:
            continue

        start = max(0.0, float(group["start"]) - clip_start)
        end = max(start + 0.2, float(group["end"]) - clip_start)

        lines.extend(
            [
                str(index),
                f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}",
                text,
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return output_path


def render_video_clip(
    clip: Dict[str, Any],
    source_audio: str | Path,
    paths: Dict[str, Path],
    bg_audio_files: Optional[List[Path]] = None,
    bg_music_gain_db: float = DEFAULT_BG_MUSIC_GAIN_DB,
    bg_music_fade_ms: int = DEFAULT_BG_MUSIC_FADE_MS,
    force: bool = False,
) -> Dict[str, Any]:
    clip_paths = get_clip_paths(clip, paths)

    image_path = clip_paths["image"]
    audio_path = clip_paths["audio"]
    video_path = clip_paths["video"]
    captions_path = clip_paths["captions"]

    if video_path.exists() and not force:
        clip["output_paths"] = {k: str(v) for k, v in clip_paths.items()}
        return clip

    if not image_path.exists():
        raise FileNotFoundError(f"Image missing for clip {clip.get('id')}: {image_path}")

    print(f"Rendering clip {clip.get('id')}: {clip.get('title')}")

    cut_audio_clip(
        source_audio=source_audio,
        start=float(clip["start"]),
        end=float(clip["end"]),
        output_path=audio_path,
    )

    bg_audio_path = None

    if bg_audio_files:
        bg_audio_path = mix_background_music(
            speech_path=audio_path,
            bg_audio_files=bg_audio_files,
            gain_db=bg_music_gain_db,
            fade_ms=bg_music_fade_ms,
        )

    clip["caption_groups"] = create_basic_caption_groups(clip)
    write_srt(clip, captions_path)

    duration = float(clip["end"]) - float(clip["start"])

    background = np.array(make_background(image_path))
    layers = [ImageClip(background, duration=duration)]

    title_duration = min(TITLE_SECONDS, max(1.4, duration * 0.18))

    title_img = np.array(text_overlay(clip.get("title", ""), title=True))
    layers.append(ImageClip(title_img, duration=title_duration).with_start(0))

    clip_start = float(clip["start"])

    for group in clip.get("caption_groups", []):
        text = clean_text(group.get("text", ""))

        if not text:
            continue

        rel_start = max(0.0, float(group["start"]) - clip_start)
        rel_end = min(duration, max(rel_start + 0.3, float(group["end"]) - clip_start))

        if rel_end <= title_duration:
            continue

        rel_start = max(rel_start, title_duration)
        caption_img = np.array(text_overlay(text, title=False))

        layers.append(
            ImageClip(caption_img, duration=rel_end - rel_start).with_start(rel_start)
        )

    audio = AudioFileClip(str(audio_path))
    video = CompositeVideoClip(layers, size=CANVAS_SIZE).with_audio(audio).with_duration(duration)

    video_path.parent.mkdir(parents=True, exist_ok=True)

    video.write_videofile(
        str(video_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        threads=4,
        logger="bar",
    )

    audio.close()
    video.close()

    clip["background_audio"] = str(bg_audio_path) if bg_audio_path else ""
    clip["output_paths"] = {k: str(v) for k, v in clip_paths.items()}

    return clip


def write_manifest_and_metadata(
    selected_clips: List[Dict[str, Any]],
    image_meta_by_clip_id: Dict[int, Dict[str, Any]],
    paths: Dict[str, Path],
    source_audio: str | Path,
) -> None:
    manifest = {
        "version": 1,
        "source_audio": str(Path(source_audio).expanduser().resolve()),
        "clips": selected_clips,
    }

    upload_metadata = {
        "clips": [],
    }

    artwork_credits = {
        "clips": [],
    }

    for clip in selected_clips:
        clip_id = int(clip["id"])
        image_meta = image_meta_by_clip_id.get(clip_id, {})
        artwork = image_meta.get("artwork")

        upload_metadata["clips"].append(
            {
                "id": clip_id,
                "title": clip.get("title", ""),
                "description": clip.get("why_it_works", ""),
                "power_quote": clip.get("power_quote", ""),
                "tags": [
                    "Catholic homily",
                    "Traditional Catholic",
                    "Catholic Shorts",
                    "Latin Mass",
                ],
                "start": clip.get("start"),
                "end": clip.get("end"),
                "duration": clip.get("length_seconds"),
                "video_path": clip.get("output_paths", {}).get("video", ""),
                "thumbnail_path": clip.get("output_paths", {}).get("image", ""),
                "captions_path": clip.get("output_paths", {}).get("captions", ""),
            }
        )

        artwork_credits["clips"].append(
            {
                "id": clip_id,
                "clip_title": clip.get("title", ""),
                "image_source_type": image_meta.get("image_source_type", ""),
                "artwork_title": (artwork or {}).get("title", ""),
                "artist": (artwork or {}).get("artist", ""),
                "source": (artwork or {}).get("source", ""),
                "source_url": (artwork or {}).get("source_url", ""),
                "license": (artwork or {}).get("license", ""),
                "image_path": image_meta.get("image_path", ""),
            }
        )

    save_json(paths["shorts_manifest"], manifest)
    save_json(paths["upload_metadata"], upload_metadata)
    save_json(paths["artwork_credits"], artwork_credits)

    print(f"Saved manifest: {paths['shorts_manifest']}")
    print(f"Saved upload metadata: {paths['upload_metadata']}")
    print(f"Saved artwork credits: {paths['artwork_credits']}")


def run_step_04_render_with_images(
    shorts_analysis_path: str | Path,
    source_audio: str | Path,
    bg_audio_dir: Optional[str | Path] = None,
    force_images: bool = False,
    force_render: bool = False,
    allow_ai_fallback: bool = True,
    max_workers: int = 3,
) -> List[Dict[str, Any]]:
    """
    Main Step #4 function.

    Flow:
    1. Load selected clips from shorts_analysis.json.
    2. Get first image first.
    3. Start rendering first video.
    4. While first video renders, search/generate images for the rest.
    5. Render remaining clips.
    6. Save manifest, metadata, and artwork credits.
    """

    load_dotenv()

    paths = resolve_output_paths(shorts_analysis_path)
    analysis = load_json(paths["shorts_analysis"])
    selected_clips = get_selected_clips(analysis)

    image_meta_by_clip_id: Dict[int, Dict[str, Any]] = {}

    bg_audio_files = []

    if bg_audio_dir:
        bg_audio_files = list_background_audio_files(bg_audio_dir)

    first_clip = selected_clips[0]
    remaining_clips = selected_clips[1:]

    print()
    print("Step #4: Render selected Shorts with images")
    print("-" * 80)
    print(f"Selected clips: {len(selected_clips)}")
    print(f"First clip: {first_clip.get('id')} - {first_clip.get('title')}")
    print("-" * 80)
    print()

    first_meta = ensure_image_for_clip(
        clip=first_clip,
        paths=paths,
        force_image=force_images,
        allow_ai_fallback=allow_ai_fallback,
    )

    image_meta_by_clip_id[int(first_clip["id"])] = first_meta

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_futures = {
            executor.submit(
                ensure_image_for_clip,
                clip,
                paths,
                force_images,
                allow_ai_fallback,
            ): clip
            for clip in remaining_clips
        }

        first_clip = render_video_clip(
            clip=first_clip,
            source_audio=source_audio,
            paths=paths,
            bg_audio_files=bg_audio_files,
            force=force_render,
        )

        for future in concurrent.futures.as_completed(image_futures):
            clip = image_futures[future]

            try:
                meta = future.result()
                image_meta_by_clip_id[int(clip["id"])] = meta
                print(f"Image ready for clip {clip.get('id')}: {clip.get('title')}")
            except Exception as exc:
                print(f"Image failed for clip {clip.get('id')}: {exc}")
                meta = {
                    "clip_id": clip.get("id"),
                    "clip_title": clip.get("title"),
                    "image_source_type": "failed",
                    "image_path": "",
                    "artwork": None,
                }
                image_meta_by_clip_id[int(clip["id"])] = meta

    rendered_clips = [first_clip]

    for clip in remaining_clips:
        rendered = render_video_clip(
            clip=clip,
            source_audio=source_audio,
            paths=paths,
            bg_audio_files=bg_audio_files,
            force=force_render,
        )
        rendered_clips.append(rendered)

    write_manifest_and_metadata(
        selected_clips=rendered_clips,
        image_meta_by_clip_id=image_meta_by_clip_id,
        paths=paths,
        source_audio=source_audio,
    )

    print()
    print("Finished Step #4")
    print("-" * 80)

    for clip in rendered_clips:
        print(f"{clip.get('id')}. {clip.get('title')} -> {clip.get('output_paths', {}).get('video', '')}")

    print("-" * 80)
    print()

    return rendered_clips


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step #4: Render selected Shorts with public-domain images first.")
    parser.add_argument(
        "shorts_analysis",
        help="Path to Video Clips/shorts_analysis.json or the Video Clips folder.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the full homily audio file.",
    )
    parser.add_argument(
        "--bg-audio-dir",
        default=None,
        help="Optional folder containing background music files.",
    )
    parser.add_argument(
        "--force-images",
        action="store_true",
        help="Force image search/generation even if image files already exist.",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Force video rendering even if video files already exist.",
    )
    parser.add_argument(
        "--no-ai-fallback",
        action="store_true",
        help="Do not generate AI images if public-domain art is not found.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of parallel image lookup/generation workers.",
    )

    args = parser.parse_args()

    run_step_04_render_with_images(
        shorts_analysis_path=args.shorts_analysis,
        source_audio=args.audio,
        bg_audio_dir=args.bg_audio_dir,
        force_images=args.force_images,
        force_render=args.force_render,
        allow_ai_fallback=not args.no_ai_fallback,
        max_workers=args.max_workers,
    )