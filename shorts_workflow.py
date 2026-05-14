import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import frontmatter
import numpy as np
from dotenv import load_dotenv
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydub import AudioSegment

from audio_clip import clip_audio_segment
from thumbnail_generator import (
    DEFAULT_IMAGE_MODEL,
    _center_crop_to_ratio,
    _generate_image_with_compatible_kwargs,
    _image_bytes_from_response,
    _openai_client,
    _supported_image_kwargs,
)


CANVAS_SIZE = (1080, 1920)
IMAGE_SIZE = (1080, 1350)
IMAGE_RATIO = 4 / 5
GRADIENT_TOP = 900
GRADIENT_BOTTOM = 1350
BOTTOM_TOP = 1350
FPS = 30
MIN_SHORT_SECONDS = 15.0
MAX_SHORT_SECONDS = 90.0
DEFAULT_MAX_CLIPS = 7
DEFAULT_MIN_CLIPS = 3
TITLE_SECONDS = 2.4
TEXT_SAFE_LEFT = 64
TEXT_SAFE_RIGHT = 270
TEXT_SAFE_TOP = BOTTOM_TOP + 44
TEXT_SAFE_BOTTOM = 310
BACKGROUND_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
DEFAULT_BG_MUSIC_GAIN_DB = -22.0
DEFAULT_BG_MUSIC_FADE_MS = 2500
TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL = (
    "Traditional Catholic visual guardrails: use pre-1962 Catholic sacred art references, "
    "traditional vestments, Latin Mass-era devotional imagery, modest reverence, and timeless church interiors. "
    "Do not depict, resemble, reference, or evoke any pope, papal portrait, papal coat of arms, "
    "or recognizable papal figure after the year 1962. Avoid modern liturgical settings, modern vestments, "
    "celebrity-like clergy portraits, contemporary church architecture, political symbols, caricature, or satire."
)


def clean_path(path: str) -> str:
    return str(path or "").strip().strip('"').strip("'")


def slugify(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value or "").strip("-").lower()
    return value or fallback


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


def find_font(bold: bool = True, serif: bool = False) -> str:
    serif_candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
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
    raise RuntimeError("Could not find a usable TrueType font for Shorts rendering.")


def load_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(find_font(bold=bold, serif=serif), size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = re.split(r"\s+", str(text or "").strip())
    lines: List[str] = []
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


def resolve_homily_root(source: str) -> str:
    source = os.path.abspath(os.path.expanduser(clean_path(source)))
    if not os.path.exists(source):
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if os.path.isdir(source):
        base = os.path.basename(source)
        if base == "working":
            return os.path.dirname(source)
        if os.path.exists(os.path.join(source, "working", "video_script.json")):
            return source
        if os.path.exists(os.path.join(source, "video_script.json")):
            return os.path.dirname(source)
        raise FileNotFoundError(
            "Could not find working/video_script.json from folder: " + source
        )

    parent = os.path.dirname(source)
    if os.path.basename(parent) == "working":
        return os.path.dirname(parent)
    if os.path.exists(os.path.join(parent, "working", "video_script.json")):
        return parent
    if os.path.exists(os.path.join(os.path.dirname(parent), "working", "video_script.json")):
        return os.path.dirname(parent)
    raise FileNotFoundError(
        "Could not resolve homily root from file. Expected a file inside a completed homily folder: "
        + source
    )


def resolve_paths(source: str) -> Dict[str, str]:
    root = resolve_homily_root(source)
    working_dir = os.path.join(root, "working")
    final_dir = os.path.join(root, "final")
    clips_dir = os.path.join(root, "Video Clips")
    video_script = os.path.join(working_dir, "video_script.json")
    audio = os.path.join(working_dir, "homily_final.mp3")

    if not os.path.exists(video_script):
        raise FileNotFoundError(f"Missing video script: {video_script}")
    if not os.path.exists(audio):
        raise FileNotFoundError(f"Missing final homily audio: {audio}")

    mdx_path = ""
    if os.path.isdir(final_dir):
        mdx_candidates = []
        for path, _, filenames in os.walk(final_dir):
            for filename in filenames:
                if filename.lower().endswith(".mdx"):
                    mdx_candidates.append(os.path.join(path, filename))
        if mdx_candidates:
            mdx_candidates.sort(key=lambda p: (-os.path.getmtime(p), p))
            mdx_path = mdx_candidates[0]

    return {
        "root": root,
        "working_dir": working_dir,
        "final_dir": final_dir,
        "clips_dir": clips_dir,
        "video_script": video_script,
        "audio": audio,
        "mdx": mdx_path,
        "manifest": os.path.join(clips_dir, "shorts_manifest.json"),
        "upload_metadata": os.path.join(clips_dir, "upload_metadata.json"),
        "images_dir": os.path.join(clips_dir, "images"),
        "audio_dir": os.path.join(clips_dir, "audio"),
        "videos_dir": os.path.join(clips_dir, "videos"),
        "captions_dir": os.path.join(clips_dir, "captions"),
    }


def default_bg_audio_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_audio_files")


def ensure_output_dirs(paths: Dict[str, str]) -> None:
    for key in ["clips_dir", "images_dir", "audio_dir", "videos_dir", "captions_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def load_video_script(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("segments"):
        raise ValueError(f"Video script has no segments: {path}")
    return data


def load_mdx_context(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        post = frontmatter.load(path)
    except Exception:
        return {}
    return {
        "title": post.metadata.get("title") or "",
        "description": post.metadata.get("description") or "",
        "thumbnail_idea": post.metadata.get("thumbnail_idea") or "",
        "keywords": post.metadata.get("keywords") or "",
        "chapters": post.metadata.get("chapters") or [],
        "shorts": post.metadata.get("shorts") or [],
    }


def normalize_word(word: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "word": str(word.get("word") or word.get("text") or "").strip(),
        "start": float(word.get("start", 0.0) or 0.0),
        "end": float(word.get("end", word.get("start", 0.0)) or 0.0),
    }


def collect_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            normalized = normalize_word(word)
            if normalized["word"]:
                words.append(normalized)
    return words


def segment_units(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments, 1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        units.append(
            {
                "index": index,
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "text": re.sub(r"\s+", " ", text),
            }
        )
    return units


def build_timed_transcript(units: List[Dict[str, Any]], max_chars: int = 52000) -> str:
    lines = []
    for unit in units:
        lines.append(
            f'{unit["index"]:04d} [{format_timestamp(unit["start"])}-{format_timestamp(unit["end"])}] {unit["text"]}'
        )
    transcript = "\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    head = transcript[: max_chars // 2]
    tail = transcript[-max_chars // 2 :]
    return head.rstrip() + "\n\n[...middle omitted for prompt length...]\n\n" + tail.lstrip()


def openai_select_clips(
    units: List[Dict[str, Any]],
    context: Dict[str, Any],
    min_clips: int,
    max_clips: int,
) -> List[Dict[str, Any]]:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or OPENAI_KEY must be set for OpenAI clip selection.")

    model = os.getenv("OPENAI_SHORTS_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)
    transcript = build_timed_transcript(units)
    context_json = json.dumps(context, ensure_ascii=False)[:6000]
    prompt = f"""
You are a Catholic homily editor creating YouTube Shorts.

Your job is NOT to merely cut the homily into small chunks.
Your job is to find the best self-contained moments and shape them into strong Shorts.

Return ONLY valid JSON:
{{
  "clips": [
    {{
      "title": "<original YouTube Shorts title, not a transcript line, <=70 chars>",
      "mdx_section_title": "<section-style heading, <=70 chars>",
      "theme": "<one sentence explaining why this clip is worth watching>",
      "clip_type": "story|teaching|warning|exhortation|reflection",
      "story_arc": "<beginning, turn, and payoff if this is a story; otherwise empty string>",
      "start_segment": <integer>,
      "end_segment": <integer>,
      "display_words": "<best concise quote or summary text for on-screen display, <=260 chars>",
      "keywords": ["keyword", "keyword"]
    }}
  ]
}}

Selection rules:
- Select no fewer than {min_clips} clips if the transcript has enough strong material.
- Select no more than {max_clips} clips.
- Each clip must be 30 to 90 seconds when possible.
- 15 to 29 seconds is allowed only if the moment is exceptionally strong and complete.
- Each clip must be a complete standalone teaching, warning, exhortation, reflection, story, or spiritual lesson.
- If a story, anecdote, example, or testimony appears, treat the ENTIRE story as ONE clip when it fits under 90 seconds.
- Do NOT split one story into multiple Shorts unless the full story is longer than 90 seconds.
- A story clip must include the setup, the key action or conflict, and the spiritual payoff.
- If a story is followed immediately by the preacher explaining why the story matters, include that explanation if the total clip remains under 90 seconds.
- Prefer clips with strong spiritual value: clear doctrine, practical moral application, emotional movement, repentance, conversion, warning, hope, or encouragement.
- Avoid announcements, housekeeping, readings, or context that cannot stand alone.
- Use non-overlapping clips.
- Use start_segment and end_segment numbers from the transcript.
- Choose start_segment early enough that the viewer understands what is happening.
- Choose end_segment late enough that the viewer receives the payoff.
- Titles must be original editorial titles, not the first words of the transcript.
- Bad title example: "There Is An Anecdote From The Life"
- Good title example: "One Sentence That Saved Her Soul"
- Good title example: "The Priest Who Gave Hope to a Dying Woman"
- Good title example: "Compassion Can Reach a Soul Years Later"
- Do not invent a quote. display_words can be a concise phrase if a verbatim quote would be too long.
- Do not select several clips from the same story. Pick the best full version of that story instead.

Narrative detection:
Look for phrases such as:
- "There is a story..."
- "There is an anecdote..."
- "For example..."
- "I remember..."
- "He said..."
- "She said..."
- "Years later..."
- "Then..."
- "Because of this..."
When these appear, check whether the surrounding segments form one story. If yes, make one complete story clip.

Existing metadata for context:
{context_json}

Timed transcript:
{transcript}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0.25,
        max_completion_tokens=5000,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a careful Catholic sermon editor. Return only JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    payload = json.loads(raw)
    clips = payload.get("clips") or []
    if not isinstance(clips, list):
        raise RuntimeError("OpenAI clip selection returned no clips list.")
    return clips


def heuristic_select_clips(
    units: List[Dict[str, Any]],
    min_clips: int,
    max_clips: int,
) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    i = 0
    while i < len(units) and len(clips) < max_clips:
        start_i = i
        end_i = i
        while end_i < len(units) - 1 and units[end_i]["end"] - units[start_i]["start"] < 45:
            end_i += 1
        while end_i < len(units) - 1 and units[end_i]["end"] - units[start_i]["start"] < 75:
            if re.search(r"[.!?][\"')\]]?$", units[end_i]["text"]):
                break
            end_i += 1

        duration = units[end_i]["end"] - units[start_i]["start"]
        if MIN_SHORT_SECONDS <= duration <= MAX_SHORT_SECONDS:
            text = " ".join(unit["text"] for unit in units[start_i : end_i + 1])
            title_words = re.findall(r"[A-Za-z0-9']+", text)[:8]
            title = " ".join(title_words).strip() or f"Homily Clip {len(clips) + 1}"
            clips.append(
                {
                    "title": titlewrap(title),
                    "mdx_section_title": titlewrap(title),
                    "theme": titlewrap(text, 140),
                    "start_segment": units[start_i]["index"],
                    "end_segment": units[end_i]["index"],
                    "display_words": titlewrap(text, 220),
                    "keywords": ["Latin Mass", "Traditional Catholic", "homily"],
                }
            )
        i = max(end_i + 1, i + 1)

    if len(clips) < min_clips:
        print(f"Warning: heuristic selector found only {len(clips)} usable clips.")
    return clips


def titlewrap(text: str, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit].strip()


def nearest_word_time(words: List[Dict[str, Any]], target: float, kind: str) -> float:
    if not words:
        return float(target)
    key = "start" if kind == "start" else "end"
    return float(min(words, key=lambda w: abs(float(w.get(key, 0.0)) - target)).get(key, target))


def words_between(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [
        word
        for word in words
        if float(word.get("start", 0.0)) >= start - 0.05
        and float(word.get("end", 0.0)) <= end + 0.15
    ]


def make_caption_groups(clip_words: List[Dict[str, Any]], clip_start: float, clip_end: float) -> List[Dict[str, Any]]:
    if not clip_words:
        return [
            {
                "start": round(clip_start, 3),
                "end": round(clip_end, 3),
                "text": "",
            }
        ]

    groups: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(w["word"].strip() for w in current).strip()
        groups.append(
            {
                "start": round(float(current[0]["start"]), 3),
                "end": round(float(current[-1]["end"]), 3),
                "text": re.sub(r"\s+", " ", text),
            }
        )
        current = []

    for word in clip_words:
        current.append(word)
        text = " ".join(w["word"].strip() for w in current)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        if len(current) >= 9 or len(text) >= 56 or duration >= 3.7 or re.search(r"[.!?]$", word["word"].strip()):
            flush()
    flush()
    return groups


def prompt_for_clip_image(title: str, theme: str) -> str:
    subject = f"{title}. {theme}".strip()
    return (
        "Create a 4:5 portrait illustration for a YouTube Short about:\n\n"
        f"- {subject}\n\n"
        "Use a reverent Catholic visual tone in light watercolor with ink accents. "
        f"{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL} "
        "The image should feel contemplative, concrete, and tied to the topic. "
        "Do not include text, captions, logos, watermarks, typography, or UI."
    )


def traditional_catholic_image_prompt(prompt: str) -> str:
    prompt = str(prompt or "").strip()
    if TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL in prompt:
        return prompt
    return (
        f"{prompt}\n\n"
        f"{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL}\n"
        "Keep the final result in a light watercolor and ink style, with no text or watermarks."
    ).strip()


def build_manifest(
    raw_clips: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    paths: Dict[str, str],
    max_clips: int,
    selection_mode: str,
) -> Dict[str, Any]:
    by_index = {unit["index"]: unit for unit in units}
    used_ranges: List[Tuple[float, float]] = []
    clips: List[Dict[str, Any]] = []

    for raw in raw_clips:
        if len(clips) >= max_clips:
            break
        try:
            start_segment = int(raw.get("start_segment"))
            end_segment = int(raw.get("end_segment"))
        except (TypeError, ValueError):
            continue
        if start_segment not in by_index or end_segment not in by_index:
            continue
        if end_segment < start_segment:
            start_segment, end_segment = end_segment, start_segment

        start = float(by_index[start_segment]["start"])
        end = float(by_index[end_segment]["end"])
        start = nearest_word_time(words, start, "start")
        end = nearest_word_time(words, end, "end")
        duration = round(end - start, 3)
        if duration < MIN_SHORT_SECONDS or duration > MAX_SHORT_SECONDS:
            continue
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue

        clip_words = words_between(words, start, end)
        text = " ".join(unit["text"] for unit in units if start_segment <= unit["index"] <= end_segment)
        title = titlewrap(
            clean_editorial_title(
                raw.get("title") or raw.get("mdx_section_title") or text,
                text,
            )
        )
        clip_id = f"clip-{len(clips) + 1:02d}"
        slug = slugify(title, clip_id)
        image_path = os.path.join(paths["images_dir"], f"{clip_id}.jpg")
        audio_path = os.path.join(paths["audio_dir"], f"{clip_id}.mp3")
        video_path = os.path.join(paths["videos_dir"], f"{clip_id}-{slug}.mp4")
        captions_path = os.path.join(paths["captions_dir"], f"{clip_id}.srt")
        theme = titlewrap(raw.get("theme") or text, 180)
        caption_groups = make_caption_groups(clip_words, start, end)

        clips.append(
            {
                "id": clip_id,
                "title": title,
                "mdx_section_title": titlewrap(raw.get("mdx_section_title") or title),
                "theme": theme,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": duration,
                "display_words": titlewrap(raw.get("display_words") or text, 260),
                "caption_groups": caption_groups,
                "image_prompt": prompt_for_clip_image(title, theme),
                "keywords": list(raw.get("keywords") or []),
                "output_paths": {
                    "image": image_path,
                    "audio": audio_path,
                    "video": video_path,
                    "captions": captions_path,
                },
            }
        )
        used_ranges.append((start, end))

    return {
        "version": 1,
        "source": {
            "root": paths["root"],
            "video_script": paths["video_script"],
            "audio": paths["audio"],
            "mdx": paths["mdx"],
        },
        "settings": {
            "canvas_size": list(CANVAS_SIZE),
            "image_size": list(IMAGE_SIZE),
            "min_seconds": MIN_SHORT_SECONDS,
            "max_seconds": MAX_SHORT_SECONDS,
            "selection_mode": selection_mode,
        },
        "clips": clips,
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def clean_editorial_title(title: str, source_text: str, limit: int = 70) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    source_text = re.sub(r"\s+", " ", str(source_text or "")).strip()

    title_words = re.findall(r"[A-Za-z0-9']+", title.lower())
    source_words = re.findall(r"[A-Za-z0-9']+", source_text.lower())

    # If the title is basically just the first words of the transcript, reject it.
    if title_words and source_words[: len(title_words)] == title_words:
        meaningful_words = [
            word for word in source_words
            if word not in {
                "the", "a", "an", "and", "or", "but", "so", "that", "there",
                "this", "these", "those", "is", "are", "was", "were", "of",
                "from", "to", "in", "on", "with", "for", "as", "it"
            }
        ]

        if "doyle" in source_text.lower():
            return "One Sentence That Saved Her Soul"

        if any(word in source_text.lower() for word in ["mercy", "compassion", "charity"]):
            return "The Power of Christian Compassion"

        if meaningful_words:
            return titlewrap(" ".join(word.capitalize() for word in meaningful_words[:7]), limit)

    return titlewrap(title, limit)

def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def image_metadata_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + ".image.json"


def save_clip_image(clip: Dict[str, Any]) -> str:
    output_path = clip["output_paths"]["image"]
    metadata_path = image_metadata_path(output_path)

    client = _openai_client()
    model = os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    image_kwargs = _supported_image_kwargs(
        client.images.generate,
        {
            "model": model,
            "prompt": traditional_catholic_image_prompt(clip["image_prompt"]),
            "size": os.getenv("OPENAI_SHORTS_IMAGE_SIZE", "1024x1536"),
            "quality": os.getenv("OPENAI_IMAGE_QUALITY", "high"),
            "output_format": "jpeg",
            "output_compression": 92,
            "response_format": "b64_json",
            "n": 1,
        },
    )
    if str(model).startswith("gpt-image"):
        image_kwargs.pop("response_format", None)
    response = _generate_image_with_compatible_kwargs(client, image_kwargs)
    image_bytes = _image_bytes_from_response(response)

    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Generated image was not readable for {clip['id']}.") from exc

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format="JPEG", quality=92, optimize=True, progressive=True)
    save_json(
        metadata_path,
        {
            "generated": True,
            "clip_id": clip.get("id"),
            "title": clip.get("title"),
            "image_prompt": traditional_catholic_image_prompt(clip.get("image_prompt")),
            "image_size": list(IMAGE_SIZE),
            "model": model,
        },
    )
    return output_path


def make_background(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Short image is not readable: {image_path}") from exc

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", CANVAS_SIZE, "black")
    canvas.paste(image, (0, 0))

    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    gradient_height = GRADIENT_BOTTOM - GRADIENT_TOP
    for offset in range(gradient_height):
        alpha = int(255 * ((offset + 1) / gradient_height))
        y = GRADIENT_TOP + offset
        ImageDraw.Draw(overlay).line([(0, y), (CANVAS_SIZE[0], y)], fill=(0, 0, 0, alpha))
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
    max_width = safe_width
    max_lines = 3
    lines = wrap_text(text, font, max_width)[:max_lines]

    def text_metrics() -> Tuple[int, List[int], List[int]]:
        line_gap = int(font_size * 0.32)
        line_widths = []
        line_heights = []
        for line_text in lines:
            bbox = font.getbbox(line_text)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        height = sum(line_heights) + line_gap * max(0, len(lines) - 1)
        return height, line_widths, line_heights

    total_height, widths, heights = text_metrics()
    while (
        (len(wrap_text(text, font, max_width)) > max_lines or total_height > safe_height)
        and font_size > 38
    ):
        font_size -= 4
        font = load_font(font_size, bold=True, serif=title)
        lines = wrap_text(text, font, max_width)[:max_lines]
        total_height, widths, heights = text_metrics()

    line_gap = int(font_size * 0.32)
    y = safe_top + max(0, (safe_height - total_height) // 2)
    fill = (206, 24, 32, 255) if title else (255, 255, 255, 255)
    shadow = (0, 0, 0, 210)

    for line, width, height in zip(lines, widths, heights):
        x = safe_left + (safe_width - width) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_gap
    return overlay


def write_srt(clip: Dict[str, Any]) -> None:
    lines = []
    clip_start = float(clip["start"])
    for index, group in enumerate(clip.get("caption_groups") or [], 1):
        text = str(group.get("text") or "").strip()
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
    output_path = clip["output_paths"]["captions"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def list_background_audio_files(bg_audio_dir: str) -> List[str]:
    bg_audio_dir = os.path.abspath(os.path.expanduser(clean_path(bg_audio_dir)))
    if not os.path.isdir(bg_audio_dir):
        return []
    files = []
    for filename in os.listdir(bg_audio_dir):
        path = os.path.join(bg_audio_dir, filename)
        if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in BACKGROUND_AUDIO_EXTENSIONS:
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
    speech_path: str,
    bg_audio_files: List[str],
    gain_db: float,
    fade_ms: int,
) -> Optional[str]:
    if not bg_audio_files:
        return None

    bg_path = random.choice(bg_audio_files)
    speech = AudioSegment.from_file(speech_path)
    duration_ms = len(speech)
    if duration_ms <= 0:
        raise RuntimeError(f"Speech audio has no duration: {speech_path}")

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


def render_clip(
    clip: Dict[str, Any],
    source_audio: str,
    bg_audio_files: List[str],
    bg_music_gain_db: float,
    bg_music_fade_ms: int,
    skip_existing: bool = True,
) -> str:
    output_path = clip["output_paths"]["video"]
    image_path = clip["output_paths"]["image"]
    if (
        skip_existing
        and os.path.exists(output_path)
        and os.path.exists(image_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(image_path)
    ):
        return output_path

    audio_path = clip["output_paths"]["audio"]
    clip_audio_segment(
        input_file=source_audio,
        start_sec=float(clip["start"]),
        end_sec=float(clip["end"]),
        output_file=audio_path,
    )
    bg_audio_path = mix_background_music(
        speech_path=audio_path,
        bg_audio_files=bg_audio_files,
        gain_db=bg_music_gain_db,
        fade_ms=bg_music_fade_ms,
    )
    if bg_audio_path:
        clip["background_audio"] = bg_audio_path
        print(f"Background music for {clip['id']}: {os.path.basename(bg_audio_path)}")
    write_srt(clip)

    duration = float(clip["duration"])
    background = np.array(make_background(image_path))
    layers = [ImageClip(background, duration=duration)]

    title_duration = min(TITLE_SECONDS, max(1.4, duration * 0.18))
    layers.append(
        ImageClip(np.array(text_overlay(clip["title"], title=True)), duration=title_duration).with_start(0)
    )

    clip_start = float(clip["start"])
    for group in clip.get("caption_groups") or []:
        text = str(group.get("text") or "").strip()
        if not text:
            continue
        rel_start = max(0.0, float(group["start"]) - clip_start)
        rel_end = min(duration, max(rel_start + 0.3, float(group["end"]) - clip_start))
        if rel_end <= title_duration:
            continue
        rel_start = max(rel_start, title_duration)
        layers.append(
            ImageClip(np.array(text_overlay(text, title=False)), duration=rel_end - rel_start).with_start(rel_start)
        )

    audio = AudioFileClip(audio_path)
    video = CompositeVideoClip(layers, size=CANVAS_SIZE).with_audio(audio).with_duration(duration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video.write_videofile(
        output_path,
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
    return output_path


def write_upload_metadata(manifest: Dict[str, Any], paths: Dict[str, str]) -> None:
    existing_by_id = {}
    if os.path.exists(paths["upload_metadata"]):
        try:
            existing = load_manifest(paths["upload_metadata"])
            existing_by_id = {
                clip.get("id"): clip
                for clip in existing.get("clips", [])
                if clip.get("id")
            }
        except Exception:
            existing_by_id = {}

    def clip_upload_record(clip: Dict[str, Any]) -> Dict[str, Any]:
        existing = existing_by_id.get(clip["id"]) or {}
        video_path = clip["output_paths"]["video"]
        existing_video_path = str(existing.get("video_path") or "")
        if (
            f"{os.sep}uploaded{os.sep}" in existing_video_path
            and os.path.exists(existing_video_path)
        ):
            video_path = existing_video_path

        return {
            "id": clip["id"],
            "title": clip["title"],
            "description": clip["theme"],
            "tags": clip.get("keywords") or [],
            "start": clip["start"],
            "end": clip["end"],
            "duration": clip["duration"],
            "video_path": video_path,
            "thumbnail_path": clip["output_paths"]["image"],
            "captions_path": clip["output_paths"]["captions"],
            "background_audio": clip.get("background_audio") or existing.get("background_audio", ""),
        }

    payload = {
        "source": manifest.get("source", {}),
        "clips": [clip_upload_record(clip) for clip in manifest.get("clips", [])],
    }
    save_json(paths["upload_metadata"], payload)


def selected_clips(args: argparse.Namespace, clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    requested_ids = []
    for value in args.clip_id or []:
        requested_ids.extend(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    if requested_ids:
        requested = set(requested_ids)
        clips = [clip for clip in clips if clip.get("id") in requested]
        missing = [clip_id for clip_id in requested_ids if clip_id not in {clip.get("id") for clip in clips}]
        if missing:
            raise ValueError(f"Requested clip id(s) not found in manifest: {', '.join(missing)}")

    if args.limit:
        clips = clips[: args.limit]
    return clips


def create_or_load_manifest(args: argparse.Namespace, paths: Dict[str, str]) -> Dict[str, Any]:
    if args.render_only:
        if not os.path.exists(paths["manifest"]):
            raise FileNotFoundError(f"--render-only requested but manifest is missing: {paths['manifest']}")
        return load_manifest(paths["manifest"])

    if os.path.exists(paths["manifest"]) and not args.force_manifest:
        print(f"Using existing manifest: {paths['manifest']}")
        return load_manifest(paths["manifest"])

    script = load_video_script(paths["video_script"])
    segments = script["segments"]
    units = segment_units(segments)
    words = collect_words(segments)
    context = load_mdx_context(paths["mdx"])

    effective_selection_mode = args.selection_mode

    if args.selection_mode == "heuristic":
        raw_clips = heuristic_select_clips(units, args.min_clips, args.max_clips)
    else:
        try:
            raw_clips = openai_select_clips(units, context, args.min_clips, args.max_clips)
        except Exception as exc:
            if args.selection_mode == "openai":
                raise
            print(f"OpenAI clip selection failed; using heuristic fallback. Reason: {exc}")
            raw_clips = heuristic_select_clips(units, args.min_clips, args.max_clips)
            effective_selection_mode = "heuristic-fallback"

    manifest = build_manifest(raw_clips, units, words, paths, args.max_clips, effective_selection_mode)
    if (
        args.selection_mode == "auto"
        and len(manifest.get("clips", [])) < args.min_clips
    ):
        print(
            f"OpenAI selection produced {len(manifest.get('clips', []))} valid clip(s); "
            "supplementing with heuristic clips."
        )
        heuristic_raw = heuristic_select_clips(units, args.min_clips, args.max_clips)
        raw_clips = raw_clips + heuristic_raw
        manifest = build_manifest(raw_clips, units, words, paths, args.max_clips, "openai+heuristic-fallback")

    if len(manifest.get("clips", [])) < args.min_clips:
        message = (
            f"Only {len(manifest.get('clips', []))} valid clip(s) were selected; "
            f"requested at least {args.min_clips}."
        )
        if args.selection_mode == "openai":
            raise RuntimeError(message)
        print("Warning: " + message)

    save_json(paths["manifest"], manifest)
    print(f"Shorts manifest saved: {paths['manifest']}")
    print(f"Clips selected: {len(manifest.get('clips', []))}")
    return manifest


def process_media(args: argparse.Namespace, manifest: Dict[str, Any], paths: Dict[str, str]) -> None:
    clips = selected_clips(args, manifest.get("clips") or [])
    if not clips:
        raise RuntimeError("No clips available to render.")

    print(f"Generating fresh clip images with {args.image_workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=max(1, args.image_workers)) as image_pool:
        image_futures = {
            clip["id"]: image_pool.submit(save_clip_image, clip)
            for clip in clips
        }
        for clip in clips:
            image_futures[clip["id"]].result()
            print(f"Image ready: {clip['output_paths']['image']}")

    if args.images_only:
        write_upload_metadata(manifest, paths)
        return

    bg_audio_files = [] if args.no_bg_music else list_background_audio_files(args.bg_audio_dir)
    if bg_audio_files:
        print(f"Background music pool: {len(bg_audio_files)} file(s)")
    elif args.no_bg_music:
        print("Background music disabled.")
    else:
        print(f"No background music files found in: {os.path.abspath(os.path.expanduser(args.bg_audio_dir))}")

    print("Rendering videos one at a time...")
    for clip in clips:
        render_clip(
            clip,
            paths["audio"],
            bg_audio_files=bg_audio_files,
            bg_music_gain_db=args.bg_music_gain_db,
            bg_music_fade_ms=args.bg_music_fade_ms,
            skip_existing=not args.force_media,
        )
        print(f"Video ready: {clip['output_paths']['video']}")

    write_upload_metadata(manifest, paths)
    print(f"Upload metadata saved: {paths['upload_metadata']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YouTube Shorts from a completed homily workflow folder.")
    parser.add_argument("source", nargs="?", help="Completed homily folder or a file inside it.")
    parser.add_argument("--manifest-only", action="store_true", help="Create/update shorts_manifest.json and stop.")
    parser.add_argument("--render-only", action="store_true", help="Render from an existing shorts_manifest.json.")
    parser.add_argument("--images-only", action="store_true", help="Generate fresh images and upload metadata, but skip video rendering.")
    parser.add_argument("--force-manifest", action="store_true", help="Regenerate shorts_manifest.json even if it exists.")
    parser.add_argument("--force-media", action="store_true", help="Regenerate audio/videos even if files exist. Images are always regenerated.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected clips.")
    parser.add_argument("--clip-id", action="append", help="Process a specific clip id, such as clip-02. Can be repeated or comma-separated.")
    parser.add_argument("--min-clips", type=int, default=DEFAULT_MIN_CLIPS)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--bg-audio-dir", default=default_bg_audio_dir(), help="Folder of background music files to randomly choose from.")
    parser.add_argument("--bg-music-gain-db", type=float, default=DEFAULT_BG_MUSIC_GAIN_DB, help="Background music gain in dB; lower is quieter.")
    parser.add_argument("--bg-music-fade-ms", type=int, default=DEFAULT_BG_MUSIC_FADE_MS, help="Fade-in/out duration for background music.")
    parser.add_argument("--no-bg-music", action="store_true", help="Disable background music mixing.")
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "openai", "heuristic"],
        default="auto",
        help="auto uses OpenAI and falls back to heuristic; openai fails hard; heuristic avoids API selection calls.",
    )
    parser.add_argument("--image-workers", type=int, default=2, help="Concurrent image-generation jobs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source:
        args.source = clean_path(input("Enter completed homily audio/folder path: "))
    if args.min_clips < 1:
        raise ValueError("--min-clips must be at least 1.")
    if args.max_clips < args.min_clips:
        raise ValueError("--max-clips must be greater than or equal to --min-clips.")

    paths = resolve_paths(args.source)
    ensure_output_dirs(paths)
    manifest = create_or_load_manifest(args, paths)

    if args.manifest_only:
        return
    process_media(args, manifest, paths)


if __name__ == "__main__":
    main()
