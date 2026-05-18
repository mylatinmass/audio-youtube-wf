import argparse
import hashlib
import json
import os
import random
import re
import shutil
from datetime import datetime, timezone
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
IMAGE_TOP = 570
TEXT_BOX_BOTTOM = 570
GRADIENT_TOP = 570
GRADIENT_BOTTOM = 1020
FPS = 30
MIN_SHORT_SECONDS = 15.0
MAX_SHORT_SECONDS = 90.0
DEFAULT_MAX_CLIPS = 7
DEFAULT_MIN_CLIPS = 3
TITLE_SECONDS = 2.4
TEXT_SAFE_LEFT = 74
TEXT_SAFE_RIGHT = 74
TEXT_SAFE_TOP = 20
TEXT_SAFE_BOTTOM = CANVAS_SIZE[1] - TEXT_BOX_BOTTOM
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


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def chapter_ranges_from_context(
    context: Dict[str, Any],
    units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    duration = float(units[-1]["end"]) if units else 0.0
    chapters: List[Dict[str, Any]] = []
    for chapter in context.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        title = re.sub(r"\s+", " ", str(chapter.get("title") or "").strip())
        start = as_float(chapter.get("start"))
        if not title or start is None:
            continue
        chapters.append({"title": title, "start": max(0.0, start)})

    chapters.sort(key=lambda item: item["start"])
    ranges: List[Dict[str, Any]] = []
    for index, chapter in enumerate(chapters):
        end = chapters[index + 1]["start"] if index + 1 < len(chapters) else duration
        if duration:
            end = min(end, duration)
        if end <= chapter["start"]:
            continue
        ranges.append(
            {
                "index": index + 1,
                "title": chapter["title"],
                "start": chapter["start"],
                "end": end,
                "duration": end - chapter["start"],
            }
        )
    return ranges


def format_chapter_map(context: Dict[str, Any], units: List[Dict[str, Any]]) -> str:
    chapter_ranges = chapter_ranges_from_context(context, units)
    lines = []
    for chapter in chapter_ranges:
        lines.append(
            f'{chapter["index"]:02d} [{format_timestamp(chapter["start"])}-{format_timestamp(chapter["end"])}] '
            f'{chapter["title"]}'
        )

    shorts = []
    for short in context.get("shorts") or []:
        if not isinstance(short, dict):
            continue
        title = re.sub(r"\s+", " ", str(short.get("title") or "").strip())
        start = as_float(short.get("start"))
        end = as_float(short.get("end"))
        quote = re.sub(r"\s+", " ", str(short.get("quote") or "").strip())
        if not title:
            continue
        time_part = ""
        if start is not None and end is not None:
            time_part = f" [{format_timestamp(start)}-{format_timestamp(end)}]"
        shorts.append(f"- {title}{time_part}: {quote}".strip())

    if not lines and not shorts:
        return "No MDX chapter or short metadata was available."

    out = []
    if lines:
        out.append("MDX YouTube chapters:")
        out.extend(lines)
    if shorts:
        out.append("\nExisting MDX short suggestions:")
        out.extend(shorts)
    return "\n".join(out)


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
    chapter_map = format_chapter_map(context, units)
    prompt = f"""
You are a Catholic homily editor creating YouTube Shorts.

Your job is NOT to hunt for isolated viral quotes.
Your job is to make chapter-aligned, self-contained Shorts that preserve the homily's structure.
Use the existing YouTube chapter plan as your editorial map. The Shorts should feel like close cousins
of the chaptered YouTube video: same sermon logic, same major movements, but packaged as vertical clips.

Return ONLY valid JSON:
{{
  "clips": [
    {{
      "major_point_id": "point-01",
      "major_point_title": "<the major idea this clip represents, <=70 chars>",
      "title": "<original YouTube Shorts title, not a transcript line, <=70 chars>",
      "mdx_section_title": "<section-style heading, <=70 chars>",
      "theme": "<one sentence explaining why this clip is worth watching>",
      "selection_reason": "<why this excerpt represents the chapter movement, <=180 chars>",
      "power_quote": "<the clearest exact line or central phrase from the clip, <=180 chars>",
      "clip_type": "story|teaching|warning|exhortation|reflection",
      "story_arc": "<beginning, turn, and payoff if this is a story; otherwise empty string>",
      "start_segment": <integer>,
      "end_segment": <integer>,
      "display_words": "<usually the power_quote; concise on-screen text, <=260 chars>",
      "keywords": ["keyword", "keyword"]
    }}
  ]
}}

Selection rules:
- First identify the 3 to 7 major ideas or movements in the homily.
- If MDX YouTube chapters are provided, use those chapters as the source of truth for the major ideas.
- Prefer one representative Short from each chapter movement instead of clustering several Shorts in one chapter.
- major_point_title should usually match the closest MDX chapter title.
- mdx_section_title should be the chapter-style title for the sermon movement; it may match major_point_title.
- title may be more YouTube Shorts friendly, but it must still clearly belong to that same chapter movement.
- Existing MDX short suggestions are strong hints. Include or refine them when they are complete, 30 to 90 seconds, and not duplicated by a better nearby moment.
- If a chapter is 30 to 90 seconds long, usually use the whole chapter as the Short.
- If a chapter is longer than 90 seconds, choose the earliest complete 45 to 90 second excerpt that establishes the chapter's main claim.
- Do not skip to a later quotable line unless the chapter opening is unusable, confusing, or purely housekeeping.
- Then choose one representative complete 30 to 90 second excerpt from each major idea.
- Select no more than {max_clips} clips. This is a ceiling, not a target.
- Try to select at least {min_clips} clips if the transcript has enough strong major ideas.
- Do not create filler clips just to reach {max_clips}.
- Do not split one major idea into several adjacent clips.
- Do not return multiple clips with the same major_point_title unless they are truly distinct sub-points.
- Avoid clips under 30 seconds. Use them only if no complete 30 to 90 second version exists.
- Each clip must be a complete standalone teaching, warning, exhortation, reflection, story, or spiritual lesson.
- If a story, anecdote, example, or testimony appears, treat the ENTIRE story as ONE clip when it fits under 90 seconds.
- Do NOT split one story into multiple Shorts unless the full story is longer than 90 seconds.
- A story clip must include the setup, the key action or conflict, and the spiritual payoff.
- If a story is followed immediately by the preacher explaining why the story matters, include that explanation if the total clip remains under 90 seconds.
- Prefer faithful representation of the chapter movement over the most dramatic quote.
- Avoid announcements, housekeeping, readings, or context that cannot stand alone.
- Use non-overlapping clips.
- Use start_segment and end_segment numbers from the transcript.
- Choose start_segment early enough that the viewer understands what is happening.
- Choose end_segment late enough that the viewer receives the payoff.
- power_quote should be the most memorable exact line when possible.
- display_words should usually match power_quote. If no concise quote exists, use a short central phrase from the clip.
- Titles must be original editorial titles, not the first words of the transcript.
- Bad title example: "There Is An Anecdote From The Life"
- Good title example: "One Sentence That Saved Her Soul"
- Good title example: "The Priest Who Gave Hope to a Dying Woman"
- Good title example: "Compassion Can Reach a Soul Years Later"
- Do not invent a quote. display_words can be a concise phrase if a verbatim quote would be too long.
- Do not select several clips from the same story. Pick the complete version of that story instead.

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

Chapter/short map:
{chapter_map}

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


POINT_KEYWORDS = {
    "father_doyle_story": {
        "title": "One Sentence That Saved Her Soul",
        "terms": [
            "father william doyle", "doyle", "woman of the night", "death row",
            "please don't sin", "god loves you", "confession", "last rites",
        ],
    },
    "christian_compassion": {
        "title": "How to Suffer With Others",
        "terms": [
            "compassion", "mercy", "suffer with", "suffer", "carry", "cross",
            "weep", "widow", "moved by mercy", "cannot give what we do not have",
        ],
    },
    "danger_of_pride": {
        "title": "The Danger of Pride",
        "terms": ["pride", "proud", "humility", "humble", "selfish", "obstacle"],
    },
    "charity_and_grace": {
        "title": "The Charity That Changes Souls",
        "terms": [
            "charity", "love of god", "love our neighbor", "grace", "salvation",
            "eternally happy", "participants in his salvation",
        ],
    },
    "sin_and_conversion": {
        "title": "The Call to Conversion",
        "terms": ["sin", "repent", "conversion", "soul", "save", "saved", "confession"],
    },
}

POWER_TERMS = [
    "god loves you",
    "please don't sin",
    "we cannot give what we do not have",
    "how can we help",
    "carry our own cross",
    "moved by mercy",
    "pride",
    "humility",
    "compassion",
    "charity",
    "love of god",
    "salvation",
]

GENERIC_SHORTS_POWER_TERMS = [
    "faith",
    "grace",
    "salvation",
    "soul",
    "souls",
    "church",
    "holy mass",
    "blessed sacrament",
    "our lord",
    "jesus christ",
    "cross",
    "sacrifice",
    "truth",
    "charity",
    "mercy",
    "sin",
    "repent",
    "conversion",
    "hope",
    "doctrine",
    "moral law",
    "catholic faith",
]


def text_signature(value: str) -> str:
    stop_words = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "with",
        "for", "from", "by", "as", "is", "are", "was", "were", "be", "being",
        "this", "that", "these", "those", "it", "its", "we", "our", "you",
    }
    words = [
        word for word in re.findall(r"[a-z0-9']+", str(value or "").lower())
        if word not in stop_words
    ]
    return " ".join(words[:10])


def raw_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def substantially_overlaps(
    start: float,
    end: float,
    used_ranges: List[Tuple[float, float]],
    ratio: float = 0.25,
) -> bool:
    duration = max(0.001, end - start)
    for used_start, used_end in used_ranges:
        overlap = overlap_seconds(start, end, used_start, used_end)
        if overlap >= 2.0 and overlap / min(duration, max(0.001, used_end - used_start)) >= ratio:
            return True
    return False


def extract_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_power_quote(text: str, fallback_limit: int = 180) -> str:
    lowered_text = str(text or "").lower()
    if "please don't sin" in lowered_text and "god loves you" in lowered_text:
        return "Please don't sin. God loves you."
    if "we cannot give what we do not have" in lowered_text:
        return "We cannot give what we do not have."

    sentences = extract_sentences(text)
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        if len(words) < 4 or len(words) > 28:
            continue
        lowered = sentence.lower()
        score = sum(5 for term in POWER_TERMS if term in lowered)
        score += min(len(words), 18) / 8
        if re.search(r"\b(how|why|cannot|must|never|always|therefore)\b", lowered):
            score += 2
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if best_sentence:
        return titlewrap(best_sentence, fallback_limit)
    return titlewrap(text, fallback_limit)


def meaningful_terms(value: str) -> List[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "with",
        "for", "from", "by", "as", "is", "are", "was", "were", "be", "being",
        "this", "that", "these", "those", "it", "its", "we", "our", "you",
        "he", "his", "her", "their", "they", "them",
    }
    return [
        word
        for word in re.findall(r"[a-z0-9']+", str(value or "").lower())
        if len(word) > 2 and word not in stop_words
    ]


def chapter_for_range(
    chapter_ranges: List[Dict[str, Any]],
    start: float,
    end: float,
) -> Optional[Dict[str, Any]]:
    best = None
    best_overlap = 0.0
    for chapter in chapter_ranges:
        overlap = overlap_seconds(start, end, float(chapter["start"]), float(chapter["end"]))
        if overlap > best_overlap:
            best = chapter
            best_overlap = overlap
    return best


def nearest_unit_index(units: List[Dict[str, Any]], target: float, key: str) -> int:
    if not units:
        return 0
    return min(
        range(len(units)),
        key=lambda idx: abs(float(units[idx].get(key, 0.0) or 0.0) - target),
    )


def clip_from_mdx_short(
    short: Dict[str, Any],
    units: List[Dict[str, Any]],
    chapter_ranges: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    start = as_float(short.get("start"))
    end = as_float(short.get("end"))
    if start is None or end is None or end <= start:
        return None
    duration = end - start
    if duration < MIN_SHORT_SECONDS or duration > MAX_SHORT_SECONDS:
        return None

    start_i = nearest_unit_index(units, start, "start")
    end_i = nearest_unit_index(units, end, "end")
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    text = " ".join(unit["text"] for unit in units[start_i : end_i + 1])
    chapter = chapter_for_range(chapter_ranges, start, end)
    chapter_title = (chapter or {}).get("title") or short.get("title") or "Homily Short"
    power_quote = titlewrap(short.get("quote") or extract_power_quote(text), 180)
    title = clean_editorial_title(short.get("title") or chapter_title, text)
    return {
        "major_point_id": "",
        "major_point_title": titlewrap(chapter_title),
        "title": titlewrap(title),
        "mdx_section_title": titlewrap(chapter_title),
        "theme": titlewrap(text, 180),
        "selection_reason": "Existing MDX short suggestion aligned to the YouTube chapter plan.",
        "power_quote": power_quote,
        "start_segment": units[start_i]["index"],
        "end_segment": units[end_i]["index"],
        "display_words": titlewrap(power_quote or text, 260),
        "keywords": list(short.get("keywords") or ["Latin Mass", "Traditional Catholic", "homily"]),
    }


def score_chapter_window(text: str, duration: float, chapter_title: str, chapter_start_delta: float) -> float:
    lowered = re.sub(r"\s+", " ", str(text or "").lower())
    score = max(0.0, 20.0 - abs(duration - 62.0) * 0.35)
    score += max(0.0, 35.0 - chapter_start_delta * 1.8)
    score += sum(2 for term in GENERIC_SHORTS_POWER_TERMS if term in lowered)
    score += sum(3 for term in meaningful_terms(chapter_title) if re.search(rf"\b{re.escape(term)}\b", lowered))
    if re.search(r"\b(therefore|because|so that|in order that|consequently|this is why)\b", lowered):
        score += 3
    if re.search(r"\b(must|cannot|never|always|only|essential|necessary|obliges)\b", lowered):
        score += 2
    if re.search(r"[.!?][\"')\]]?$", text.strip()):
        score += 5
    if re.search(r"\b(announcement|subscribe|youtube|housekeeping|vestibule)\b", lowered):
        score -= 25
    return score


def chapter_candidate(
    units: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    chapter: Dict[str, Any],
) -> Dict[str, Any]:
    text = " ".join(unit["text"] for unit in units[start_idx : end_idx + 1])
    start = float(units[start_idx]["start"])
    end = float(units[end_idx]["end"])
    chapter_start = float(chapter["start"])
    return {
        "score": score_chapter_window(text, end - start, str(chapter["title"]), abs(start - chapter_start)),
        "major_point_title": chapter["title"],
        "text": text,
        "start_segment": units[start_idx]["index"],
        "end_segment": units[end_idx]["index"],
        "start": start,
        "end": end,
        "duration": end - start,
    }


def representative_window_for_chapter(
    units: List[Dict[str, Any]],
    chapter: Dict[str, Any],
    used_ranges: List[Tuple[float, float]],
) -> Optional[Dict[str, Any]]:
    chapter_start = float(chapter["start"])
    chapter_end = float(chapter["end"])
    indexed_units = [
        (idx, unit)
        for idx, unit in enumerate(units)
        if float(unit["end"]) > chapter_start and float(unit["start"]) < chapter_end
    ]
    if not indexed_units:
        return None

    first_idx = indexed_units[0][0]
    last_idx = indexed_units[-1][0]
    chapter_duration = float(chapter["duration"])
    if 30.0 <= chapter_duration <= MAX_SHORT_SECONDS:
        start = float(units[first_idx]["start"])
        end = float(units[last_idx]["end"])
        if not substantially_overlaps(start, end, used_ranges):
            return chapter_candidate(units, first_idx, last_idx, chapter)

    opening_candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    for pos, (start_idx, start_unit) in enumerate(indexed_units):
        start = max(float(start_unit["start"]), chapter_start)
        start_delta = start - chapter_start
        if start_delta > 35.0 and opening_candidates:
            break
        for end_idx, end_unit in indexed_units[pos:]:
            end = min(float(end_unit["end"]), chapter_end)
            duration = end - start
            if duration < 30.0:
                continue
            if duration > MAX_SHORT_SECONDS:
                break
            if substantially_overlaps(start, end, used_ranges):
                continue
            text = " ".join(unit["text"] for unit in units[start_idx : end_idx + 1])
            if duration < 42.0 and not re.search(r"[.!?][\"')\]]?$", text.strip()):
                continue
            candidate = chapter_candidate(units, start_idx, end_idx, chapter)
            fallback_candidates.append(candidate)
            if start_delta <= 35.0 and duration >= 45.0:
                opening_candidates.append(candidate)

    if opening_candidates:
        opening_candidates.sort(
            key=lambda item: (
                float(item["start"]) - chapter_start,
                abs(float(item["duration"]) - 62.0),
                -float(item["score"]),
            )
        )
        return opening_candidates[0]

    fallback_candidates.sort(
        key=lambda item: (
            float(item["start"]) - chapter_start,
            abs(float(item["duration"]) - 62.0),
            -float(item["score"]),
        )
    )
    return fallback_candidates[0] if fallback_candidates else None


def chapter_based_heuristic_select_clips(
    units: List[Dict[str, Any]],
    context: Dict[str, Any],
    min_clips: int,
    max_clips: int,
) -> List[Dict[str, Any]]:
    chapter_ranges = chapter_ranges_from_context(context, units)
    if not chapter_ranges and not context.get("shorts"):
        return []

    clips: List[Dict[str, Any]] = []
    used_ranges: List[Tuple[float, float]] = []
    used_chapters = set()

    for short in context.get("shorts") or []:
        if len(clips) >= max_clips or not isinstance(short, dict):
            break
        clip = clip_from_mdx_short(short, units, chapter_ranges)
        if not clip:
            continue
        start = float(units[nearest_unit_index(units, float(short["start"]), "start")]["start"])
        end = float(units[nearest_unit_index(units, float(short["end"]), "end")]["end"])
        if substantially_overlaps(start, end, used_ranges):
            continue
        clips.append(clip)
        used_ranges.append((start, end))
        if clip.get("major_point_title"):
            used_chapters.add(text_signature(clip["major_point_title"]))

    candidates = []
    for chapter in chapter_ranges:
        chapter_key = text_signature(chapter["title"])
        if chapter_key in used_chapters:
            continue
        candidate = representative_window_for_chapter(units, chapter, used_ranges)
        if candidate:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    for candidate in candidates:
        if len(clips) >= max_clips:
            break
        start = float(candidate["start"])
        end = float(candidate["end"])
        if substantially_overlaps(start, end, used_ranges):
            continue
        power_quote = extract_power_quote(candidate["text"])
        chapter_title = candidate["major_point_title"]
        clips.append(
            {
                "major_point_id": "",
                "major_point_title": titlewrap(chapter_title),
                "title": titlewrap(chapter_title),
                "mdx_section_title": titlewrap(chapter_title),
                "theme": titlewrap(candidate["text"], 180),
                "selection_reason": "Representative 30-90 second excerpt from this YouTube chapter.",
                "power_quote": power_quote,
                "start_segment": candidate["start_segment"],
                "end_segment": candidate["end_segment"],
                "display_words": titlewrap(power_quote, 260),
                "keywords": ["Latin Mass", "Traditional Catholic", "homily"],
            }
        )
        used_ranges.append((start, end))

    clips.sort(key=lambda clip: int(clip["start_segment"]))
    for index, clip in enumerate(clips, 1):
        clip["major_point_id"] = f"point-{index:02d}"

    if len(clips) < min_clips:
        print(f"Warning: chapter-based heuristic found only {len(clips)} chapter-aligned clip(s).")
    return clips


def classify_major_point(text: str) -> Tuple[str, str, int]:
    lowered = re.sub(r"\s+", " ", str(text or "").lower())
    best_key = "general_homily_point"
    best_title = "A Strong Homily Moment"
    best_score = 0
    for key, data in POINT_KEYWORDS.items():
        score = 0
        for term in data["terms"]:
            term = term.lower()
            if " " in term:
                score += 4 * lowered.count(term)
            else:
                score += len(re.findall(rf"\b{re.escape(term)}\b", lowered))
        if score > best_score:
            best_key = key
            best_title = data["title"]
            best_score = score
    return best_key, best_title, best_score


def score_heuristic_candidate(text: str, duration: float, point_score: int) -> float:
    lowered = text.lower()
    score = float(point_score * 7)
    score += max(0.0, 24.0 - abs(duration - 62.0) * 0.45)
    score += sum(6 for term in POWER_TERMS if term in lowered)
    if re.search(r"\b(story|anecdote|example|remember|said|years later|then)\b", lowered):
        score += 8
    if re.search(r"\b(therefore|because|so that|in order that|and so)\b", lowered):
        score += 4
    if re.search(r"[.!?][\"')\]]?$", text.strip()):
        score += 3
    if re.search(r"\b(announcement|subscribe|youtube|housekeeping)\b", lowered):
        score -= 20
    return score


def heuristic_select_clips(
    units: List[Dict[str, Any]],
    min_clips: int,
    max_clips: int,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    chapter_clips = chapter_based_heuristic_select_clips(units, context or {}, min_clips, max_clips)
    if len(chapter_clips) >= min_clips:
        return chapter_clips

    candidates: List[Dict[str, Any]] = []
    for start_i in range(0, len(units), 2):
        for end_i in range(start_i, len(units)):
            duration = units[end_i]["end"] - units[start_i]["start"]
            if duration < 30.0:
                continue
            if duration > MAX_SHORT_SECONDS:
                break
            if duration < 42.0 and not re.search(r"[.!?][\"')\]]?$", units[end_i]["text"]):
                continue
            text = " ".join(unit["text"] for unit in units[start_i : end_i + 1])
            point_key, point_title, point_score = classify_major_point(text)
            if point_score <= 0:
                continue
            score = score_heuristic_candidate(text, duration, point_score)
            if point_key == "father_doyle_story":
                if units[start_i]["index"] <= 2:
                    score += 24
                else:
                    score -= 18
                if "hope" in text.lower():
                    score += 18
                if "repented" in text.lower():
                    score += 10
                if duration < 78.0:
                    score -= 18
            candidates.append(
                {
                    "score": score,
                    "point_key": point_key,
                    "major_point_title": point_title,
                    "text": text,
                    "start_segment": units[start_i]["index"],
                    "end_segment": units[end_i]["index"],
                    "start": units[start_i]["start"],
                    "end": units[end_i]["end"],
                    "duration": duration,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    clips: List[Dict[str, Any]] = []
    used_points = set()
    used_ranges: List[Tuple[float, float]] = []
    for candidate in candidates:
        if len(clips) >= max_clips:
            break
        if candidate["point_key"] in used_points:
            continue
        start = float(candidate["start"])
        end = float(candidate["end"])
        if substantially_overlaps(start, end, used_ranges):
            continue

        point_number = len(clips) + 1
        power_quote = extract_power_quote(candidate["text"])
        title = candidate["major_point_title"]
        clips.append(
            {
                "major_point_id": f"point-{point_number:02d}",
                "major_point_title": title,
                "title": titlewrap(title),
                "mdx_section_title": titlewrap(title),
                "theme": titlewrap(candidate["text"], 180),
                "selection_reason": "Representative complete moment for this major homily point.",
                "power_quote": power_quote,
                "start_segment": candidate["start_segment"],
                "end_segment": candidate["end_segment"],
                "display_words": titlewrap(power_quote, 260),
                "keywords": ["Latin Mass", "Traditional Catholic", "homily"],
            }
        )
        used_points.add(candidate["point_key"])
        used_ranges.append((start, end))

    clips.sort(key=lambda clip: int(clip["start_segment"]))
    for index, clip in enumerate(clips, 1):
        clip["major_point_id"] = f"point-{index:02d}"

    if len(clips) < min_clips:
        print(f"Warning: heuristic selector found only {len(clips)} strong point-first clip(s).")
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
    used_major_points = set()
    used_point_clusters = set()
    used_titles = set()
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
        allow_short = raw_truthy(raw.get("allow_short") or raw.get("exceptional_short"))
        if duration > MAX_SHORT_SECONDS or duration < (MIN_SHORT_SECONDS if allow_short else 30.0):
            continue
        if any(overlap_seconds(start, end, used_start, used_end) > 1.5 for used_start, used_end in used_ranges):
            continue

        clip_words = words_between(words, start, end)
        text = " ".join(unit["text"] for unit in units if start_segment <= unit["index"] <= end_segment)
        title = titlewrap(
            clean_editorial_title(
                raw.get("title") or raw.get("mdx_section_title") or text,
                text,
            )
        )
        inferred_point_key, inferred_point_title, _ = classify_major_point(text)
        major_point_title = titlewrap(
            raw.get("major_point_title")
            or raw.get("mdx_section_title")
            or inferred_point_title
            or title
        )
        title_key = text_signature(title)
        point_key = text_signature(major_point_title) or inferred_point_key
        distinct_subpoint = raw_truthy(raw.get("distinct_subpoint"))
        explicit_major_point = bool(raw.get("major_point_title") or raw.get("mdx_section_title"))
        if not distinct_subpoint and point_key and point_key in used_major_points:
            continue
        classified_point_key = inferred_point_key if inferred_point_key != "general_homily_point" else ""
        if not explicit_major_point and not distinct_subpoint and classified_point_key and classified_point_key in used_point_clusters:
            continue
        if title_key and title_key in used_titles:
            continue

        power_quote = titlewrap(raw.get("power_quote") or extract_power_quote(text), 180)
        display_words = titlewrap(raw.get("display_words") or power_quote or text, 260)
        clip_id = f"clip-{len(clips) + 1:02d}"
        slug = slugify(title, clip_id)
        image_path = os.path.join(paths["images_dir"], f"{clip_id}-{slug}.jpg")
        audio_path = os.path.join(paths["audio_dir"], f"{clip_id}.mp3")
        video_path = os.path.join(paths["videos_dir"], f"{clip_id}-{slug}.mp4")
        captions_path = os.path.join(paths["captions_dir"], f"{clip_id}.srt")
        theme = titlewrap(raw.get("theme") or text, 180)
        caption_groups = make_caption_groups(clip_words, start, end)

        clips.append(
            {
                "id": clip_id,
                "major_point_id": f"point-{len(clips) + 1:02d}",
                "major_point_title": major_point_title,
                "title": title,
                "mdx_section_title": titlewrap(raw.get("mdx_section_title") or major_point_title or title),
                "theme": theme,
                "selection_reason": titlewrap(raw.get("selection_reason") or "Representative complete moment for this major homily point.", 180),
                "power_quote": power_quote,
                "start": round(start, 3),
                "end": round(end, 3),
                "clip_key": hashlib.sha1(f"{round(start, 3):.3f}-{round(end, 3):.3f}".encode("utf-8")).hexdigest()[:12],
                "duration": duration,
                "display_words": display_words,
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
        if point_key:
            used_major_points.add(point_key)
        if not explicit_major_point and classified_point_key:
            used_point_clusters.add(classified_point_key)
        if title_key:
            used_titles.add(title_key)

    clips.sort(key=lambda clip: float(clip.get("start", 0.0) or 0.0))
    for index, clip in enumerate(clips, 1):
        clip_id = f"clip-{index:02d}"
        slug = slugify(clip.get("title") or clip_id, clip_id)
        clip["id"] = clip_id
        clip["major_point_id"] = f"point-{index:02d}"
        clip["output_paths"] = {
            "image": os.path.join(paths["images_dir"], f"{clip_id}-{slug}.jpg"),
            "audio": os.path.join(paths["audio_dir"], f"{clip_id}.mp3"),
            "video": os.path.join(paths["videos_dir"], f"{clip_id}-{slug}.mp4"),
            "captions": os.path.join(paths["captions_dir"], f"{clip_id}.srt"),
        }

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
            "max_clips": max_clips,
            "selection_mode": selection_mode,
        },
        "clips": clips,
    }


def cap_manifest_clips(manifest: Dict[str, Any], max_clips: int) -> Dict[str, Any]:
    clips = manifest.get("clips") or []
    if not max_clips or len(clips) <= max_clips:
        return manifest

    capped = dict(manifest)
    capped["clips"] = clips[:max_clips]
    settings = dict(capped.get("settings") or {})
    settings["max_clips"] = max_clips
    settings["capped_from_clips"] = len(clips)
    capped["settings"] = settings
    print(f"Using first {max_clips} clip(s) from manifest; manifest contains {len(clips)}.")
    return capped


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clip_key(clip: Dict[str, Any]) -> str:
    start = round(float(clip.get("start", 0.0) or 0.0), 3)
    end = round(float(clip.get("end", 0.0) or 0.0), 3)
    basis = f"{start:.3f}-{end:.3f}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def canonical_image_path(clip: Dict[str, Any], paths: Dict[str, str]) -> str:
    slug = slugify(clip.get("title") or clip.get("id") or "clip", clip.get("id") or "clip")
    return os.path.join(paths["images_dir"], f"{clip['id']}-{slug}.jpg")


def image_assets_path(paths: Dict[str, str]) -> str:
    return os.path.join(paths["images_dir"], "image_assets.json")


def load_image_assets(paths: Dict[str, str]) -> Dict[str, Any]:
    path = image_assets_path(paths)
    if os.path.exists(path):
        try:
            return load_manifest(path)
        except Exception:
            pass
    return {"version": 1, "assets": []}


def save_image_assets(paths: Dict[str, str], assets: Dict[str, Any]) -> None:
    assets["version"] = 1
    save_json(image_assets_path(paths), assets)


def asset_by_clip_key(assets: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    for asset in assets.get("assets") or []:
        if asset.get("clip_key") == key:
            return asset
    return None


def upsert_image_asset(assets: Dict[str, Any], record: Dict[str, Any]) -> None:
    existing = asset_by_clip_key(assets, record["clip_key"])
    if existing:
        existing.update(record)
        return
    assets.setdefault("assets", []).append(record)


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


def ensure_clip_image(
    clip: Dict[str, Any],
    paths: Dict[str, str],
    assets: Dict[str, Any],
    refresh: bool = False,
) -> str:
    key = clip_key(clip)
    clip["clip_key"] = key
    canonical_path = canonical_image_path(clip, paths)
    previous_path = clip["output_paths"].get("image")
    asset = asset_by_clip_key(assets, key)

    if not refresh:
        if os.path.exists(canonical_path):
            clip["output_paths"]["image"] = canonical_path
            upsert_image_asset(
                assets,
                {
                    "clip_id": clip.get("id"),
                    "clip_key": key,
                    "title": clip.get("title"),
                    "title_slug": slugify(clip.get("title"), clip.get("id", "clip")),
                    "image_path": canonical_path,
                    "image_prompt": traditional_catholic_image_prompt(clip.get("image_prompt")),
                    "manual_or_existing": True,
                },
            )
            return canonical_path

        asset_path = str((asset or {}).get("image_path") or "")
        if asset_path and os.path.exists(asset_path):
            os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
            shutil.copy2(asset_path, canonical_path)
            clip["output_paths"]["image"] = canonical_path
            asset["image_path"] = canonical_path
            asset["title"] = clip.get("title")
            asset["title_slug"] = slugify(clip.get("title"), clip.get("id", "clip"))
            return canonical_path

        if previous_path and os.path.exists(previous_path):
            os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
            shutil.copy2(previous_path, canonical_path)
            clip["output_paths"]["image"] = canonical_path
            upsert_image_asset(
                assets,
                {
                    "clip_id": clip.get("id"),
                    "clip_key": key,
                    "title": clip.get("title"),
                    "title_slug": slugify(clip.get("title"), clip.get("id", "clip")),
                    "image_path": canonical_path,
                    "image_prompt": traditional_catholic_image_prompt(clip.get("image_prompt")),
                    "manual_or_existing": True,
                },
            )
            return canonical_path

    clip["output_paths"]["image"] = canonical_path
    generated_path = save_clip_image(clip)
    upsert_image_asset(
        assets,
        {
            "clip_id": clip.get("id"),
            "clip_key": key,
            "title": clip.get("title"),
            "title_slug": slugify(clip.get("title"), clip.get("id", "clip")),
            "image_path": generated_path,
            "image_prompt": traditional_catholic_image_prompt(clip.get("image_prompt")),
            "image_size": list(IMAGE_SIZE),
            "model": os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "refreshed": bool(refresh),
        },
    )
    return generated_path


def make_background(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Short image is not readable: {image_path}") from exc

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", CANVAS_SIZE, "black")
    canvas.paste(image, (0, IMAGE_TOP))

    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    gradient_height = GRADIENT_BOTTOM - GRADIENT_TOP
    for offset in range(gradient_height):
        alpha = int(255 * (1 - ((offset + 1) / gradient_height)))
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
            "major_point_id": clip.get("major_point_id", ""),
            "major_point_title": clip.get("major_point_title", ""),
            "title": clip["title"],
            "description": clip["theme"],
            "selection_reason": clip.get("selection_reason", ""),
            "power_quote": clip.get("power_quote", ""),
            "tags": clip.get("keywords") or [],
            "start": clip["start"],
            "end": clip["end"],
            "clip_key": clip.get("clip_key") or clip_key(clip),
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

    if not requested_ids and args.max_clips:
        clips = clips[: args.max_clips]

    if args.limit:
        clips = clips[: args.limit]
    return clips


def refresh_image_ids(args: argparse.Namespace) -> set:
    ids = set()
    for value in args.refresh_image or []:
        ids.update(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    return ids


def create_or_load_manifest(args: argparse.Namespace, paths: Dict[str, str]) -> Dict[str, Any]:
    if args.render_only:
        if not os.path.exists(paths["manifest"]):
            raise FileNotFoundError(f"--render-only requested but manifest is missing: {paths['manifest']}")
        return cap_manifest_clips(load_manifest(paths["manifest"]), args.max_clips)

    if os.path.exists(paths["manifest"]) and not args.force_manifest:
        print(f"Using existing manifest: {paths['manifest']}")
        return cap_manifest_clips(load_manifest(paths["manifest"]), args.max_clips)

    script = load_video_script(paths["video_script"])
    segments = script["segments"]
    units = segment_units(segments)
    words = collect_words(segments)
    context = load_mdx_context(paths["mdx"])

    effective_selection_mode = args.selection_mode

    if args.selection_mode == "heuristic":
        raw_clips = heuristic_select_clips(units, args.min_clips, args.max_clips, context)
    else:
        try:
            raw_clips = openai_select_clips(units, context, args.min_clips, args.max_clips)
        except Exception as exc:
            if args.selection_mode == "openai":
                raise
            print(f"OpenAI clip selection failed; using heuristic fallback. Reason: {exc}")
            raw_clips = heuristic_select_clips(units, args.min_clips, args.max_clips, context)
            effective_selection_mode = "heuristic-fallback"

    manifest = build_manifest(raw_clips, units, words, paths, args.max_clips, effective_selection_mode)
    if (
        args.selection_mode == "auto"
        and len(manifest.get("clips", [])) < args.min_clips
    ):
        needed = args.min_clips - len(manifest.get("clips", []))
        print(
            f"OpenAI selection produced {len(manifest.get('clips', []))} valid clip(s); "
            f"supplementing with {needed} heuristic clip(s)."
        )
        heuristic_raw = heuristic_select_clips(units, args.min_clips, args.max_clips, context)
        for supplement_count in range(needed, len(heuristic_raw) + 1):
            supplemented_raw = raw_clips + heuristic_raw[:supplement_count]
            manifest = build_manifest(
                supplemented_raw,
                units,
                words,
                paths,
                args.max_clips,
                "openai+heuristic-fallback",
            )
            if len(manifest.get("clips", [])) >= args.min_clips:
                raw_clips = supplemented_raw
                break

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
    manifest = cap_manifest_clips(manifest, args.max_clips)
    clips = selected_clips(args, manifest.get("clips") or [])
    if not clips:
        raise RuntimeError("No clips available to render.")

    refresh_ids = refresh_image_ids(args)
    all_clip_ids = {clip.get("id") for clip in manifest.get("clips") or []}
    missing_refresh_ids = sorted(refresh_ids - all_clip_ids)
    if missing_refresh_ids:
        raise ValueError(f"Requested refresh image id(s) not found in manifest: {', '.join(missing_refresh_ids)}")
    unselected_refresh_ids = sorted(refresh_ids - {clip.get("id") for clip in clips})
    if unselected_refresh_ids:
        raise ValueError(
            "Requested refresh image id(s) are not selected for this run: "
            + ", ".join(unselected_refresh_ids)
        )

    image_assets = load_image_assets(paths)
    print("Preparing clip images...")
    for clip in clips:
        refresh = args.refresh_all_images or clip.get("id") in refresh_ids
        image_path = ensure_clip_image(clip, paths, image_assets, refresh=refresh)
        action = "Refreshed" if refresh else "Image ready"
        print(f"{action}: {image_path}")
    save_image_assets(paths, image_assets)
    save_json(paths["manifest"], manifest)

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
    parser.add_argument("--force-media", action="store_true", help="Regenerate audio/videos even if files exist. Images are reused unless refreshed.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected clips.")
    parser.add_argument("--clip-id", action="append", help="Process a specific clip id, such as clip-02. Can be repeated or comma-separated.")
    parser.add_argument("--refresh-image", action="append", help="Regenerate image for a specific clip id, such as clip-02. Can be repeated or comma-separated.")
    parser.add_argument("--refresh-all-images", action="store_true", help="Regenerate images for every selected clip.")
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
    parser.add_argument("--image-workers", type=int, default=2, help="Deprecated compatibility option; image assets are prepared serially.")
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
