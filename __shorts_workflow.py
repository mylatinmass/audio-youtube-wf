import argparse
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydub import AudioSegment

try:
    from audio_clip import clip_audio_segment
except Exception:
    clip_audio_segment = None

try:
    from thumbnail_generator import (
        DEFAULT_IMAGE_MODEL,
        _center_crop_to_ratio,
        _generate_image_with_compatible_kwargs,
        _image_bytes_from_response,
        _openai_client,
        _supported_image_kwargs,
    )
except Exception:
    DEFAULT_IMAGE_MODEL = "gpt-image-1"
    _openai_client = None
    _generate_image_with_compatible_kwargs = None
    _image_bytes_from_response = None
    _supported_image_kwargs = None


CANVAS_SIZE = (1080, 1920)
IMAGE_SIZE = (1080, 1350)
IMAGE_RATIO = 4 / 5
IMAGE_TOP = 570
TEXT_BOX_BOTTOM = 570
GRADIENT_TOP = 570
GRADIENT_BOTTOM = 1020
FPS = 30
MIN_SHORT_SECONDS = 30.0
MAX_SHORT_SECONDS = 90.0
DEFAULT_MAX_CLIPS = 10
DEFAULT_MIN_CLIPS = 4
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


@dataclass
class Paths:
    root: str
    source_json: str
    audio: str
    output_dir: str
    analysis: str
    manifest: str
    upload_metadata: str
    images_dir: str
    audio_dir: str
    videos_dir: str
    captions_dir: str


def clean_path(path: str) -> str:
    return str(path or "").strip().strip('"').strip("'")


def slugify(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value or "").strip("-").lower()
    return value or fallback


def titlewrap(text: str, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit].strip()


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


def format_short_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(source_json: str, audio_path: str = "", output_dir: str = "") -> Paths:
    source_json = os.path.abspath(os.path.expanduser(clean_path(source_json)))
    if not os.path.exists(source_json):
        raise FileNotFoundError(f"Input JSON does not exist: {source_json}")

    root = os.path.dirname(source_json)
    output_dir = os.path.abspath(os.path.expanduser(clean_path(output_dir))) if output_dir else os.path.join(root, "Video Clips")
    audio_path = os.path.abspath(os.path.expanduser(clean_path(audio_path))) if audio_path else ""

    return Paths(
        root=root,
        source_json=source_json,
        audio=audio_path,
        output_dir=output_dir,
        analysis=os.path.join(output_dir, "shorts_analysis.json"),
        manifest=os.path.join(output_dir, "shorts_manifest.json"),
        upload_metadata=os.path.join(output_dir, "upload_metadata.json"),
        images_dir=os.path.join(output_dir, "images"),
        audio_dir=os.path.join(output_dir, "audio"),
        videos_dir=os.path.join(output_dir, "videos"),
        captions_dir=os.path.join(output_dir, "captions"),
    )


def ensure_output_dirs(paths: Paths) -> None:
    for path in [paths.output_dir, paths.images_dir, paths.audio_dir, paths.videos_dir, paths.captions_dir]:
        os.makedirs(path, exist_ok=True)


def normalize_segment(segment: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
    if not text:
        return None
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    if end <= start:
        return None
    return {"index": index, "start": start, "end": end, "text": text, "words": segment.get("words") or []}


def load_segments(source_json: str) -> List[Dict[str, Any]]:
    data = load_json(source_json)
    raw_segments = data.get("homily_segments") or data.get("segments") or []
    if not raw_segments:
        raise ValueError("JSON must contain either 'homily_segments' or 'segments'.")

    segments: List[Dict[str, Any]] = []
    for i, segment in enumerate(raw_segments, 1):
        normalized = normalize_segment(segment, i)
        if normalized:
            segments.append(normalized)

    if not segments:
        raise ValueError("No usable timestamped segments found.")
    return segments


def collect_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            start = float(word.get("start", segment["start"]) or segment["start"])
            end = float(word.get("end", start) or start)
            words.append({"word": text, "start": start, "end": end})
    return words


def build_timed_transcript(segments: List[Dict[str, Any]], max_chars: int = 65000) -> str:
    lines = []
    for segment in segments:
        lines.append(
            f'{segment["index"]:04d} [{format_timestamp(segment["start"])}-{format_timestamp(segment["end"])}] {segment["text"]}'
        )
    transcript = "\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    head = transcript[: max_chars // 2]
    tail = transcript[-max_chars // 2 :]
    return head.rstrip() + "\n\n[...middle omitted for prompt length...]\n\n" + tail.lstrip()


def seconds_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def clean_editorial_title(title: str, source_text: str, limit: int = 70) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    source_text = re.sub(r"\s+", " ", str(source_text or "")).strip()
    title_words = re.findall(r"[A-Za-z0-9']+", title.lower())
    source_words = re.findall(r"[A-Za-z0-9']+", source_text.lower())

    if title_words and source_words[: len(title_words)] == title_words:
        if "mass" in source_text.lower():
            return "Always the Mass"
        if "charity" in source_text.lower():
            return "Charity Is Not Niceness"
        if "life" in source_text.lower() and "not ours" in source_text.lower():
            return "Our Lives Are Not Our Own"
        meaningful = [
            word for word in source_words
            if word not in {"the", "a", "an", "and", "or", "but", "so", "that", "there", "this", "is", "are", "was", "were", "of", "from", "to", "in", "on", "with", "for", "as", "it", "we", "our"}
        ]
        if meaningful:
            return titlewrap(" ".join(w.capitalize() for w in meaningful[:7]), limit)
    return titlewrap(title, limit)


def extract_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_power_quote(text: str, fallback_limit: int = 180) -> str:
    sentences = extract_sentences(text)
    power_terms = [
        "always", "must", "cannot", "not ours", "belongs to god", "charity", "mass", "cross", "sacrifice", "grace", "christ", "soul", "souls", "love", "mercy", "truth", "conversion",
    ]
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        if len(words) < 4 or len(words) > 32:
            continue
        lowered = sentence.lower()
        score = sum(4 for term in power_terms if term in lowered)
        score += min(len(words), 20) / 10
        if re.search(r"\b(how|why|cannot|must|never|always|therefore|only)\b", lowered):
            score += 3
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return titlewrap(best_sentence or text, fallback_limit)


def clip_text(segments: List[Dict[str, Any]], start_segment: int, end_segment: int) -> str:
    return " ".join(s["text"] for s in segments if start_segment <= int(s["index"]) <= end_segment)


def clip_times_from_segments(segments: List[Dict[str, Any]], start_segment: int, end_segment: int) -> Tuple[float, float]:
    by_index = {int(s["index"]): s for s in segments}
    if start_segment not in by_index or end_segment not in by_index:
        raise ValueError(f"Invalid segment range: {start_segment}-{end_segment}")
    if end_segment < start_segment:
        start_segment, end_segment = end_segment, start_segment
    return float(by_index[start_segment]["start"]), float(by_index[end_segment]["end"])


def openai_analyze_clips(segments: List[Dict[str, Any]], min_clips: int, max_clips: int) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or OPENAI_KEY must be set for AI analysis.")

    model = os.getenv("OPENAI_SHORTS_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)
    transcript = build_timed_transcript(segments)

    prompt = f"""
You are a YouTube Shorts editor for Catholic sermon/homily audio.

Analyze the timed transcript and identify only the strongest 30 to 90 second clips.
The clips must work as independent short, impactful videos.

Return ONLY valid JSON in this exact shape:
{{
  "summary": {{
    "total_clips_identified": 0,
    "recommended_to_produce": 0,
    "overall_notes": ""
  }},
  "clips": [
    {{
      "rank": 1,
      "start_segment": 1,
      "end_segment": 10,
      "start_time": "0:00",
      "end_time": "0:45",
      "duration_seconds": 45,
      "title": "Original editorial title, not the first transcript line",
      "strength_score": 10,
      "clip_type": "story|teaching|warning|exhortation|reflection",
      "why_it_works": "Why this stands alone as a strong short.",
      "hook": "The opening idea that pulls the viewer in.",
      "power_quote": "The strongest exact line or phrase from the clip.",
      "display_words": "Concise on-screen quote or phrase.",
      "thumbnail_idea": "Concrete thumbnail concept.",
      "image_prompt": "Detailed 4:5 image prompt with no text in the image.",
      "youtube_description": "One sentence description.",
      "keywords": ["keyword", "keyword"],
      "selected": true
    }}
  ]
}}

Selection rules:
- First find the sermon’s major movements.
- Select no more than {max_clips} clips. This is a ceiling, not a target.
- Try to select at least {min_clips} clips only if the sermon has enough strong moments.
- Do not create filler clips.
- Each clip must be between 30 and 90 seconds.
- A clip must have a clear beginning, middle, and payoff.
- If a story appears, keep the whole story together when possible.
- Avoid overlapping clips.
- Do not choose multiple clips from the same exact idea unless they are truly distinct.
- Titles must be original editorial titles, not the first words of the transcript.
- Favor practical, memorable, visual, spiritually strong clips.
- Include image_prompt in light watercolor and ink style unless another style is clearly better.
- No image prompt should request text, captions, logos, or watermarks.
- Use start_segment and end_segment numbers from the transcript.
- strength_score is 1 to 10, where 10 is excellent.
- selected should be true only for clips worth producing.

Traditional Catholic image guardrail for image prompts:
{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL}

Timed transcript:
{transcript}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_completion_tokens=6000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a careful Catholic sermon editor. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    payload = json.loads(raw)
    if not isinstance(payload.get("clips"), list):
        raise RuntimeError("AI returned JSON without a clips list.")
    return payload


def heuristic_analyze_clips(segments: List[Dict[str, Any]], min_clips: int, max_clips: int) -> Dict[str, Any]:
    power_terms = [
        "our lives are not our own", "we don't belong to ourselves", "what we can control", "morning offering", "always the mass", "sacrifice of christ", "drop represents", "charity is not", "we must disappear", "he must increase", "love of god", "cross", "grace", "altar",
    ]
    candidates: List[Dict[str, Any]] = []

    for start_i in range(len(segments)):
        for end_i in range(start_i, len(segments)):
            start = float(segments[start_i]["start"])
            end = float(segments[end_i]["end"])
            duration = end - start
            if duration < MIN_SHORT_SECONDS:
                continue
            if duration > MAX_SHORT_SECONDS:
                break
            text = " ".join(s["text"] for s in segments[start_i : end_i + 1])
            lowered = text.lower()
            score = sum(6 for term in power_terms if term in lowered)
            score += max(0.0, 20.0 - abs(duration - 60.0) * 0.3)
            if re.search(r"\b(must|cannot|always|therefore|why|how|only)\b", lowered):
                score += 3
            if re.search(r"[.!?][\"')\]]?$", text.strip()):
                score += 2
            if score >= 20:
                candidates.append({
                    "score": score,
                    "start_segment": int(segments[start_i]["index"]),
                    "end_segment": int(segments[end_i]["index"]),
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "text": text,
                })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    used_ranges: List[Tuple[float, float]] = []
    clips: List[Dict[str, Any]] = []

    for candidate in candidates:
        if len(clips) >= max_clips:
            break
        start = candidate["start"]
        end = candidate["end"]
        if any(seconds_overlap(start, end, a, b) > 2.0 for a, b in used_ranges):
            continue
        text = candidate["text"]
        power_quote = extract_power_quote(text)
        title = infer_title_from_text(text)
        clips.append({
            "rank": len(clips) + 1,
            "start_segment": candidate["start_segment"],
            "end_segment": candidate["end_segment"],
            "start_time": format_short_time(start),
            "end_time": format_short_time(end),
            "duration_seconds": round(candidate["duration"], 1),
            "title": title,
            "strength_score": min(10, max(6, round(candidate["score"] / 8))),
            "clip_type": infer_clip_type(text),
            "why_it_works": "This excerpt has a clear spiritual point and can stand alone as a short clip.",
            "hook": titlewrap(extract_sentences(text)[0] if extract_sentences(text) else text, 140),
            "power_quote": power_quote,
            "display_words": titlewrap(power_quote, 160),
            "thumbnail_idea": infer_thumbnail_idea(text),
            "image_prompt": prompt_for_clip_image(title, text),
            "youtube_description": titlewrap(text, 160),
            "keywords": ["Catholic homily", "Traditional Catholic", "Catholic Shorts", "Latin Mass"],
            "selected": True,
        })
        used_ranges.append((start, end))

    clips.sort(key=lambda c: c["start_segment"])
    for i, clip in enumerate(clips, 1):
        clip["rank"] = i

    return {
        "summary": {
            "total_clips_identified": len(clips),
            "recommended_to_produce": len([c for c in clips if c.get("selected")]),
            "overall_notes": "Generated by heuristic fallback. Review titles and timing before rendering.",
        },
        "clips": clips,
    }


def infer_clip_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(story|anecdote|said|then|years later|remember)\b", lowered):
        return "story"
    if re.search(r"\b(warning|danger|enemy|persecution|hate|reject)\b", lowered):
        return "warning"
    if re.search(r"\b(must|need to|have to|let us|we should)\b", lowered):
        return "exhortation"
    if re.search(r"\b(is|means|represents|because|therefore)\b", lowered):
        return "teaching"
    return "reflection"


def infer_title_from_text(text: str) -> str:
    lowered = text.lower()
    if "our lives are not our own" in lowered or "we don't belong to ourselves" in lowered:
        return "Our Lives Are Not Our Own"
    if "what we can control" in lowered:
        return "What Can We Control?"
    if "morning offering" in lowered:
        return "Go Back to the Morning Offering"
    if "always the mass" in lowered:
        return "Always the Mass"
    if "one drop" in lowered or "drop represents" in lowered:
        return "The Drop of Water in the Chalice"
    if "charity has been reduced" in lowered or "charity is not" in lowered:
        return "Charity Is Not Niceness"
    if "disappear into christ" in lowered or "he must increase" in lowered:
        return "We Must Disappear Into Christ"
    if "world will" in lowered or "persecuted" in lowered:
        return "The World Will Not Accept This Message"
    return titlewrap(extract_power_quote(text), 70)


def infer_thumbnail_idea(text: str) -> str:
    lowered = text.lower()
    if "mass" in lowered or "altar" in lowered:
        return "A reverent altar and chalice glowing with sacred light."
    if "drop" in lowered or "chalice" in lowered:
        return "A single drop of water falling into a chalice."
    if "morning" in lowered:
        return "A person kneeling in morning prayer beside a bed."
    if "charity" in lowered:
        return "A cross with two people helping each other in sacrificial love."
    if "world" in lowered or "persecut" in lowered:
        return "A Christian standing firm while the dark world rejects the cross."
    return "A reverent Catholic scene with a person praying before a crucifix."


def prompt_for_clip_image(title: str, theme: str) -> str:
    return (
        "Create a 4:5 portrait illustration for a YouTube Short about:\n\n"
        f"Title: {title}\n"
        f"Theme: {titlewrap(theme, 300)}\n\n"
        "Use a reverent Catholic visual tone in light watercolor with ink accents. "
        f"{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL} "
        "The image should feel contemplative, concrete, and tied to the topic. "
        "Do not include text, captions, logos, watermarks, typography, or UI."
    )


def normalize_analysis_payload(payload: Dict[str, Any], segments: List[Dict[str, Any]], max_clips: int) -> Dict[str, Any]:
    by_index = {int(s["index"]): s for s in segments}
    normalized_clips: List[Dict[str, Any]] = []
    used_ranges: List[Tuple[float, float]] = []

    for raw in payload.get("clips") or []:
        if len(normalized_clips) >= max_clips:
            break
        try:
            start_segment = int(raw.get("start_segment"))
            end_segment = int(raw.get("end_segment"))
        except Exception:
            continue
        if start_segment not in by_index or end_segment not in by_index:
            continue
        if end_segment < start_segment:
            start_segment, end_segment = end_segment, start_segment

        start, end = clip_times_from_segments(segments, start_segment, end_segment)
        duration = end - start
        if duration < MIN_SHORT_SECONDS or duration > MAX_SHORT_SECONDS:
            continue
        if any(seconds_overlap(start, end, a, b) > 2.0 for a, b in used_ranges):
            continue

        text = clip_text(segments, start_segment, end_segment)
        title = clean_editorial_title(raw.get("title") or infer_title_from_text(text), text)
        power_quote = titlewrap(raw.get("power_quote") or extract_power_quote(text), 180)
        display_words = titlewrap(raw.get("display_words") or power_quote, 220)
        image_prompt = raw.get("image_prompt") or prompt_for_clip_image(title, text)
        if TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL not in image_prompt:
            image_prompt = f"{image_prompt}\n\n{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL}\nDo not include text or watermarks."

        normalized_clips.append({
            "rank": len(normalized_clips) + 1,
            "id": f"clip-{len(normalized_clips) + 1:02d}",
            "start_segment": start_segment,
            "end_segment": end_segment,
            "start": round(start, 3),
            "end": round(end, 3),
            "start_time": format_short_time(start),
            "end_time": format_short_time(end),
            "duration_seconds": round(duration, 1),
            "title": title,
            "strength_score": int(raw.get("strength_score") or 7),
            "clip_type": raw.get("clip_type") or infer_clip_type(text),
            "why_it_works": titlewrap(raw.get("why_it_works") or "This excerpt stands alone as a complete short clip.", 240),
            "hook": titlewrap(raw.get("hook") or (extract_sentences(text)[0] if extract_sentences(text) else text), 180),
            "power_quote": power_quote,
            "display_words": display_words,
            "thumbnail_idea": raw.get("thumbnail_idea") or infer_thumbnail_idea(text),
            "image_prompt": image_prompt,
            "youtube_description": titlewrap(raw.get("youtube_description") or text, 220),
            "keywords": list(raw.get("keywords") or ["Catholic homily", "Traditional Catholic", "Catholic Shorts"]),
            "selected": bool(raw.get("selected", True)),
            "source_text": text,
        })
        used_ranges.append((start, end))

    summary = payload.get("summary") or {}
    return {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_clips_identified": len(normalized_clips),
            "recommended_to_produce": len([c for c in normalized_clips if c.get("selected")]),
            "overall_notes": summary.get("overall_notes") or "Review selected clips before rendering.",
        },
        "clips": normalized_clips,
    }


def print_analysis_table(analysis: Dict[str, Any]) -> None:
    clips = analysis.get("clips") or []
    if not clips:
        print("No clips identified.")
        return

    print("\nIdentified Shorts:\n")
    print(f"{'ID':<8} {'Time':<15} {'Len':<7} {'Score':<7} {'Sel':<5} Title")
    print("-" * 80)
    for clip in clips:
        time_range = f"{clip.get('start_time')}-{clip.get('end_time')}"
        print(
            f"{clip.get('id', ''):<8} {time_range:<15} {clip.get('duration_seconds', ''):<7} "
            f"{clip.get('strength_score', ''):<7} {str(clip.get('selected', False)):<5} {clip.get('title', '')}"
        )
    print("\nEdit shorts_analysis.json to change selected true/false, titles, image prompts, or timing before rendering.\n")


def analysis_to_manifest(analysis: Dict[str, Any], paths: Paths) -> Dict[str, Any]:
    clips = []
    for clip in analysis.get("clips") or []:
        if not clip.get("selected", True):
            continue
        clip_id = clip.get("id") or f"clip-{len(clips) + 1:02d}"
        slug = slugify(clip.get("title"), clip_id)
        key = hashlib.sha1(f"{clip.get('start')}-{clip.get('end')}-{clip.get('title')}".encode("utf-8")).hexdigest()[:12]
        clips.append({
            **clip,
            "id": clip_id,
            "clip_key": key,
            "duration": float(clip.get("duration_seconds") or 0),
            "caption_groups": [],
            "output_paths": {
                "image": os.path.join(paths.images_dir, f"{clip_id}-{slug}.jpg"),
                "audio": os.path.join(paths.audio_dir, f"{clip_id}.mp3"),
                "video": os.path.join(paths.videos_dir, f"{clip_id}-{slug}.mp4"),
                "captions": os.path.join(paths.captions_dir, f"{clip_id}.srt"),
            },
        })

    return {
        "version": 2,
        "source": {"source_json": paths.source_json, "audio": paths.audio, "root": paths.root},
        "settings": {
            "canvas_size": list(CANVAS_SIZE),
            "image_size": list(IMAGE_SIZE),
            "min_seconds": MIN_SHORT_SECONDS,
            "max_seconds": MAX_SHORT_SECONDS,
        },
        "clips": clips,
    }


def nearest_word_time(words: List[Dict[str, Any]], target: float, kind: str) -> float:
    if not words:
        return float(target)
    key = "start" if kind == "start" else "end"
    return float(min(words, key=lambda w: abs(float(w.get(key, 0.0)) - target)).get(key, target))


def words_between(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [
        word for word in words
        if float(word.get("start", 0.0)) >= start - 0.05 and float(word.get("end", 0.0)) <= end + 0.15
    ]


def make_caption_groups(clip_words: List[Dict[str, Any]], clip_start: float, clip_end: float, fallback_text: str = "") -> List[Dict[str, Any]]:
    if not clip_words:
        sentences = extract_sentences(fallback_text)
        if not sentences:
            return []
        duration = clip_end - clip_start
        groups = []
        current_time = clip_start
        for sentence in sentences[:20]:
            group_duration = min(4.0, max(1.5, duration / max(1, len(sentences))))
            groups.append({"start": round(current_time, 3), "end": round(min(clip_end, current_time + group_duration), 3), "text": titlewrap(sentence, 80)})
            current_time += group_duration
            if current_time >= clip_end:
                break
        return groups

    groups: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(w["word"].strip() for w in current).strip()
        groups.append({
            "start": round(float(current[0]["start"]), 3),
            "end": round(float(current[-1]["end"]), 3),
            "text": re.sub(r"\s+", " ", text),
        })
        current = []

    for word in clip_words:
        current.append(word)
        text = " ".join(w["word"].strip() for w in current)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        if len(current) >= 9 or len(text) >= 56 or duration >= 3.7 or re.search(r"[.!?]$", word["word"].strip()):
            flush()
    flush()
    return groups


def write_srt(clip: Dict[str, Any]) -> None:
    lines = []
    clip_start = float(clip["start"])
    for index, group in enumerate(clip.get("caption_groups") or [], 1):
        text = str(group.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(group["start"]) - clip_start)
        end = max(start + 0.2, float(group["end"]) - clip_start)
        lines.extend([
            str(index),
            f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}",
            text,
            "",
        ])
    output_path = clip["output_paths"]["captions"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


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


def make_placeholder_image(path: str, title: str, prompt: str) -> str:
    image = Image.new("RGB", IMAGE_SIZE, (20, 20, 20))
    draw = ImageDraw.Draw(image)
    font_title = load_font(56, bold=True, serif=True)
    font_small = load_font(30, bold=False, serif=False)
    lines = wrap_text(title.upper(), font_title, IMAGE_SIZE[0] - 120)[:4]
    y = 180
    for line in lines:
        bbox = font_title.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font_title, fill=(230, 230, 230))
        y += 76
    small = wrap_text("Image prompt saved in analysis JSON", font_small, IMAGE_SIZE[0] - 160)[:2]
    y = IMAGE_SIZE[1] - 220
    for line in small:
        bbox = font_small.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font_small, fill=(170, 170, 170))
        y += 42
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="JPEG", quality=92)
    prompt_path = os.path.splitext(path)[0] + ".prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    return path


def generate_clip_image(clip: Dict[str, Any], refresh: bool = False, use_openai_image: bool = True) -> str:
    output_path = clip["output_paths"]["image"]
    if os.path.exists(output_path) and not refresh:
        return output_path

    prompt = clip.get("image_prompt") or prompt_for_clip_image(clip.get("title", "Short"), clip.get("source_text", ""))
    if not use_openai_image or _openai_client is None:
        return make_placeholder_image(output_path, clip.get("title", "Short"), prompt)

    try:
        client = _openai_client()
        model = os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
        kwargs = {
            "model": model,
            "prompt": prompt,
            "size": os.getenv("OPENAI_SHORTS_IMAGE_SIZE", "1024x1536"),
            "quality": os.getenv("OPENAI_IMAGE_QUALITY", "high"),
            "output_format": "jpeg",
            "output_compression": 92,
            "response_format": "b64_json",
            "n": 1,
        }
        if _supported_image_kwargs:
            kwargs = _supported_image_kwargs(client.images.generate, kwargs)
        if str(model).startswith("gpt-image"):
            kwargs.pop("response_format", None)
        response = _generate_image_with_compatible_kwargs(client, kwargs) if _generate_image_with_compatible_kwargs else client.images.generate(**kwargs)
        image_bytes = _image_bytes_from_response(response) if _image_bytes_from_response else BytesIO(response.data[0].b64_json).getvalue()
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = center_crop_to_ratio(image, IMAGE_RATIO)
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, format="JPEG", quality=92, optimize=True, progressive=True)
        return output_path
    except Exception as exc:
        print(f"Image generation failed for {clip.get('id')}; using placeholder. Reason: {exc}")
        return make_placeholder_image(output_path, clip.get("title", "Short"), prompt)


def make_background(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Short image is not readable: {image_path}") from exc

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
        widths, heights = [], []
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
    shadow = (0, 0, 0, 210)

    for line, width, height in zip(lines, widths, heights):
        x = safe_left + (safe_width - width) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_gap
    return overlay


def cut_audio(input_file: str, start_sec: float, end_sec: float, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if clip_audio_segment is not None:
        clip_audio_segment(input_file=input_file, start_sec=start_sec, end_sec=end_sec, output_file=output_file)
        return
    audio = AudioSegment.from_file(input_file)
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    audio[start_ms:end_ms].export(output_file, format="mp3")


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


def mix_background_music(speech_path: str, bg_audio_files: List[str], gain_db: float, fade_ms: int) -> Optional[str]:
    if not bg_audio_files:
        return None
    bg_path = random.choice(bg_audio_files)
    speech = AudioSegment.from_file(speech_path)
    duration_ms = len(speech)
    music = AudioSegment.from_file(bg_path)
    music = loop_audio_to_duration(music, duration_ms)
    music = music.set_frame_rate(speech.frame_rate).set_channels(speech.channels)
    music = music + float(gain_db)
    fade_ms = max(0, min(int(fade_ms), duration_ms // 2))
    if fade_ms:
        music = music.fade_in(fade_ms).fade_out(fade_ms)
    speech.overlay(music).export(speech_path, format="mp3")
    return bg_path


def render_clip(clip: Dict[str, Any], source_audio: str, bg_audio_files: List[str], bg_gain_db: float, bg_fade_ms: int, force: bool = False) -> str:
    output_path = clip["output_paths"]["video"]
    image_path = clip["output_paths"]["image"]
    if os.path.exists(output_path) and not force:
        return output_path

    audio_path = clip["output_paths"]["audio"]
    cut_audio(source_audio, float(clip["start"]), float(clip["end"]), audio_path)
    bg_audio_path = mix_background_music(audio_path, bg_audio_files, bg_gain_db, bg_fade_ms)
    if bg_audio_path:
        clip["background_audio"] = bg_audio_path

    write_srt(clip)

    duration = float(clip["end"]) - float(clip["start"])
    background = np.array(make_background(image_path))
    layers = [ImageClip(background, duration=duration)]

    title_duration = min(TITLE_SECONDS, max(1.4, duration * 0.18))
    layers.append(ImageClip(np.array(text_overlay(clip["title"], title=True)), duration=title_duration).with_start(0))

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
        layers.append(ImageClip(np.array(text_overlay(text, title=False)), duration=rel_end - rel_start).with_start(rel_start))

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


def write_upload_metadata(manifest: Dict[str, Any], paths: Paths) -> None:
    payload = {
        "source": manifest.get("source", {}),
        "clips": [
            {
                "id": clip.get("id"),
                "title": clip.get("title"),
                "description": clip.get("youtube_description") or clip.get("why_it_works"),
                "power_quote": clip.get("power_quote"),
                "tags": clip.get("keywords") or [],
                "start": clip.get("start"),
                "end": clip.get("end"),
                "duration": clip.get("duration_seconds") or clip.get("duration"),
                "video_path": clip.get("output_paths", {}).get("video"),
                "thumbnail_path": clip.get("output_paths", {}).get("image"),
                "captions_path": clip.get("output_paths", {}).get("captions"),
                "background_audio": clip.get("background_audio", ""),
            }
            for clip in manifest.get("clips", [])
        ],
    }
    save_json(paths.upload_metadata, payload)


def selected_clip_ids(args: argparse.Namespace) -> set:
    ids = set()
    for value in args.clip_id or []:
        ids.update(item.strip() for item in str(value).split(",") if item.strip())
    return ids


def prepare_manifest_for_render(paths: Paths, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not os.path.exists(paths.analysis):
        raise FileNotFoundError(f"Missing analysis file: {paths.analysis}. Run --analyze first.")
    analysis = load_json(paths.analysis)
    manifest = analysis_to_manifest(analysis, paths)
    words = collect_words(segments)
    for clip in manifest.get("clips") or []:
        clip_words = words_between(words, float(clip["start"]), float(clip["end"]))
        clip["caption_groups"] = make_caption_groups(clip_words, float(clip["start"]), float(clip["end"]), clip.get("source_text", ""))
    save_json(paths.manifest, manifest)
    return manifest


def run_analyze(args: argparse.Namespace, paths: Paths, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if os.path.exists(paths.analysis) and not args.force_analysis:
        analysis = load_json(paths.analysis)
        print(f"Using existing analysis: {paths.analysis}")
        print_analysis_table(analysis)
        return analysis

    if args.selection_mode == "heuristic":
        raw = heuristic_analyze_clips(segments, args.min_clips, args.max_clips)
    else:
        try:
            raw = openai_analyze_clips(segments, args.min_clips, args.max_clips)
        except Exception as exc:
            if args.selection_mode == "openai":
                raise
            print(f"OpenAI analysis failed; using heuristic fallback. Reason: {exc}")
            raw = heuristic_analyze_clips(segments, args.min_clips, args.max_clips)

    analysis = normalize_analysis_payload(raw, segments, args.max_clips)
    save_json(paths.analysis, analysis)
    print(f"Shorts analysis saved: {paths.analysis}")
    print_analysis_table(analysis)
    return analysis


def run_render(args: argparse.Namespace, paths: Paths, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not paths.audio:
        raise ValueError("Rendering requires --audio /path/to/homily_final.mp3")
    if not os.path.exists(paths.audio):
        raise FileNotFoundError(f"Audio file does not exist: {paths.audio}")

    manifest = prepare_manifest_for_render(paths, segments)
    requested_ids = selected_clip_ids(args)
    clips = manifest.get("clips") or []
    if requested_ids:
        clips = [c for c in clips if c.get("id") in requested_ids]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise RuntimeError("No selected clips available to render.")

    bg_audio_files = [] if args.no_bg_music else list_background_audio_files(args.bg_audio_dir)

    for clip in clips:
        print(f"Preparing {clip['id']}: {clip['title']} ({clip['start_time']}-{clip['end_time']})")
        generate_clip_image(clip, refresh=args.refresh_images, use_openai_image=not args.no_openai_images)
        if not args.images_only:
            video_path = render_clip(
                clip,
                paths.audio,
                bg_audio_files,
                args.bg_music_gain_db,
                args.bg_music_fade_ms,
                force=args.force_media,
            )
            print(f"Rendered: {video_path}")

    manifest["clips"] = clips if requested_ids or args.limit else manifest.get("clips", [])
    save_json(paths.manifest, manifest)
    write_upload_metadata(manifest, paths)
    print(f"Upload metadata saved: {paths.upload_metadata}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze homily transcripts and render YouTube Shorts.")
    parser.add_argument("source", nargs="?", help="Path to homily JSON with homily_segments or segments.")
    parser.add_argument("--audio", default="", help="Path to the full homily audio file. Required for rendering.")
    parser.add_argument("--output-dir", default="", help="Output folder. Defaults to ./Video Clips beside the JSON.")

    parser.add_argument("--analyze", action="store_true", help="Create shorts_analysis.json only.")
    parser.add_argument("--render", action="store_true", help="Render selected clips from shorts_analysis.json.")
    parser.add_argument("--all", action="store_true", help="Analyze, then render selected clips.")

    parser.add_argument("--force-analysis", action="store_true", help="Regenerate shorts_analysis.json.")
    parser.add_argument("--force-media", action="store_true", help="Regenerate audio/videos even if they already exist.")
    parser.add_argument("--images-only", action="store_true", help="Generate images and metadata but skip video rendering.")
    parser.add_argument("--refresh-images", action="store_true", help="Regenerate images for selected clips.")
    parser.add_argument("--no-openai-images", action="store_true", help="Use placeholder images instead of OpenAI image generation.")

    parser.add_argument("--min-clips", type=int, default=DEFAULT_MIN_CLIPS)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--limit", type=int, help="Render only the first N selected clips.")
    parser.add_argument("--clip-id", action="append", help="Render a specific clip id, such as clip-02. Can be repeated or comma-separated.")
    parser.add_argument("--selection-mode", choices=["auto", "openai", "heuristic"], default="auto")

    parser.add_argument("--bg-audio-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_audio_files"))
    parser.add_argument("--bg-music-gain-db", type=float, default=DEFAULT_BG_MUSIC_GAIN_DB)
    parser.add_argument("--bg-music-fade-ms", type=int, default=DEFAULT_BG_MUSIC_FADE_MS)
    parser.add_argument("--no-bg-music", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source:
        args.source = clean_path(input("Enter path to homily JSON: "))
    if args.min_clips < 1:
        raise ValueError("--min-clips must be at least 1.")
    if args.max_clips < args.min_clips:
        raise ValueError("--max-clips must be greater than or equal to --min-clips.")

    paths = resolve_paths(args.source, args.audio, args.output_dir)
    ensure_output_dirs(paths)
    segments = load_segments(paths.source_json)

    if not args.analyze and not args.render and not args.all:
        args.analyze = True

    if args.all:
        run_analyze(args, paths, segments)
        run_render(args, paths, segments)
        return

    if args.analyze:
        run_analyze(args, paths, segments)

    if args.render:
        run_render(args, paths, segments)


if __name__ == "__main__":
    main()
