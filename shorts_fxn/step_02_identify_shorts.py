import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


MIN_SHORT_SECONDS = 30
MAX_SHORT_SECONDS = 90
DEFAULT_MIN_CLIPS = 1
DEFAULT_MAX_CLIPS = 16


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_homily_json(homily_folder: str | Path) -> Path:
    folder = Path(homily_folder).expanduser().resolve()

    if folder.is_file() and folder.suffix.lower() == ".json":
        return folder

    candidates = [
        folder / "working" / "video_script.json",
        folder / "working" / "homily.json",
        folder / "video_script.json",
        folder / "homily.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in folder.rglob("*.json"):
        try:
            data = load_json(candidate)
            if data.get("segments") or data.get("homily_segments"):
                return candidate
        except Exception:
            continue

    raise FileNotFoundError(f"No usable homily JSON found in: {folder}")


def get_output_folder(homily_json_path: str | Path) -> Path:
    homily_json_path = Path(homily_json_path).resolve()
    parent = homily_json_path.parent

    if parent.name.lower() == "working":
        return parent.parent / "Video Clips"

    return parent / "Video Clips"


def normalize_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_segments = data.get("homily_segments") or data.get("segments") or []

    if not raw_segments:
        raise ValueError("Homily JSON must contain 'homily_segments' or 'segments'.")

    segments: List[Dict[str, Any]] = []

    for i, segment in enumerate(raw_segments, start=1):
        text = clean_text(segment.get("text", ""))

        if not text:
            continue

        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
        except Exception:
            continue

        if end <= start:
            continue

        segments.append(
            {
                "index": i,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    if not segments:
        raise ValueError("No valid timestamped segments found.")

    return segments


def build_timed_transcript(segments: List[Dict[str, Any]], max_chars: int = 65000) -> str:
    lines = []

    for segment in segments:
        lines.append(
            f'{segment["index"]:04d} '
            f'[{format_time(segment["start"])}-{format_time(segment["end"])}] '
            f'{segment["text"]}'
        )

    transcript = "\n".join(lines)

    if len(transcript) <= max_chars:
        return transcript

    half = max_chars // 2

    return (
        transcript[:half].rstrip()
        + "\n\n[...middle omitted for prompt length...]\n\n"
        + transcript[-half:].lstrip()
    )


def get_segment_range_times(
    segments: List[Dict[str, Any]],
    start_segment: int,
    end_segment: int,
) -> Tuple[float, float, float]:
    by_index = {int(s["index"]): s for s in segments}

    if start_segment not in by_index:
        raise ValueError(f"Invalid start_segment: {start_segment}")

    if end_segment not in by_index:
        raise ValueError(f"Invalid end_segment: {end_segment}")

    if end_segment < start_segment:
        start_segment, end_segment = end_segment, start_segment

    start = float(by_index[start_segment]["start"])
    end = float(by_index[end_segment]["end"])

    return start, end, end - start


def clip_text_from_range(
    segments: List[Dict[str, Any]],
    start_segment: int,
    end_segment: int,
) -> str:
    if end_segment < start_segment:
        start_segment, end_segment = end_segment, start_segment

    return clean_text(
        " ".join(
            segment["text"]
            for segment in segments
            if start_segment <= int(segment["index"]) <= end_segment
        )
    )


SHORTS_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "name": "shorts_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall_notes": {
                "type": "string",
                "description": "Short editor notes about the sermon and the quality of the Shorts found.",
            },
            "clips": {
                "type": "array",
                "description": "Usable Shorts found in the sermon.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_segment": {
                            "type": "integer",
                            "description": "First segment number included in the clip.",
                        },
                        "end_segment": {
                            "type": "integer",
                            "description": "Last segment number included in the clip.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Original editorial YouTube Shorts title. Not just the first transcript words.",
                        },
                        "strength_score": {
                            "type": "integer",
                            "description": "Strength from 1 to 10.",
                        },
                        "clip_type": {
                            "type": "string",
                            "enum": [
                                "story",
                                "teaching",
                                "warning",
                                "exhortation",
                                "reflection",
                            ],
                        },
                        "why_it_works": {
                            "type": "string",
                            "description": "Why the clip works as a standalone short.",
                        },
                        "power_quote": {
                            "type": "string",
                            "description": "Strongest exact phrase or line from the clip.",
                        },
                        "image_idea": {
                            "type": "string",
                            "description": "Simple visual idea for public-domain art search or AI image generation.",
                        },
                        "image_search_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search terms for public-domain artwork.",
                        },
                    },
                    "required": [
                        "start_segment",
                        "end_segment",
                        "title",
                        "strength_score",
                        "clip_type",
                        "why_it_works",
                        "power_quote",
                        "image_idea",
                        "image_search_terms",
                    ],
                },
            },
        },
        "required": ["overall_notes", "clips"],
    },
}


def build_analysis_prompt(
    segments: List[Dict[str, Any]],
    min_clips: int,
    max_clips: int,
    retry_mode: bool = False,
) -> str:
    transcript = build_timed_transcript(segments)

    retry_text = ""

    if retry_mode:
        retry_text = """
This is a retry because the previous attempt produced no usable normalized clips.

Be more practical:
- Do not return an empty clips list unless the transcript is only announcements or unusable audio.
- Use the segment timestamps carefully.
- Make each clip between 30 and 90 seconds.
- Prefer complete ranges that begin and end on natural thoughts.
- It is acceptable to include clips that are 6/10 if they stand alone.
"""

    return f"""
You are a careful YouTube Shorts editor.

Analyze this Catholic homily transcript from beginning to end and identify all usable independent 30 to 90 second Shorts.

{retry_text}

Important:
Do NOT only choose the top few best clips.
Do NOT stop after finding 2 or 3 strong clips.
Scan the entire homily sequentially and identify every section that can stand alone as a short, impactful video.

The output will be displayed in this table:

ID | Time | Length | Title | Strength | Image Idea

Your job:
- Identify all usable Shorts in the homily, up to {max_clips}.
- A usable Short is a 30 to 90 second section that has one complete idea.
- The clip must be understandable without the rest of the sermon.
- It should have a clear beginning, middle, and payoff.
- Include strong clips even if they are not the absolute best.
- Do not return filler.
- Do not return clips below 6/10.
- Do not overlap clips unless absolutely necessary.
- Do not split one story into several clips if the full story fits under 90 seconds.

Expected behavior:
- If the homily contains 2 usable clips, return 2.
- If the homily contains 8 usable clips, return 8.
- If the homily contains 14 usable clips, return 14.
- The number should come from the transcript, not from an artificial target.

Strength score:
10 = excellent, should definitely produce
8-9 = strong
6-7 = usable but optional
1-5 = do not include

Clip selection method:
1. Read the entire transcript.
2. Break it into major movements or complete thoughts.
3. For each movement, decide whether a 30 to 90 second standalone clip exists.
4. If yes, include it.
5. Make sure the final list covers the whole sermon, not just one section.

Before returning each clip:
- Verify that start_segment and end_segment produce a clip between 30 and 90 seconds.
- Do not guess segment numbers.
- Use only segment numbers that appear in the transcript.
- Prefer non-overlapping sequential clips.

Title rules:
- Titles must be original editorial titles.
- Do not use the first words of the transcript as the title.
- Titles should be short, strong, and YouTube-friendly.

Image rules:
- Image ideas should be simple and visual.
- Prefer Renaissance, medieval, Baroque, or traditional sacred art ideas when possible.
- image_search_terms should help find public-domain artwork from:
  - The Met
  - National Gallery of Art
  - Art Institute of Chicago
  - Rijksmuseum

Return only schema-valid JSON.

Timed transcript:
{transcript}
""".strip()


def call_openai_for_shorts(
    segments: List[Dict[str, Any]],
    min_clips: int = DEFAULT_MIN_CLIPS,
    max_clips: int = DEFAULT_MAX_CLIPS,
    model: Optional[str] = None,
    retry_mode: bool = False,
) -> Dict[str, Any]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_KEY in environment.")

    model = model or os.getenv("OPENAI_SHORTS_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0.15,
        max_completion_tokens=7000,
        messages=[
            {
                "role": "system",
                "content": "You are a careful YouTube Shorts editor. Return only schema-valid JSON.",
            },
            {
                "role": "user",
                "content": build_analysis_prompt(
                    segments=segments,
                    min_clips=min_clips,
                    max_clips=max_clips,
                    retry_mode=retry_mode,
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SHORTS_ANALYSIS_SCHEMA,
        },
    )

    raw = response.choices[0].message.content or "{}"

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {raw[:500]}") from exc


def normalize_ai_results(
    ai_payload: Dict[str, Any],
    segments: List[Dict[str, Any]],
    max_clips: int = DEFAULT_MAX_CLIPS,
) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    used_ranges: List[Tuple[float, float]] = []
    rejected: List[Dict[str, Any]] = []

    raw_clips = ai_payload.get("clips", [])

    print()
    print(f"AI returned {len(raw_clips)} raw clip candidate(s).")
    print()

    for raw in raw_clips:
        if len(normalized) >= max_clips:
            break

        try:
            start_segment = int(raw["start_segment"])
            end_segment = int(raw["end_segment"])
            start, end, duration = get_segment_range_times(
                segments,
                start_segment,
                end_segment,
            )
        except Exception as exc:
            rejected.append(
                {
                    "title": raw.get("title", "Untitled"),
                    "reason": f"Invalid segment range: {exc}",
                    "raw": raw,
                }
            )
            continue

        if duration < MIN_SHORT_SECONDS:
            rejected.append(
                {
                    "title": raw.get("title", "Untitled"),
                    "time": f"{format_time(start)}-{format_time(end)}",
                    "duration": round(duration, 2),
                    "reason": f"Too short. Minimum is {MIN_SHORT_SECONDS}s.",
                    "raw": raw,
                }
            )
            continue

        if duration > MAX_SHORT_SECONDS:
            rejected.append(
                {
                    "title": raw.get("title", "Untitled"),
                    "time": f"{format_time(start)}-{format_time(end)}",
                    "duration": round(duration, 2),
                    "reason": f"Too long. Maximum is {MAX_SHORT_SECONDS}s.",
                    "raw": raw,
                }
            )
            continue

        overlaps = False

        for used_start, used_end in used_ranges:
            overlap = max(0.0, min(end, used_end) - max(start, used_start))

            if overlap > 2.0:
                overlaps = True
                rejected.append(
                    {
                        "title": raw.get("title", "Untitled"),
                        "time": f"{format_time(start)}-{format_time(end)}",
                        "duration": round(duration, 2),
                        "reason": f"Overlaps existing clip by {round(overlap, 2)}s.",
                        "raw": raw,
                    }
                )
                break

        if overlaps:
            continue

        title = clean_text(raw.get("title", "Untitled Short"))
        strength_score = int(raw.get("strength_score", 6))
        source_text = clip_text_from_range(segments, start_segment, end_segment)

        normalized.append(
            {
                "id": len(normalized) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "time": f"{format_time(start)}-{format_time(end)}",
                "length": f"{int(round(duration))}s",
                "length_seconds": round(duration, 3),
                "title": title,
                "strength": f"{strength_score}/10",
                "strength_score": strength_score,
                "clip_type": raw.get("clip_type", "teaching"),
                "why_it_works": clean_text(raw.get("why_it_works", "")),
                "power_quote": clean_text(raw.get("power_quote", "")),
                "image_idea": clean_text(raw.get("image_idea", "")),
                "image_search_terms": raw.get("image_search_terms", []),
                "selected": False,
                "start_segment": start_segment,
                "end_segment": end_segment,
                "source_text": source_text,
            }
        )

        used_ranges.append((start, end))

    if rejected:
        print("Rejected AI clip candidate(s):")
        print("-" * 100)

        for item in rejected:
            print(
                f"{item.get('title', 'Untitled')} | "
                f"{item.get('time', 'no time')} | "
                f"{item.get('duration', 'no duration')}s | "
                f"{item.get('reason')}"
            )

        print("-" * 100)
        print()

    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_notes": clean_text(ai_payload.get("overall_notes", "")),
        "clips": normalized,
        "debug": {
            "raw_clip_count": len(raw_clips),
            "accepted_clip_count": len(normalized),
            "rejected_clip_count": len(rejected),
            "rejected": rejected,
        },
    }


def print_shorts_table(analysis: Dict[str, Any]) -> None:
    clips = analysis.get("clips", [])

    if not clips:
        print("No usable Shorts found.")
        debug = analysis.get("debug", {})
        if debug:
            print(
                f"Debug: raw={debug.get('raw_clip_count', 0)}, "
                f"accepted={debug.get('accepted_clip_count', 0)}, "
                f"rejected={debug.get('rejected_clip_count', 0)}"
            )
        return

    print()
    print("Usable Shorts Identified")
    print("-" * 100)
    print(f"{'ID':<4} {'Time':<13} {'Length':<8} {'Strength':<10} {'Title':<36} Image Idea")
    print("-" * 100)

    for clip in clips:
        print(
            f"{clip['id']:<4} "
            f"{clip['time']:<13} "
            f"{clip['length']:<8} "
            f"{clip['strength']:<10} "
            f"{clip['title'][:35]:<36} "
            f"{clip['image_idea'][:60]}"
        )

    print("-" * 100)
    print()
    print("Selection example for Step #3:")
    print("1, 2-5, 7, 9-12")
    print()


def identify_usable_shorts_from_folder(
    homily_folder: str | Path,
    min_clips: int = DEFAULT_MIN_CLIPS,
    max_clips: int = DEFAULT_MAX_CLIPS,
    model: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    homily_json_path = find_homily_json(homily_folder)
    output_folder = get_output_folder(homily_json_path)
    output_path = output_folder / "shorts_analysis.json"
    raw_output_path = output_folder / "shorts_analysis_raw_ai.json"

    if output_path.exists() and not force:
        analysis = load_json(output_path)
        print(f"Using existing analysis: {output_path}")
        print_shorts_table(analysis)
        return analysis

    homily_data = load_json(homily_json_path)
    segments = normalize_segments(homily_data)

    ai_payload = call_openai_for_shorts(
        segments=segments,
        min_clips=min_clips,
        max_clips=max_clips,
        model=model,
        retry_mode=False,
    )

    save_json(raw_output_path, ai_payload)
    print(f"Saved raw AI response: {raw_output_path}")

    analysis = normalize_ai_results(
        ai_payload=ai_payload,
        segments=segments,
        max_clips=max_clips,
    )

    if len(analysis.get("clips", [])) == 0:
        print("No normalized clips accepted. Retrying once with a stricter practical prompt...")

        retry_payload = call_openai_for_shorts(
            segments=segments,
            min_clips=min_clips,
            max_clips=max_clips,
            model=model,
            retry_mode=True,
        )

        retry_raw_output_path = output_folder / "shorts_analysis_raw_ai_retry.json"
        save_json(retry_raw_output_path, retry_payload)
        print(f"Saved retry raw AI response: {retry_raw_output_path}")

        retry_analysis = normalize_ai_results(
            ai_payload=retry_payload,
            segments=segments,
            max_clips=max_clips,
        )

        if len(retry_analysis.get("clips", [])) > 0:
            analysis = retry_analysis

    save_json(output_path, analysis)

    print(f"Saved Shorts analysis: {output_path}")
    print_shorts_table(analysis)

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step #2: Identify usable Shorts from a homily.")
    parser.add_argument("homily_folder", help="Homily folder or direct JSON path.")
    parser.add_argument("--min-clips", type=int, default=DEFAULT_MIN_CLIPS)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate shorts_analysis.json.")

    args = parser.parse_args()

    identify_usable_shorts_from_folder(
        homily_folder=args.homily_folder,
        min_clips=args.min_clips,
        max_clips=args.max_clips,
        model=args.model,
        force=args.force,
    )