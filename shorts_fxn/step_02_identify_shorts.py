import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


MIN_SHORT_SECONDS = 15
MAX_SHORT_SECONDS = 90
DEFAULT_MIN_CLIPS = 1
DEFAULT_MAX_CLIPS = 16
SHORTS_ANALYSIS_VERSION = 2
SHORTS_ANALYSIS_PROMPT_VERSION = "broad_exact_time_v5"
DEFAULT_REVIEW_TARGET_CLIPS = 12


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


def source_text_for_time_range(
    segments: List[Dict[str, Any]],
    start: float,
    end: float,
) -> str:
    overlapping = []

    for segment in segments:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])

        if segment_end <= start or segment_start >= end:
            continue

        overlapping.append(segment["text"])

    return clean_text(" ".join(overlapping))


def segment_bounds_for_time_range(
    segments: List[Dict[str, Any]],
    start: float,
    end: float,
) -> Tuple[Optional[int], Optional[int]]:
    indexes = []

    for segment in segments:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])

        if segment_end <= start or segment_start >= end:
            continue

        indexes.append(int(segment["index"]))

    if not indexes:
        return None, None

    return min(indexes), max(indexes)


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
                        "start_time": {
                            "type": "number",
                            "description": "Exact clip start time in seconds from the beginning of the homily audio.",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Exact clip end time in seconds from the beginning of the homily audio.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Original editorial YouTube Shorts title. Not just the first transcript words.",
                        },
                        "main_idea": {
                            "type": "string",
                            "description": "One or two sentence summary of the complete idea that makes this clip worth considering.",
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
                        "start_time",
                        "end_time",
                        "title",
                        "main_idea",
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
    homily_start = float(segments[0]["start"])
    homily_end = float(segments[-1]["end"])
    homily_duration = max(0.0, homily_end - homily_start)
    target_min = min(max_clips, max(DEFAULT_REVIEW_TARGET_CLIPS, min_clips))

    retry_text = ""

    if retry_mode:
        retry_text = f"""
This is a retry because the previous attempt produced too few usable normalized clips.

Be more practical:
- Do not return an empty clips list unless the transcript is only announcements or unusable audio.
- Use the segment timestamps carefully.
- Make each clip between 15 and 90 seconds.
- Prefer complete ranges that begin and end on full sentences and natural thoughts.
- It is acceptable to include clips that are 7/10 if they clearly stand alone.
- This is a review-list generation pass, not a final production decision.
- Return at least {target_min} valid candidates unless the transcript truly has fewer than {target_min} complete standalone ideas.
- A 7/10 complete idea is better than omitting a usable candidate.
- Do not stop after 5, 6, or 7 clips when there are more complete thoughts later in the homily.
- If an idea naturally runs longer than 90 seconds, choose the strongest 45 to 90 second subrange that still has a complete idea.
- Do not return any candidate longer than 90 seconds; long candidates waste the retry.
- Do not return any time later than {round(homily_end, 2)} seconds.
"""

    return f"""
You are a careful YouTube Shorts editor and homily analyst.

Analyze this Catholic homily transcript from beginning to end and identify a broad ranked list of the strongest independent 15 to 90 second Shorts.

{retry_text}

Homily timing:
- The homily starts at {round(homily_start, 2)} seconds.
- The homily ends at {round(homily_end, 2)} seconds.
- The homily duration is {format_time(homily_duration)} ({round(homily_duration, 2)} seconds).
- Every start_time and end_time must be within this range.

Important:
Return more than only the safest few.
Do not return merely "usable" filler.
Scan the entire homily sequentially, and return every section with a complete sentence-level idea that stands out.
The ideal clip feels like something a viewer could encounter without context and still understand, remember, and want to share.
When the homily has enough material, a good result is usually {target_min} to {max_clips} ranked candidates.
For ordinary 20 to 35 minute homilies, do not assume that only 3 to 5 clips exist. Build a broad review list first; the user will choose final renders later.

The output will be displayed in this table:

ID | Time | Length | Strength | Title | Main Idea

Your job:
- Identify a ranked list of the best Shorts in the homily, up to {max_clips}.
- A strong Short is a 15 to 90 second section that contains one complete idea.
- The clip must be understandable without the rest of the sermon.
- It must begin on a complete sentence or natural opening thought.
- It must end after the idea resolves, not in the middle of a sentence, setup, or transition.
- It should have a clear hook, development, and payoff.
- Include strong clips even if they are not the absolute best, but reject ordinary connective tissue.
- Do not return filler, setup-only passages, repeated points, or clips that need the previous minute to make sense.
- Do not return clips below 7/10.
- Avoid heavy overlap. Small overlap is okay when two clips have distinct ideas.
- Do not split one story into several clips if the full story fits under 90 seconds.
- Prefer 35 to 75 seconds when possible. Use shorter clips when the idea is genuinely complete. Use the full 90 seconds only when the full idea requires it.

Expected behavior:
- If the homily contains 2 strong clips, return 2.
- If the homily contains 8 strong clips, return 8.
- If the homily contains 14 strong clips, return 14.
- The number should come from the transcript, not from an artificial target.
- It is better to return 12 good-or-better candidates for review than only 3 perfect candidates.
- If you are unsure between including and omitting a complete 7/10 idea, include it.
- If one topic has a long arc, select the best shorter complete sub-idea instead of returning a clip over 90 seconds.

Strength score:
10 = excellent, should definitely produce
8-9 = strong
7 = good enough if it is complete and standalone
1-6 = do not include

Clip selection method:
1. Silently map the sermon: opening claim, scripture/doctrine, examples, warnings, applications, and final exhortation.
2. Identify candidate windows around complete thoughts, not around arbitrary transcript segment blocks.
3. For each candidate, choose exact start_time and end_time in seconds from the homily audio.
4. Adjust the start_time and end_time until the clip starts and ends cleanly.
5. Include the candidate only if it passes the quality gates below.
6. Rank the returned candidates from strongest to weakest.
7. Make sure you considered the whole sermon, not just one section.
8. Before finalizing, count your clips. If the count is below {target_min}, scan again for complete 7/10 ideas you skipped.

Quality gates for every returned clip:
- Complete sentence gate: the first included words should not feel like the continuation of an omitted sentence.
- Complete idea gate: the clip must contain a full claim, explanation, example, exhortation, or warning.
- Standalone gate: a viewer should not need prior names, pronouns, references, or setup from outside the clip.
- Payoff gate: the final included segment should complete the thought with a strong landing.
- Highlight gate: the idea should be surprising, emotionally forceful, spiritually piercing, memorable, or especially clear.

Before returning each clip:
- Verify that start_time and end_time produce a clip between 15 and 90 seconds.
- Use plain seconds from the beginning of the homily. Decimal seconds are allowed.
- Do not write timestamps as MM.SS. For 8:52, write 532, not 8.52.
- Do not return times after {round(homily_end, 2)} seconds.
- Do not use segment numbers as the start/end contract.
- Prefer non-overlapping ranked clips, but keep distinct ideas if the overlap is minor.
- Read the exact selected text one more time and reject it if it begins or ends awkwardly.

Title rules:
- Titles must be original editorial titles.
- Do not use the first words of the transcript as the title.
- Titles should be short, strong, and YouTube-friendly.

Main idea rules:
- main_idea should explain the standout idea, like an editor note.
- Do not quote the whole clip in main_idea.
- Make main_idea useful for choosing which clips to render.

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


def uncovered_ranges_for_clips(
    segments: List[Dict[str, Any]],
    clips: List[Dict[str, Any]],
    minimum_gap_seconds: float = 45.0,
) -> List[Tuple[float, float]]:
    if not segments:
        return []

    homily_start = float(segments[0]["start"])
    homily_end = float(segments[-1]["end"])
    covered = sorted(
        (
            max(homily_start, float(clip.get("start", 0))),
            min(homily_end, float(clip.get("end", 0))),
        )
        for clip in clips
        if float(clip.get("end", 0)) > float(clip.get("start", 0))
    )
    merged: List[Tuple[float, float]] = []

    for start, end in covered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue

        merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    gaps = []
    cursor = homily_start

    for start, end in merged:
        if start - cursor >= minimum_gap_seconds:
            gaps.append((cursor, start))

        cursor = max(cursor, end)

    if homily_end - cursor >= minimum_gap_seconds:
        gaps.append((cursor, homily_end))

    return gaps


def build_gap_fill_prompt(
    segments: List[Dict[str, Any]],
    accepted_clips: List[Dict[str, Any]],
    min_total_clips: int,
    max_clips: int,
) -> str:
    transcript = build_timed_transcript(segments)
    homily_end = float(segments[-1]["end"])
    needed = max(1, min(max_clips - len(accepted_clips), min_total_clips - len(accepted_clips) + 3))
    existing = "\n".join(
        (
            f"- {clip.get('time', '')} ({clip.get('length', '')}) "
            f"{clip.get('title', '')}: {clean_text(clip.get('main_idea', ''))}"
        )
        for clip in accepted_clips
    )
    gaps = uncovered_ranges_for_clips(segments, accepted_clips)
    gap_text = "\n".join(
        f"- {format_time(start)} to {format_time(end)}"
        for start, end in gaps
    ) or "- No large uncovered gaps; find distinct non-overlapping ideas around the accepted clips."

    return f"""
You are doing a final gap-fill pass for a Catholic homily Shorts workflow.

The previous pass found only {len(accepted_clips)} usable clips. The workflow needs a broader review list, ideally at least {min_total_clips} and up to {max_clips}.

Already accepted clips:
{existing}

Focus on these under-covered time ranges first:
{gap_text}

Return {needed} to {max_clips - len(accepted_clips)} additional candidates if possible.

Rules:
- Return only NEW clips that do not heavily overlap the accepted clips above.
- Each clip must be 15 to 90 seconds.
- Use exact start_time and end_time in seconds from the beginning of the homily.
- If a strong idea is longer than 90 seconds, choose the strongest complete 45 to 90 second subrange.
- Include complete 7/10 ideas. This is a review list; the user will reject weaker clips manually.
- Prioritize complete ideas in the middle of the sermon that a broad first pass may have skipped.
- Do not return filler, but do not omit good standalone thoughts merely because they are not the top 5.
- Do not return any time later than {round(homily_end, 2)} seconds.

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
        temperature=0.35,
        max_completion_tokens=11000,
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


def call_openai_for_gap_fill(
    segments: List[Dict[str, Any]],
    accepted_clips: List[Dict[str, Any]],
    min_total_clips: int,
    max_clips: int = DEFAULT_MAX_CLIPS,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_KEY in environment.")

    model = model or os.getenv("OPENAI_SHORTS_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0.45,
        max_completion_tokens=9000,
        messages=[
            {
                "role": "system",
                "content": "You are a careful YouTube Shorts editor. Return only schema-valid JSON.",
            },
            {
                "role": "user",
                "content": build_gap_fill_prompt(
                    segments=segments,
                    accepted_clips=accepted_clips,
                    min_total_clips=min_total_clips,
                    max_clips=max_clips,
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


def normalized_clips_to_raw_payload(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_clips = []

    for clip in analysis.get("clips", []):
        raw_clips.append(
            {
                "start_time": float(clip.get("start", 0)),
                "end_time": float(clip.get("end", 0)),
                "title": clip.get("title", "Untitled Short"),
                "main_idea": clip.get("main_idea", ""),
                "strength_score": int(clip.get("strength_score", 7)),
                "clip_type": clip.get("clip_type", "teaching"),
                "why_it_works": clip.get("why_it_works") or clip.get("main_idea", ""),
                "power_quote": clip.get("power_quote", ""),
                "image_idea": clip.get("image_idea", ""),
                "image_search_terms": clip.get("image_search_terms", []),
            }
        )

    return raw_clips


def normalize_ai_results(
    ai_payload: Dict[str, Any],
    segments: List[Dict[str, Any]],
    max_clips: int = DEFAULT_MAX_CLIPS,
) -> Dict[str, Any]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    raw_clips = ai_payload.get("clips", [])

    print()
    print(f"AI returned {len(raw_clips)} raw clip candidate(s).")
    print()

    def heavy_overlap(first: Dict[str, Any], second: Dict[str, Any]) -> Tuple[bool, float]:
        overlap = max(
            0.0,
            min(float(first["end"]), float(second["end"]))
            - max(float(first["start"]), float(second["start"])),
        )

        if overlap <= 0:
            return False, 0.0

        first_duration = float(first["end"]) - float(first["start"])
        second_duration = float(second["end"]) - float(second["start"])
        smaller_duration = max(0.01, min(first_duration, second_duration))

        return overlap >= 15.0 and (overlap / smaller_duration) >= 0.35, overlap

    transcript_end = max(float(segment["end"]) for segment in segments) if segments else 0.0

    def mmss_like_number_to_seconds(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if number < 0 or number >= 60:
            return None

        minutes = int(number)
        seconds = int(round((number - minutes) * 100))

        if seconds >= 60:
            return None

        return float(minutes * 60 + seconds)

    def time_range_from_raw(raw: Dict[str, Any]) -> Tuple[float, float, float]:
        if "start_time" in raw and "end_time" in raw:
            start = float(raw["start_time"])
            end = float(raw["end_time"])

            if end < start:
                start, end = end, start

            duration = end - start

            if duration < MIN_SHORT_SECONDS or duration > MAX_SHORT_SECONDS or end > transcript_end + 0.5:
                mmss_start = mmss_like_number_to_seconds(raw["start_time"])
                mmss_end = mmss_like_number_to_seconds(raw["end_time"])

                if mmss_start is not None and mmss_end is not None:
                    if mmss_end < mmss_start:
                        mmss_start, mmss_end = mmss_end, mmss_start

                    mmss_duration = mmss_end - mmss_start

                    if (
                        MIN_SHORT_SECONDS <= mmss_duration <= MAX_SHORT_SECONDS
                        and mmss_end <= transcript_end + 0.5
                    ):
                        return mmss_start, mmss_end, mmss_duration
        else:
            start_segment = int(raw["start_segment"])
            end_segment = int(raw["end_segment"])
            start, end, _duration = get_segment_range_times(
                segments,
                start_segment,
                end_segment,
            )

        if end < start:
            start, end = end, start

        return start, end, end - start

    for raw_order, raw in enumerate(raw_clips, start=1):
        if len(accepted) >= max_clips:
            break

        try:
            start, end, duration = time_range_from_raw(raw)
        except Exception as exc:
            rejected.append(
                {
                    "title": raw.get("title", "Untitled"),
                    "reason": f"Invalid time range: {exc}",
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

        title = clean_text(raw.get("title", "Untitled Short"))

        try:
            strength_score = int(raw.get("strength_score", 6))
        except (TypeError, ValueError):
            strength_score = 6

        if strength_score < 7:
            rejected.append(
                {
                    "title": title,
                    "time": f"{format_time(start)}-{format_time(end)}",
                    "duration": round(duration, 2),
                    "reason": f"Strength below 7/10: {strength_score}/10.",
                    "raw": raw,
                }
            )
            continue

        start_segment, end_segment = segment_bounds_for_time_range(segments, start, end)
        source_text = source_text_for_time_range(segments, start, end)

        if not source_text:
            rejected.append(
                {
                    "title": title,
                    "time": f"{format_time(start)}-{format_time(end)}",
                    "duration": round(duration, 2),
                    "reason": "No transcript text overlaps this time range.",
                    "raw": raw,
                }
            )
            continue

        main_idea = clean_text(raw.get("main_idea", ""))
        why_it_works = clean_text(raw.get("why_it_works", ""))

        candidate = {
            "_raw_order": raw_order,
            "start": round(start, 3),
            "end": round(end, 3),
            "time": f"{format_time(start)}-{format_time(end)}",
            "length": f"{int(round(duration))}s",
            "length_seconds": round(duration, 3),
            "title": title,
            "main_idea": main_idea,
            "strength": f"{strength_score}/10",
            "strength_score": strength_score,
            "clip_type": raw.get("clip_type", "teaching"),
            "why_it_works": why_it_works or main_idea,
            "power_quote": clean_text(raw.get("power_quote", "")),
            "image_idea": clean_text(raw.get("image_idea", "")),
            "image_search_terms": raw.get("image_search_terms", []),
            "selected": False,
            "start_segment": start_segment,
            "end_segment": end_segment,
            "source_text": source_text,
        }

        replaced_existing = False
        rejected_for_overlap = False

        for existing in list(accepted):
            is_heavy, overlap = heavy_overlap(candidate, existing)

            if not is_heavy:
                continue

            if strength_score > int(existing.get("strength_score", 0)) + 1:
                accepted.remove(existing)
                replaced_existing = True
                rejected.append(
                    {
                        "title": existing.get("title", "Untitled"),
                        "time": existing.get("time", "no time"),
                        "duration": existing.get("length_seconds", "no duration"),
                        "reason": (
                            f"Replaced by stronger overlapping candidate "
                            f"'{title}' ({strength_score}/10)."
                        ),
                        "raw": {
                            "replaced_by": raw,
                            "original_raw_order": existing.get("_raw_order"),
                        },
                    }
                )
                continue

            rejected_for_overlap = True
            rejected.append(
                {
                    "title": title,
                    "time": f"{format_time(start)}-{format_time(end)}",
                    "duration": round(duration, 2),
                    "reason": (
                        f"Heavy overlap with stronger-ranked clip "
                        f"'{existing.get('title')}' by {round(overlap, 2)}s."
                    ),
                    "raw": raw,
                }
            )
            break

        if rejected_for_overlap:
            continue

        accepted.append(candidate)

        if replaced_existing:
            accepted.sort(key=lambda clip: int(clip.get("_raw_order", 0)))

    normalized = []

    for clip in sorted(accepted, key=lambda item: int(item.get("_raw_order", 0)))[:max_clips]:
        clip = dict(clip)
        clip.pop("_raw_order", None)
        clip["id"] = len(normalized) + 1
        normalized.append(clip)

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
        "version": SHORTS_ANALYSIS_VERSION,
        "prompt_version": SHORTS_ANALYSIS_PROMPT_VERSION,
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
    print("-" * 140)
    print(f"{'ID':<4} {'Time':<13} {'Length':<8} {'Strength':<10} {'Title':<34} Main Idea")
    print("-" * 140)

    for clip in clips:
        main_idea = clean_text(clip.get("main_idea") or clip.get("why_it_works") or clip.get("image_idea", ""))
        print(
            f"{clip['id']:<4} "
            f"{clip['time']:<13} "
            f"{clip['length']:<8} "
            f"{clip['strength']:<10} "
            f"{clip['title'][:33]:<34} "
            f"{main_idea[:72]}"
        )

    print("-" * 140)
    print()
    print("Selection example for Step #3:")
    print("1, 2-5, 7, 9-12")
    print()


def existing_analysis_is_current(
    analysis: Dict[str, Any],
    min_clips: int,
    max_clips: int,
) -> Tuple[bool, str]:
    if analysis.get("prompt_version") == SHORTS_ANALYSIS_PROMPT_VERSION:
        return True, "current prompt version"

    clips = analysis.get("clips", [])
    target_min_candidates = min(max_clips, max(DEFAULT_REVIEW_TARGET_CLIPS, min_clips))

    if len(clips) < target_min_candidates:
        return (
            False,
            (
                f"old analysis has {len(clips)} clip(s), below the new "
                f"{target_min_candidates}-candidate review target"
            ),
        )

    if any("main_idea" not in clip for clip in clips):
        return False, "old analysis is missing Main Idea fields"

    return True, "old analysis already has a broad candidate list"


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
        is_current, reason = existing_analysis_is_current(
            analysis=analysis,
            min_clips=min_clips,
            max_clips=max_clips,
        )

        if is_current:
            print(f"Using existing analysis: {output_path} ({reason})")
            print_shorts_table(analysis)
            return analysis

        print(f"Regenerating Shorts analysis: {reason}.")

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

    target_min_candidates = min(max_clips, max(DEFAULT_REVIEW_TARGET_CLIPS, min_clips))

    if len(analysis.get("clips", [])) < target_min_candidates:
        print(
            f"Only {len(analysis.get('clips', []))} normalized clip(s) accepted. "
            "Retrying once with a stricter practical prompt..."
        )

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

        if len(retry_analysis.get("clips", [])) > len(analysis.get("clips", [])):
            analysis = retry_analysis

    if len(analysis.get("clips", [])) < target_min_candidates:
        print(
            f"Only {len(analysis.get('clips', []))} clip(s) after retry. "
            "Running one gap-fill pass for under-covered sermon sections..."
        )

        gap_payload = call_openai_for_gap_fill(
            segments=segments,
            accepted_clips=analysis.get("clips", []),
            min_total_clips=target_min_candidates,
            max_clips=max_clips,
            model=model,
        )

        gap_raw_output_path = output_folder / "shorts_analysis_raw_ai_gap_fill.json"
        save_json(gap_raw_output_path, gap_payload)
        print(f"Saved gap-fill raw AI response: {gap_raw_output_path}")

        combined_payload = {
            "overall_notes": clean_text(
                " ".join(
                    [
                        analysis.get("overall_notes", ""),
                        gap_payload.get("overall_notes", ""),
                    ]
                )
            ),
            "clips": normalized_clips_to_raw_payload(analysis) + gap_payload.get("clips", []),
        }
        combined_analysis = normalize_ai_results(
            ai_payload=combined_payload,
            segments=segments,
            max_clips=max_clips,
        )

        if len(combined_analysis.get("clips", [])) > len(analysis.get("clips", [])):
            analysis = combined_analysis

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
