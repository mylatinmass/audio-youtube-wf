import csv
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shorts_fxn.step_02_identify_shorts import (
    clean_text,
    find_homily_json,
    format_time,
    get_output_folder,
    load_json,
    normalize_segments,
    print_shorts_table,
    save_json,
)


def parse_time_to_seconds(value: str) -> float:
    value = clean_text(value).strip()

    if not value:
        raise ValueError("Missing time value.")

    parts = value.split(":")

    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Invalid time value: {value}")


def clean_manual_title(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^\*\*(.*?)\*\*$", r"\1", value)
    value = re.sub(r"^['\"]|['\"]$", "", value)
    return clean_text(value)


def parse_markdown_table_row(line: str) -> Optional[Tuple[int, float, float, str]]:
    line = line.strip()

    if not line.startswith("|") or not line.endswith("|"):
        return None

    cells = [cell.strip() for cell in line.strip("|").split("|")]

    if len(cells) < 4:
        return None

    if re.fullmatch(r":?-+:?", cells[0]) or cells[0].lower() in {"#", "id"}:
        return None

    if not re.search(r"\d", cells[0]):
        return None

    try:
        clip_id = int(re.sub(r"\D+", "", cells[0]))
        start = parse_time_to_seconds(cells[1])
        end = parse_time_to_seconds(cells[2])
    except Exception:
        return None

    title = clean_manual_title(" | ".join(cells[3:]))

    if not title:
        return None

    return clip_id, start, end, title


def parse_plain_row(line: str) -> Optional[Tuple[int, float, float, str]]:
    match = re.match(
        r"^\s*(\d+)[\).\s|-]+(\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?)\s+"
        r"(\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?)\s+(.+?)\s*$",
        line,
    )

    if not match:
        return None

    return (
        int(match.group(1)),
        parse_time_to_seconds(match.group(2)),
        parse_time_to_seconds(match.group(3)),
        clean_manual_title(match.group(4)),
    )


def normalize_csv_header(value: str) -> str:
    value = str(value or "").strip().lower().lstrip("\ufeff")
    return re.sub(r"[^a-z0-9#]+", "", value)


def csv_value(row: Dict[str, str], *names: str) -> str:
    normalized = {
        normalize_csv_header(key): value
        for key, value in row.items()
    }

    for name in names:
        key = normalize_csv_header(name)

        if key in normalized:
            return str(normalized[key] or "").strip()

    return ""


def parse_csv_clips(text: str) -> List[Dict[str, Any]]:
    sample = str(text or "").lstrip()

    if "," not in sample:
        return []

    reader = csv.DictReader(io.StringIO(sample))

    if not reader.fieldnames:
        return []

    headers = {normalize_csv_header(header) for header in reader.fieldnames}

    if not ({"start", "end", "title"} <= headers):
        return []

    clips: List[Dict[str, Any]] = []

    for row_number, row in enumerate(reader, start=1):
        start_raw = csv_value(row, "start")
        end_raw = csv_value(row, "end")
        title = clean_manual_title(csv_value(row, "title"))

        if not start_raw or not end_raw or not title:
            continue

        clip_id_raw = csv_value(row, "#", "id", "clip", "clip_id", "clipid")

        try:
            clip_id = int(clip_id_raw) if clip_id_raw else row_number
            start = parse_time_to_seconds(start_raw)
            end = parse_time_to_seconds(end_raw)
        except Exception:
            continue

        if end < start:
            start, end = end, start

        duration = end - start

        if duration <= 0:
            continue

        clips.append(
            {
                "id": clip_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "title": title,
            }
        )

    return clips


def parse_manual_clips_table(text: str) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = parse_csv_clips(text)

    if not clips:
        for line in str(text or "").splitlines():
            parsed = parse_markdown_table_row(line) or parse_plain_row(line)

            if not parsed:
                continue

            clip_id, start, end, title = parsed

            if end < start:
                start, end = end, start

            duration = end - start

            if duration <= 0:
                continue

            clips.append(
                {
                    "id": clip_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "title": title,
                }
            )

    clips.sort(key=lambda clip: (float(clip["start"]), int(clip["id"])))

    for index, clip in enumerate(clips, start=1):
        clip["id"] = index

    if not clips:
        raise ValueError(
            "No manual clips found. Expected CSV columns Start,End,Title "
            "or a markdown table with # | Start | End | Title."
        )

    return clips


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


def image_search_terms_for_title(title: str) -> List[str]:
    title = clean_text(title)
    terms = [
        f"{title} Catholic painting",
        f"{title} sacred art",
        f"{title} Renaissance painting",
        f"{title} Baroque painting",
    ]

    lowered = title.lower()

    if "eucharist" in lowered or "communion" in lowered or "mass" in lowered:
        terms.extend(
            [
                "Last Supper painting",
                "Mass of Saint Gregory painting",
                "Eucharist Catholic painting",
            ]
        )

    if "sacred heart" in lowered:
        terms.extend(["Sacred Heart of Jesus painting", "Christ Sacred Heart painting"])

    if "mary" in lowered or "our lady" in lowered:
        terms.extend(["Virgin Mary painting", "Immaculate Heart of Mary painting"])

    if "joseph" in lowered:
        terms.extend(["Saint Joseph painting", "Holy Family painting"])

    if "sin" in lowered or "sinner" in lowered:
        terms.extend(["Christ forgiving sinner painting", "Prodigal Son painting"])

    return list(dict.fromkeys(term for term in terms if term.strip()))[:8]


def build_manual_analysis(
    manual_clips: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = []
    warnings = []
    transcript_end = max(float(segment["end"]) for segment in segments) if segments else 0.0

    for clip in manual_clips:
        start = float(clip["start"])
        end = float(clip["end"])
        duration = end - start
        title = clean_manual_title(clip["title"])
        start_segment, end_segment = segment_bounds_for_time_range(segments, start, end)
        source_text = source_text_for_time_range(segments, start, end)
        selected = True

        if duration < 15:
            warnings.append(
                f"{title}: duration is {duration:.1f}s, below the usual 15s Shorts target. "
                "Leaving it unselected."
            )
            selected = False

        if duration > 90:
            warnings.append(
                f"{title}: duration is {duration:.1f}s, above the usual 90s Shorts target. "
                "Leaving it unselected."
            )
            selected = False

        if end > transcript_end + 0.5:
            warnings.append(
                f"{title}: ends at {format_time(end)}, beyond transcript end {format_time(transcript_end)}. "
                "Leaving it unselected."
            )
            selected = False
        elif start >= transcript_end:
            warnings.append(
                f"{title}: starts at {format_time(start)}, beyond transcript end {format_time(transcript_end)}. "
                "Leaving it unselected."
            )
            selected = False

        if not source_text:
            warnings.append(f"{title}: no transcript text overlapped {format_time(start)}-{format_time(end)}.")

        normalized.append(
            {
                "id": len(normalized) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "time": f"{format_time(start)}-{format_time(end)}",
                "length": f"{int(round(duration))}s",
                "length_seconds": round(duration, 3),
                "title": title,
                "strength": "manual",
                "strength_score": 10,
                "clip_type": "manual",
                "why_it_works": "Manually selected clip.",
                "power_quote": title,
                "image_idea": title,
                "image_search_terms": image_search_terms_for_title(title),
                "selected": selected,
                "start_segment": start_segment,
                "end_segment": end_segment,
                "source_text": source_text,
            }
        )

    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_notes": "Manual Shorts table supplied by user.",
        "manual_source": True,
        "selected_ids": [int(clip["id"]) for clip in normalized if clip.get("selected")],
        "selected_count": len([clip for clip in normalized if clip.get("selected")]),
        "clips": normalized,
        "debug": {
            "manual_clip_count": len(normalized),
            "warnings": warnings,
        },
    }


def create_shorts_analysis_from_manual_table(
    homily_folder: str | Path,
    manual_table_path: str | Path,
    force: bool = True,
) -> Dict[str, Any]:
    manual_table_path = Path(manual_table_path).expanduser().resolve()

    if not manual_table_path.exists():
        raise FileNotFoundError(f"Manual clips table not found: {manual_table_path}")

    homily_json_path = find_homily_json(homily_folder)
    output_folder = get_output_folder(homily_json_path)
    output_path = output_folder / "shorts_analysis.json"

    if output_path.exists() and not force:
        analysis = load_json(output_path)
        print(f"Using existing analysis: {output_path}")
        print_shorts_table(analysis)
        return analysis

    homily_data = load_json(homily_json_path)
    segments = normalize_segments(homily_data)
    manual_clips = parse_manual_clips_table(manual_table_path.read_text(encoding="utf-8"))
    analysis = build_manual_analysis(manual_clips, segments)

    save_json(output_path, analysis)

    print(f"Saved manual Shorts analysis: {output_path}")

    warnings = analysis.get("debug", {}).get("warnings") or []
    if warnings:
        print()
        print("Manual clip warning(s):")
        for warning in warnings:
            print(f"- {warning}")

    selected_ids = analysis.get("selected_ids", [])
    unselected = [
        int(clip["id"])
        for clip in analysis.get("clips", [])
        if not clip.get("selected")
    ]

    print()
    print(f"Manual clips selected for rendering: {len(selected_ids)} of {len(analysis.get('clips', []))}")
    if unselected:
        print(f"Unselected manual clip IDs: {', '.join(str(item) for item in unselected)}")

    print_shorts_table(analysis)
    return analysis
