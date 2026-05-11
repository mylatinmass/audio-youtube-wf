# mdx_generator.py
import datetime
import json
import os
import re
from typing import Optional, List, Dict, Any, Tuple

import openai
from text_find import find_homily  # your existing helper


# ---------------------------
# OpenAI setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_KEY is not set in the environment.")


# ---------------------------
# Helpers for date/propers
# ---------------------------
def get_previous_sunday(d: datetime.date) -> datetime.date:
    """If today is Sunday -> same day; else go back to previous Sunday."""
    return d - datetime.timedelta(days=(d.weekday() + 1) % 7)


def infer_mass_hint(homily_text: str) -> Optional[str]:
    """
    Very light heuristic to extract a Mass/feast name likely said aloud.
    Returns a normalized key you can use in the table below.
    """
    txt = (homily_text or "").lower()

    # Nth Sunday after Pentecost
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s+sunday\s+after\s+pentecost\b", txt)
    if m:
        n = int(m.group(1))
        suffix = "th" if n in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}-sunday-after-pentecost"

    # Advent
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s+sunday\s+of\s+advent\b", txt)
    if m:
        n = int(m.group(1))
        suffix = "th" if n in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}-sunday-of-advent"

    # Lent
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s+sunday\s+in\s+lent\b", txt)
    if m:
        n = int(m.group(1))
        suffix = "th" if n in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}-sunday-in-lent"

    FEASTS = [
        ("immaculate conception", "immaculate-conception"),
        ("christ the king", "christ-the-king"),
        ("sacred heart", "sacred-heart"),
        ("epiphany", "epiphany"),
        ("ascension", "ascension"),
        ("corpus christi", "corpus-christi"),
        ("annunciation", "annunciation"),
        ("assumption", "assumption"),
        ("purification", "purification"),
        ("presentation", "presentation"),
        ("nativity of our lord", "christmas-day"),
        ("christmas", "christmas-day"),
        ("all saints", "all-saints"),
    ]
    for needle, key in FEASTS:
        if needle in txt:
            return key

    return None


def get_1962_reading(previous_sunday: datetime.date, mass_hint: Optional[str]):
    """
    Return ONE liturgical text (1962 Missal) for the identified Mass, or None.
    """
    ymd = previous_sunday.strftime("%Y-%m-%d")

    by_mass = {}
    by_date = {}

    if mass_hint and mass_hint in by_mass:
        return by_mass[mass_hint]
    if ymd in by_date:
        return by_date[ymd]
    return None


# ---------------------------
# Segment normalizer
# ---------------------------
def _normalize_segments(segments: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    Ensure segments are a list of dicts with keys: start, end, text.
    Accepts dicts or tuples/lists.
    """
    normalized: List[Dict[str, Any]] = []

    for s in segments or []:
        if isinstance(s, dict):
            normalized.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "") or "").strip(),
            })
            continue

        if isinstance(s, (list, tuple)):
            start = 0.0
            end = 0.0
            text = ""

            if len(s) >= 3:
                if isinstance(s[0], (int, float)) and isinstance(s[1], (int, float)):
                    start = s[0]
                    end = s[1]
                    text = s[2]
                elif len(s) >= 4 and isinstance(s[1], (int, float)) and isinstance(s[2], (int, float)):
                    start = s[1]
                    end = s[2]
                    text = s[3]
                else:
                    start, end, text = s[0], s[1], s[2]

            normalized.append({
                "start": float(start),
                "end": float(end),
                "text": str(text or "").strip(),
            })

    return normalized


# ---------------------------
# NEW: Title keywords -> inject into description
# ---------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by",
    "as", "at", "into", "about", "is", "are", "was", "were", "be", "been", "being",
}

def _title_keywords(title: str) -> List[str]:
    # Keep meaningful words; preserve capitalization nicely
    raw_words = re.findall(r"[A-Za-z0-9']+", (title or ""))
    words = []
    for w in raw_words:
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        if len(lw) < 4:
            continue
        words.append(w.strip("'"))
    # de-dupe while preserving order
    out = []
    seen = set()
    for w in words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def inject_title_keywords_into_youtube_description(youtube_description: str, title: str) -> str:
    """
    Looks for the subscribe keyword line and appends title-derived keywords to it.
    Keeps existing keywords intact; avoids duplicates.
    """
    if not youtube_description:
        return youtube_description

    title_kws = _title_keywords(title)
    if not title_kws:
        return youtube_description

    # Normalize line endings
    desc = youtube_description.replace("\r\n", "\n").replace("\r", "\n")

    marker = "Subscribe to this channel for videos about:"
    idx = desc.find(marker)
    if idx == -1:
        # If marker is missing, just append a keywords line at end
        extra = ", ".join(title_kws)
        return (desc.rstrip() + "\n\nKeywords: " + extra).strip()

    # Expect the next line to contain the comma-separated keyword list
    lines = desc.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == marker:
            # Next non-empty line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                # no keyword line exists; add one
                lines.append(", ".join(title_kws))
                return "\n".join(lines).strip()

            existing_line = lines[j].strip()
            # Parse existing keywords
            existing = [x.strip() for x in existing_line.split(",") if x.strip()]
            seen = {x.lower() for x in existing}
            for kw in title_kws:
                if kw.lower() not in seen:
                    existing.append(kw)
                    seen.add(kw.lower())
            lines[j] = ", ".join(existing)
            return "\n".join(lines).strip()

    return desc.strip()


# ---------------------------
# NEW: YAML writers that preserve newlines
# ---------------------------
def _yaml_escape_inline(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def _yaml_block_scalar(key: str, value: str, indent: int = 0) -> List[str]:
    """
    Write YAML block scalar:
      key: |-
        line1
        line2
    """
    pad = " " * indent
    out = [f"{pad}{key}: |-"]
    # Ensure we always have a string and preserve empty lines
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    for line in value.split("\n"):
        out.append(f"{pad}  {line}")
    return out


def _slugify(s: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return slug or "section"


def _format_time(seconds: Any) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _duration_from_segments(segments: List[Dict[str, Any]], homily_text: str) -> float:
    if segments:
        return max(float(s.get("end", 0.0) or 0.0) for s in segments)

    # Conservative spoken-word estimate for prompt guidance only.
    word_count = len(re.findall(r"\S+", homily_text or ""))
    return (word_count / 135.0) * 60.0 if word_count else 0.0


def _chapter_guidance(duration: float) -> str:
    if duration <= 0:
        return (
            "Use enough chapters to guide the listener through the argument. "
            "Prefer fewer strong, specific chapters over many generic labels."
        )

    minutes = duration / 60.0
    target = max(3, min(14, round(minutes / 2.5)))
    low = max(3, target - 2)
    high = min(16, target + 3)
    return (
        f"The homily appears to be about {minutes:.1f} minutes. A strong YouTube chapter plan "
        f"will usually land around {target} chapters, with {low}-{high} acceptable when the logic "
        "of the sermon calls for it. Do not place chapters by clock interval; place them where "
        "the preacher begins a new claim, image, example, doctrine, warning, exhortation, or conclusion."
    )


def _build_timed_transcript(segments: List[Dict[str, Any]], homily_text: str) -> str:
    """
    Give the model a transcript with timestamps it can actually reason over.
    The blocks are only for readability/context; chapter choices should be semantic.
    """
    if not segments:
        return "No timestamped segments were provided.\n\n" + (homily_text or "").strip()

    blocks: List[str] = []
    cur_text: List[str] = []
    cur_start: Optional[float] = None
    cur_end: float = 0.0
    cur_chars = 0

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end, cur_chars
        if cur_start is None or not cur_text:
            return
        idx = len(blocks) + 1
        text = " ".join(t.strip() for t in cur_text if t.strip())
        blocks.append(f"{idx:03d} [{_format_time(cur_start)}-{_format_time(cur_end)}] {text}")
        cur_text = []
        cur_start = None
        cur_end = 0.0
        cur_chars = 0

    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if cur_start is None:
            cur_start = start

        cur_text.append(text)
        cur_end = end
        cur_chars += len(text)

        if cur_chars >= 700 and re.search(r"[.!?][\"')\]]?$", text):
            flush()
        elif cur_chars >= 1100:
            flush()

    flush()

    transcript = "\n".join(blocks)
    # Keep prompts within a practical size while preserving the beginning and end.
    max_chars = 65000
    if len(transcript) <= max_chars:
        return transcript

    head = transcript[: max_chars // 2]
    tail = transcript[-max_chars // 2 :]
    return head.rstrip() + "\n\n[...middle of transcript omitted for prompt length...]\n\n" + tail.lstrip()


def _snap_to_segment_start(value: Any, segments: List[Dict[str, Any]], max_delta: float = 18.0) -> Optional[float]:
    if value is None:
        return None
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return None
    if not segments:
        return seconds

    starts = [float(s.get("start", 0.0) or 0.0) for s in segments]
    nearest = min(starts, key=lambda x: abs(x - seconds))
    if abs(nearest - seconds) <= max_delta:
        return nearest
    return seconds


def _unique_anchor(title: str, used: set) -> str:
    base = _slugify(title)
    anchor = base
    i = 2
    while anchor in used:
        anchor = f"{base}-{i}"
        i += 1
    used.add(anchor)
    return anchor


def _normalize_chapters(
    chapters: List[Any],
    segments: List[Dict[str, Any]],
    duration: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_titles = set()
    used_anchors = set()

    for ch in chapters or []:
        if not isinstance(ch, dict):
            continue
        title = re.sub(r"\s+", " ", str(ch.get("title") or "").strip())
        if not title:
            continue
        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        start = _snap_to_segment_start(ch.get("start"), segments)
        if start is not None and duration > 0:
            start = max(0.0, min(float(start), duration))

        anchor = str(ch.get("anchor") or "").strip() or title
        out.append({
            "title": title[:90],
            "anchor": _unique_anchor(anchor, used_anchors),
            "start": start,
        })

    out.sort(key=lambda x: float(x["start"]) if x["start"] is not None else 999999.0)

    # The first homily chapter should identify the opening movement, not disappear.
    if out and out[0]["start"] is not None and float(out[0]["start"]) > 20.0:
        out[0]["start"] = 0.0

    return out


def _fallback_chapters_from_headings(
    headings: List[Any],
    toc: List[Any],
    segments: List[Dict[str, Any]],
    duration: float,
) -> List[Dict[str, Any]]:
    heading_dicts = [h for h in headings or [] if isinstance(h, dict)]
    if heading_dicts:
        return _normalize_chapters(
            [
                {
                    "title": h.get("title"),
                    "anchor": h.get("title"),
                    "start": h.get("start"),
                }
                for h in heading_dicts
            ],
            segments,
            duration,
        )

    titles = [str(t or "").strip() for t in toc or [] if str(t or "").strip()]
    if not titles:
        return []

    if not segments:
        starts = [None for _ in titles]
    else:
        step = max(1, len(segments) // max(1, len(titles)))
        starts = [segments[min(i * step, len(segments) - 1)]["start"] for i in range(len(titles))]

    return _normalize_chapters(
        [{"title": title, "anchor": title, "start": start} for title, start in zip(titles, starts)],
        segments,
        duration,
    )


def _normalize_headings(
    headings: List[Any],
    chapters: List[Dict[str, Any]],
    toc: List[Any],
    segments: List[Dict[str, Any]],
    para_count: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    used = set()

    for ch in chapters or []:
        title = re.sub(r"\s+", " ", str(ch.get("title") or "").strip())
        if not title:
            continue
        start = _snap_to_segment_start(ch.get("start"), segments)
        key = (round(float(start or 0.0), 1), title.lower())
        if key in used:
            continue
        used.add(key)
        normalized.append({
            "title": title[:90],
            "level": "h2",
            "start": start,
            "para_index": None,
        })

    for h in headings or []:
        if not isinstance(h, dict):
            continue
        title = re.sub(r"\s+", " ", str(h.get("title") or "").strip())
        if not title:
            continue
        lvl = str(h.get("level") or "h2").lower()
        if lvl not in ("h2", "h3"):
            lvl = "h2"

        start = _snap_to_segment_start(h.get("start"), segments)
        para_index = None
        if start is None:
            try:
                para_index = int(h.get("para_index", 0))
            except (TypeError, ValueError):
                para_index = 0
            para_index = max(0, min(para_index, max(0, para_count - 1)))

        key = (
            round(float(start or 0.0), 1) if start is not None else None,
            title.lower(),
        )
        if key in used:
            continue
        if lvl == "h2" and any(title.lower() == ch.get("title", "").lower() for ch in chapters):
            continue
        used.add(key)

        normalized.append({
            "title": title[:90],
            "level": lvl,
            "start": start,
            "para_index": para_index,
        })

    if not normalized and toc:
        for i, t in enumerate(toc):
            title = str(t or "").strip()
            if not title:
                continue
            normalized.append({
                "title": title,
                "level": "h2",
                "start": None,
                "para_index": min(i, max(0, para_count - 1)),
            })

    normalized.sort(
        key=lambda h: (
            float(h["start"]) if h.get("start") is not None else 999999.0,
            int(h["para_index"]) if h.get("para_index") is not None else 999999,
        )
    )
    return normalized


def _paragraphize_segment_texts(segment_texts: List[str]) -> List[str]:
    paragraphs: List[str] = []
    cur: List[str] = []
    cur_chars = 0

    for text in segment_texts:
        text = str(text or "").strip()
        if not text:
            continue
        cur.append(text)
        cur_chars += len(text)
        if cur_chars >= 650 and re.search(r"[.!?][\"')\]]?$", text):
            paragraphs.append(" ".join(cur).strip())
            cur = []
            cur_chars = 0
        elif cur_chars >= 1050:
            paragraphs.append(" ".join(cur).strip())
            cur = []
            cur_chars = 0

    if cur:
        paragraphs.append(" ".join(cur).strip())
    return paragraphs


def _render_body_from_segments(
    segments: List[Dict[str, Any]],
    headings: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    starts = [float(s.get("start", 0.0) or 0.0) for s in segments]
    boundaries: List[Tuple[int, Dict[str, Any]]] = []
    seen_idx = set()

    for h in headings:
        if h.get("start") is None:
            continue
        start = float(h["start"])
        idx = min(range(len(starts)), key=lambda i: abs(starts[i] - start))
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        boundaries.append((idx, h))

    boundaries.sort(key=lambda x: x[0])

    if not boundaries or boundaries[0][0] != 0:
        first_heading = next((h for h in headings if h.get("level") == "h2"), None)
        intro_title = first_heading["title"] if first_heading else "Opening the Homily"
        boundaries.insert(0, (0, {"title": intro_title, "level": "h2"}))

    out: List[str] = []
    rendered_titles: List[str] = []

    for pos, (idx, heading) in enumerate(boundaries):
        next_idx = boundaries[pos + 1][0] if pos + 1 < len(boundaries) else len(segments)
        if next_idx <= idx:
            continue

        tag = "##" if heading.get("level") == "h2" else "###"
        title = str(heading.get("title") or "").strip()
        if title:
            out.append(f"{tag} {title}")
            rendered_titles.append(title)

        section_texts = [str(s.get("text") or "").strip() for s in segments[idx:next_idx]]
        out.extend(_paragraphize_segment_texts(section_texts))

    return out, rendered_titles


# ---------------------------
# Core generator
# ---------------------------
def mdx_generator(homily_text: str, segments: Optional[List[Dict[str, Any]]] = None) -> str:
    today = datetime.date.today()
    previous_sunday = get_previous_sunday(today)
    date_str = previous_sunday.strftime("%Y-%m-%d")
    mod_date_str = today.strftime("%Y-%m-%d")

    mass_hint = infer_mass_hint(homily_text)
    reading = get_1962_reading(previous_sunday, mass_hint)
    liturgical_block = ""
    if reading and reading.get("text") and reading.get("label"):
        liturgical_block = (
            f"> **{reading['label']} (1962 Missal)**\n"
            f"> {reading['text']}\n"
        )

    normalized_segments = _normalize_segments(segments)

    duration = _duration_from_segments(normalized_segments, homily_text)
    chapter_guidance = _chapter_guidance(duration)
    timed_transcript = _build_timed_transcript(normalized_segments, homily_text)

    prompt = f"""
You are an MDX formatter for a Traditional Catholic homily. DO NOT return the homily body.
Output ONLY a single JSON object with these keys:

{{
  "front_matter": {{
    "title": "<string>",
    "description": "<string>",
    "keywords": "<comma-separated>",
    "youtube_category": "Education",
    "youtube_description": "<EXACTLY three paragraphs beginning with the donation preface as provided below>",
    "youtube_hash": "<comma-separated>",
    "thumbnail_idea": "<short visual subject phrase for the YouTube thumbnail>",
    "mdx_file": "src/mds/lectures/<kebab>.mdx",
    "category": "lectures",
    "slug": "/<kebab>",
    "date": "{date_str}",
    "modDate": "{mod_date_str}",
    "author": "<string or empty if unknown>",
    "media_type": "video",
    "media_path": "<YouTube ID or empty>",
    "media_title": "<= same as title>",
    "media_alt": "<string>",
    "media_aria": "<string>",
    "prev_topic_label": "",
    "prev_topic_path": "",
    "next_topic_label": "",
    "next_topic_path": ""
  }},
  "toc": ["<H2 or H3 title>", "<H2/H3>", "..."],
  "headings": [
    {{ "start": 0.0, "level": "h2", "title": "<chapter-style heading>" }},
    {{ "start": 183.4, "level": "h3", "title": "<optional supporting heading>" }}
  ],
  "summary_paragraphs": ["<para 1>", "<para 2>", "<para 3 (optional)>"],
  "shorts": [
    {{
      "title": "<<=80 chars>",
      "quote": "<<=220 chars verbatim>",
      "start": <float|null>,
      "end": <float|null>,
      "keywords": ["word","word"]
    }}
  ],
  "chapters": [
    {{
      "title": "<section title used as chapter>",
      "anchor": "<kebab-case-anchor>",
      "start": <float|null>
    }}
  ]
}}

STRICT RULES:
- Use ONLY the HOMILY_TEXT for headings/TOC/summaries/quotes/shorts/chapters. Do NOT invent lines.
- Do NOT echo or return the homily body anywhere in the JSON.
- thumbnail_idea should be a concise phrase naming the central sermon subject for an image prompt.
- Do not include style instructions in thumbnail_idea.

YOUTUBE DESCRIPTION (IMPORTANT FORMATTING):
- youtube_description MUST preserve paragraph breaks using blank lines.
- It MUST begin EXACTLY with these lines (each on its own line):

Please click on the link to Contribute to our project.
https://www.mylatinmass.com/donate

Thank you. All contributions are greatly appreciated.
- - -
ABOUT THIS VIDEO:

- Then write EXACTLY three paragraphs, separated by ONE blank line between paragraphs.
- After those three paragraphs, add ONE blank line and then EXACTLY these two lines:

Subscribe to this channel for videos about:
Latin Mass, Traditional Catholic Teaching, Tridentine Mass, SSPX, Our Lady of Victory, Our Lady of the Most Holy Rosary, Saint Philomena, and more...

KEYWORDS:
- Front matter keywords must include: "Latin Mass", "Tridentine Mass", "Traditional Catholic" + other relevant terms.

HEADINGS & TOC:
- Before writing JSON, silently map the homily's argument: opening claim, scripture/doctrine, examples, applications, warnings, and final exhortation.
- Think like a careful Catholic editor creating reader wayfinding and YouTube chapters.
- Headings must describe what the NEXT section is about, not summarize the previous section.
- Prefer concrete, specific titles tied to the sermon: doctrine, image, Gospel scene, warning, virtue, sin, grace, or exhortation.
- Avoid generic labels such as "Introduction", "Faith and Hope", "The Gospel", "Conclusion", "Reflection", or "Final Thoughts" unless the title names the actual point.
- Use Title Case, 3-8 words, no punctuation unless needed.
- Bad headings: "The Gospel", "Faith and Love", "Conclusion".
- Better headings: "The Pharisee Refuses Mercy", "Grace Begins With Contrition", "Carry the Cross Without Complaint".
- h2 headings should be the main YouTube chapter movements. h3 headings are allowed only when a shorter subsection genuinely helps the reader.
- Use start times from TIMED_TRANSCRIPT. The first h2/chapter should begin at 0.0.
- The toc should match the visible heading titles in order.
- {chapter_guidance}

SHORTS/CHAPTERS:
- Chapters are for YouTube and must be excellent.
- Every chapter title must be useful to someone deciding whether to keep watching.
- Use the preacher's actual logic, not equal spacing. It is fine for two chapters to be close together if the homily turns sharply; it is also fine for a chapter to run longer if one idea is being developed.
- Chapters should mirror the h2 headings and use the same start seconds.
- If there are h3 headings, include them in headings but normally not in chapters.
- If TIMED_TRANSCRIPT is unavailable, set chapter starts to null.

HOMILY_TEXT (do NOT echo back):
{homily_text}

TIMED_TRANSCRIPT (use this for chapter timing and logical section starts):
{timed_transcript}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.25,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful MDX formatter for Catholic homilies. "
                    "Return only a single valid JSON object per instructions. "
                    "Never include the homily text itself in your response."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    payload_raw = (response.choices[0].message.content or "").strip()
    if not payload_raw:
        raise RuntimeError("Model returned empty content.")

    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", payload_raw, re.DOTALL)
    if m:
        payload_raw = m.group(1).strip()

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON ({e}).")

    fm = payload.get("front_matter", {}) or {}
    toc = payload.get("toc", []) or []
    raw_headings = payload.get("headings", []) or []
    summary_paras = payload.get("summary_paragraphs", []) or []
    shorts = payload.get("shorts", []) or []
    raw_chapters = payload.get("chapters", []) or []
    chapters = _normalize_chapters(raw_chapters, normalized_segments, duration)
    if not chapters:
        chapters = _fallback_chapters_from_headings(raw_headings, toc, normalized_segments, duration)

    kebab = _slugify(fm.get("title") or "homily")

    fm_defaults = {
        "title": fm.get("title") or "Untitled Homily",
        "description": fm.get("description") or "",
        "keywords": fm.get("keywords") or "Latin Mass, Tridentine Mass, Traditional Catholic",
        "youtube_category": fm.get("youtube_category") or "Education",
        "youtube_description": fm.get("youtube_description") or "",
        "youtube_hash": fm.get("youtube_hash") or "",
        "thumbnail_idea": fm.get("thumbnail_idea") or fm.get("title") or "Traditional Catholic homily",
        "mdx_file": fm.get("mdx_file") or f"src/mds/lectures/{kebab}.mdx",
        "category": fm.get("category") or "lectures",
        "slug": fm.get("slug") or f"/{kebab}",
        "date": fm.get("date") or date_str,
        "modDate": fm.get("modDate") or mod_date_str,
        "author": fm.get("author") or "",
        "media_type": fm.get("media_type") or "video",
        "media_path": fm.get("media_path") or "",
        "media_title": fm.get("media_title") or (fm.get("title") or "Untitled Homily"),
        "media_alt": fm.get("media_alt") or (fm.get("title") or "Homily video"),
        "media_aria": fm.get("media_aria") or (fm.get("title") or "Homily video"),
        "prev_topic_label": fm.get("prev_topic_label") or "",
        "prev_topic_path": fm.get("prev_topic_path") or "",
        "next_topic_label": fm.get("next_topic_label") or "",
        "next_topic_path": fm.get("next_topic_path") or "",
    }

    # ✅ FIX #2: Inject title-based keywords into description keyword list
    fm_defaults["youtube_description"] = inject_title_keywords_into_youtube_description(
        fm_defaults["youtube_description"],
        fm_defaults["title"],
    )

    # ---------------------------
    # YAML front matter
    # ---------------------------
    yaml_lines: List[str] = ["---"]

    # Write most fields inline...
    inline_keys = [
        "title",
        "description",
        "keywords",
        "youtube_category",
        "youtube_hash",
        "thumbnail_idea",
        "mdx_file",
        "category",
        "slug",
        "date",
        "modDate",
        "author",
        "media_type",
        "media_path",
        "media_title",
        "media_alt",
        "media_aria",
        "prev_topic_label",
        "prev_topic_path",
        "next_topic_label",
        "next_topic_path",
    ]
    for k in inline_keys:
        yaml_lines.append(f'{k}: "{_yaml_escape_inline(str(fm_defaults[k]))}"')

    # ✅ FIX #1: write youtube_description as a block scalar to preserve line breaks
    yaml_lines.extend(_yaml_block_scalar("youtube_description", fm_defaults["youtube_description"]))

    # shorts array
    yaml_lines.append("shorts:")
    for clip in shorts:
        t = clip.get("title") or ""
        q = clip.get("quote") or ""
        st = clip.get("start", None)
        en = clip.get("end", None)
        kws = clip.get("keywords") or []
        yaml_lines.append(
            f'  - {{"title": "{_yaml_escape_inline(t)}", "quote": "{_yaml_escape_inline(q)}", '
            f'"start": {json.dumps(st)}, "end": {json.dumps(en)}, '
            f'"keywords": {json.dumps(kws, ensure_ascii=False)} }}'
        )

    # chapters array
    yaml_lines.append("chapters:")
    for ch in chapters:
        title = ch.get("title") or ""
        anchor = ch.get("anchor") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        start = ch.get("start", None)
        yaml_lines.append(
            f'  - {{"title": "{_yaml_escape_inline(title)}", "anchor": "{_yaml_escape_inline(anchor)}", '
            f'"start": {json.dumps(start)} }}'
        )

    yaml_lines.append("---\n")

    # Body: H1
    body_lines: List[str] = [f'# {fm_defaults["title"]}\n']

    if liturgical_block:
        body_lines.append(liturgical_block + "\n")

    homily_stripped = homily_text.strip()
    paras = [p for p in re.split(r"\n\s*\n", homily_stripped) if p.strip()]
    if not paras:
        paras = [""]

    normalized_headings = _normalize_headings(
        raw_headings,
        chapters,
        toc,
        normalized_segments,
        len(paras),
    )

    if normalized_segments:
        out_paras, rendered_heading_titles = _render_body_from_segments(
            normalized_segments,
            normalized_headings,
        )
    else:
        ins_map: Dict[int, List[str]] = {}
        rendered_heading_titles: List[str] = []
        for h in normalized_headings:
            idx = max(0, min(int(h.get("para_index") or 0), len(paras) - 1))
            tag = "##" if h["level"] == "h2" else "###"
            ins_map.setdefault(idx, []).append(f"{tag} {h['title']}")
            rendered_heading_titles.append(h["title"])

        out_paras = []
        for i, p in enumerate(paras):
            if i in ins_map:
                for hdr in ins_map[i]:
                    out_paras.append(hdr)
            out_paras.append(p)

    chapter_titles = [ch["title"] for ch in chapters if ch.get("title")]
    toc_source = [
        str(t).strip()
        for t in (rendered_heading_titles or chapter_titles or toc)
        if str(t).strip()
    ]
    if toc_source:
        body_lines.append("## Summary of Headings\n")
        for t in toc_source:
            anchor = _slugify(t)
            body_lines.append(f"- [{t}](#{anchor})")
        body_lines.append("")

        # Guarantee each TOC anchor exists in the body.
        existing = {t.lower() for t in rendered_heading_titles}
        for t in toc_source:
            if t.lower() not in existing:
                out_paras.append(f"## {t}")
                existing.add(t.lower())

    body_lines.append("\n\n".join(out_paras))
    body_lines.append("")

    if summary_paras:
        body_lines.append("\n## Summary\n")
        for sp in summary_paras:
            body_lines.append(sp)

    mdx_page = "\n".join(yaml_lines + body_lines).strip()
    return mdx_page


# ---------------------------
# Helpers: extract ONLY the homily
# ---------------------------
def extract_homily_from_transcript(
    transcript: Dict[str, Any],
    transcription_json_path: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    if "homily_text" in transcript and "homily_segments" in transcript:
        return transcript["homily_text"], transcript["homily_segments"]

    audio_file = transcript.get("audio_file", "path/to/audio.mp3")
    working_dir = os.path.dirname(transcription_json_path)

    start, end, text, segments = find_homily(
        transcript,
        audio_file=audio_file,
        working_dir=working_dir,
    )
    return text, segments


# ---------------------------
# CLI wrapper
# ---------------------------
def generate_mdx_from_json(transcription_json_path: str) -> str:
    with open(transcription_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    homily_text, homily_segments = extract_homily_from_transcript(
        transcript, transcription_json_path
    )

    mdx_content = mdx_generator(homily_text, segments=homily_segments)
    return mdx_content


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python mdx_generator.py path/to/transcription.txt.json")
        sys.exit(1)

    transcription_json_path = sys.argv[1]
    mdx_content = generate_mdx_from_json(transcription_json_path)
    print(mdx_content)
