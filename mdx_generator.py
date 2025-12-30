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

    _compact = []
    for s in normalized_segments:
        _compact.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s["text"] or "")[:120],
        })
    segments_json = json.dumps(_compact[:2000], ensure_ascii=False)

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
    {{ "para_index": 0, "level": "h2", "title": "<title>" }},
    {{ "para_index": 3, "level": "h3", "title": "<title>" }}
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
- Paragraph indices come from splitting HOMILY_TEXT on blank lines.

SHORTS/CHAPTERS:
- If SEGMENTS_JSON is provided, align timing; else set null.

HOMILY_TEXT (do NOT echo back):
{homily_text}

SEGMENTS_JSON:
{segments_json}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        max_completion_tokens=2048,
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
    headings = payload.get("headings", []) or []
    summary_paras = payload.get("summary_paragraphs", []) or []
    shorts = payload.get("shorts", []) or []
    chapters = payload.get("chapters", []) or []

    kebab = (fm.get("title") or "homily").strip().lower()
    kebab = re.sub(r"[^a-z0-9]+", "-", kebab).strip("-")

    fm_defaults = {
        "title": fm.get("title") or "Untitled Homily",
        "description": fm.get("description") or "",
        "keywords": fm.get("keywords") or "Latin Mass, Tridentine Mass, Traditional Catholic",
        "youtube_category": fm.get("youtube_category") or "Education",
        "youtube_description": fm.get("youtube_description") or "",
        "youtube_hash": fm.get("youtube_hash") or "",
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

    if toc:
        body_lines.append("## Summary of Headings\n")
        for t in toc:
            anchor = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
            body_lines.append(f"- [{t}](#{anchor})")
        body_lines.append("")

    homily_stripped = homily_text.strip()
    paras = [p for p in re.split(r"\n\s*\n", homily_stripped) if p.strip()]

    ins_map: Dict[int, List[str]] = {}
    for h in headings:
        idx = max(0, int(h.get("para_index", 0)))
        lvl = (h.get("level") or "h2").lower()
        ttl = h.get("title") or ""
        tag = "##" if lvl == "h2" else "###"
        ins_map.setdefault(idx, []).append(f"{tag} {ttl}")

    out_paras: List[str] = []
    for i, p in enumerate(paras):
        if i in ins_map:
            for hdr in ins_map[i]:
                out_paras.append(hdr)
        out_paras.append(p)

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
