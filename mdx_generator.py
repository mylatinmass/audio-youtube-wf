# mdx_generator.py
import datetime
import json
import os
import re
from typing import Optional, List, Dict, Any

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

    # Common feasts (expand over time)
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

    Shape:
    {
      "label": "Introit" | "Collect" | ...,
      "text": "“…” (reference)"
    }
    """
    ymd = previous_sunday.strftime("%Y-%m-%d")

    # ========== START: your curated library (expand over time) ==========
    by_mass = {
        # "20th-sunday-after-pentecost": {
        #     "label": "Communion",
        #     "text": "“Tu mandasti mandata tua custodiri nimis…” (Ps 118) — Communion (1962 Missal)"
        # },
        # "all-saints": {
        #     "label": "Introit",
        #     "text": "“Gaudeamus omnes in Domino…” — Introit, Missa Omnium Sanctorum (1962)"
        # },
        # "christ-the-king": {
        #     "label": "Collect",
        #     "text": "“Omnipotens sempiterne Deus, qui dilecto Filio tuo universorum Rege...” — Collect (1962)"
        # },
    }

    by_date = {
        # "2025-11-02": {"label": "Communion", "text": "…” (ref)"},
    }
    # ========== END: your curated library ==========

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

    Accepts:
      - dict segments: {"start": ..., "end": ..., "text": ...}
      - tuple/list segments, e.g. (start, end, text) or (id, start, end, text)

    This prevents 'AttributeError: tuple object has no attribute get'
    when upstream code passes tuples instead of dicts.
    """
    normalized: List[Dict[str, Any]] = []

    for s in segments or []:
        # Case 1: already a dict from Whisper/JSON
        if isinstance(s, dict):
            normalized.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "") or "").strip(),
            })
            continue

        # Case 2: tuple or list
        if isinstance(s, (list, tuple)):
            start = 0.0
            end = 0.0
            text = ""

            if len(s) >= 3:
                # looks like (start, end, text)
                if isinstance(s[0], (int, float)) and isinstance(s[1], (int, float)):
                    start = s[0]
                    end = s[1]
                    text = s[2]
                # looks like (id, start, end, text)
                elif len(s) >= 4 and isinstance(s[1], (int, float)) and isinstance(s[2], (int, float)):
                    start = s[1]
                    end = s[2]
                    text = s[3]
                else:
                    # fallback: assume first three = (start, end, text)
                    start, end, text = s[0], s[1], s[2]

            normalized.append({
                "start": float(start),
                "end": float(end),
                "text": str(text or "").strip(),
            })

    return normalized


# ---------------------------
# Core generator
# ---------------------------
def mdx_generator(homily_text: str, segments: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generates the MDX page structure using ONLY the homily text.
    - The model returns JSON (front matter, TOC, headings, summary, shorts, chapters).
    - We assemble final MDX locally, preserving the homily text unchanged
      (just inserting headings before certain paragraphs).
    """

    today = datetime.date.today()
    previous_sunday = get_previous_sunday(today)
    date_str = previous_sunday.strftime("%Y-%m-%d")
    mod_date_str = today.strftime("%Y-%m-%d")

    # Determine Mass/feast hint from the homily text, then fetch ONE 1962 reading
    mass_hint = infer_mass_hint(homily_text)
    reading = get_1962_reading(previous_sunday, mass_hint)
    liturgical_block = ""
    if reading and reading.get("text") and reading.get("label"):
        liturgical_block = (
            f"> **{reading['label']} (1962 Missal)**\n"
            f"> {reading['text']}\n"
        )

    # Normalize segments first so we can handle dicts OR tuples
    normalized_segments = _normalize_segments(segments)

    # Compact segments to avoid token bloat
    _compact = []
    for s in normalized_segments:
        _compact.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s["text"] or "")[:120],  # slightly more context for better alignment
        })
    segments_json = json.dumps(_compact[:2000], ensure_ascii=False)  # hard cap

    # Prompt for JSON only; we assemble MDX locally
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
  "toc": [
    "<H2 or H3 title>", "<H2/H3>", "..."
  ],
  "headings": [
    {{ "para_index": 0, "level": "h2", "title": "<title>" }},
    {{ "para_index": 3, "level": "h3", "title": "<title>" }}
  ],
  "summary_paragraphs": [
    "<para 1>", "<para 2>", "<para 3 (optional)>"
  ],
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
- Use ONLY the HOMILY_TEXT for:
  - headings,
  - TOC titles,
  - summaries,
  - quotes,
  - shorts,
  - chapters.
  Do NOT invent lines.
- Do NOT echo or return the homily body anywhere in the JSON.
- Design subsections so each H2/H3 could stand alone as a short spiritual "mini-article."
- Prefer clear, punchy titles that can be used as anchor links and YouTube chapter titles.

YOUTUBE DESCRIPTION:
- youtube_description MUST begin exactly with:

  Please click on the link to Contribute to our project.
  https://www.mylatinmass.com/donate

  Thank you. All contributions are greatly appreciated.
  - - -
  ABOUT THIS VIDEO:

  Then write EXACTLY three paragraphs:
  (1) Main thesis + liturgical/scriptural context, referencing the Traditional Latin Mass / Tridentine Mass.
  (2) 2–3 key insights from the homily.
  (3) Pastoral application and encouragement for the listener.

  After those three paragraphs, add a blank line and then EXACTLY this text:

  Subscribe to this channel for videos about:
  Latin Mass, Traditional Catholic Teaching, Tridentine Mass, SSPX, Our Lady of Victory, Our Lady of the Most Holy Rosary, Saint Philomena, and more...

KEYWORDS:
- Front matter keywords must include at least these three terms:
  "Latin Mass", "Tridentine Mass", "Traditional Catholic"
  plus other relevant tags from the homily.

HEADINGS & TOC:
- For "headings", compute paragraph indices by splitting HOMILY_TEXT on blank lines.
  Insert each heading BEFORE the paragraph at the given index.
- "toc" should include all H2 and H3 titles in reading order.

SHORTS:
- If SEGMENTS_JSON is provided, align the quoted text to segments and compute start/end seconds
  (~30–45s windows at natural pauses). If you cannot find a good match, set both to null.

CHAPTERS:
- Chapters should correspond to the MAJOR H2 sections (and optionally big H3s).
- Each chapter title should be suitable as a YouTube chapter title.
- If SEGMENTS_JSON is provided, align each chapter to the approximate start second where
  that section begins in the audio. If you cannot find it, set start to null.
- "anchor" should be a kebab-case version of the title (for MDX slug anchors).

CONTEXT FOR PAGE HEADER (INCLUDE ONLY IF PRESENT WHEN ASSEMBLING LOCALLY; DO NOT COPY):
LITURGICAL_READING_BLOCK:
{liturgical_block if liturgical_block else "(none)"}

HOMILY_TEXT (unchanged, used only for analysis — do NOT echo back):
{homily_text}

SEGMENTS_JSON (optional for timing):
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
        debug_path = os.path.join(os.path.dirname(__file__), "last_model_response.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as dbg:
                dbg.write(str(response))
        except Exception:
            pass
        raise RuntimeError(
            "Model returned empty content. Full response saved to last_model_response.txt for inspection."
        )

    # Strip fences if any
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", payload_raw, re.DOTALL)
    if m:
        payload_raw = m.group(1).strip()

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as e:
        debug_path = os.path.join(os.path.dirname(__file__), "last_model_payload.json.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as dbg:
                dbg.write(payload_raw)
        except Exception:
            pass
        raise RuntimeError(
            f"Model did not return valid JSON ({e}). Raw payload saved to {debug_path}."
        )

    fm = payload.get("front_matter", {}) or {}
    toc = payload.get("toc", []) or []
    headings = payload.get("headings", []) or []
    summary_paras = payload.get("summary_paragraphs", []) or []
    shorts = payload.get("shorts", []) or []
    chapters = payload.get("chapters", []) or []

    # Build YAML front matter
    def _yaml_escape(s: str) -> str:
        return (s or "").replace('"', '\\"')

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

    yaml_lines: List[str] = ["---"]
    for k in [
        "title",
        "description",
        "keywords",
        "youtube_category",
        "youtube_description",
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
    ]:
        yaml_lines.append(f'{k}: "{_yaml_escape(str(fm_defaults[k]))}"')

    # shorts array
    yaml_lines.append("shorts:")
    for clip in shorts:
        t = clip.get("title") or ""
        q = clip.get("quote") or ""
        st = clip.get("start", None)
        en = clip.get("end", None)
        kws = clip.get("keywords") or []
        yaml_lines.append(
            f'  - {{"title": "{_yaml_escape(t)}", "quote": "{_yaml_escape(q)}", '
            f'"start": {json.dumps(st)}, "end": {json.dumps(en)}, '
            f'"keywords": {json.dumps(kws, ensure_ascii=False)} }}'
        )

    # chapters array (for YouTube chapters, etc.)
    yaml_lines.append("chapters:")
    for ch in chapters:
        title = ch.get("title") or ""
        anchor = ch.get("anchor") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        start = ch.get("start", None)
        yaml_lines.append(
            f'  - {{"title": "{_yaml_escape(title)}", "anchor": "{_yaml_escape(anchor)}", '
            f'"start": {json.dumps(start)} }}'
        )

    yaml_lines.append("---\n")

    # Body: H1
    body_lines: List[str] = [f'# {fm_defaults["title"]}\n']

    # Single 1962 Missal reading (only if present)
    if liturgical_block:
        body_lines.append(liturgical_block + "\n")

    # TOC
    if toc:
        body_lines.append("## Summary of Headings\n")
        for t in toc:
            anchor = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
            body_lines.append(f"- [{t}](#{anchor})")
        body_lines.append("")

    # Insert headings into the homily text locally
    homily_stripped = homily_text.strip()
    paras = [p for p in re.split(r"\n\s*\n", homily_stripped) if p.strip()]

    # Map para index -> list of headings
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

    # Homily body with headings, preserving 100% of homily text
    body_lines.append("\n\n".join(out_paras))
    body_lines.append("")  # spacer

    # End summary (2–3 paragraphs)
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
) -> (str, List[Dict[str, Any]]):
    """
    Ensure we ONLY work with the homily portion.

    Priority order:
    1. If transcript already has 'homily_text' and 'homily_segments', use them.
    2. Else, call your existing find_homily(...) helper to extract homily text + segments.
    """
    # Case 1: pre-extracted homily from an upstream script
    if "homily_text" in transcript and "homily_segments" in transcript:
        return transcript["homily_text"], transcript["homily_segments"]

    # Case 2: use your finder to locate the homily in the full transcript
    audio_file = transcript.get("audio_file", "path/to/audio.mp3")
    working_dir = os.path.dirname(transcription_json_path)

    start, end, text, segments = find_homily(
        transcript,
        audio_file=audio_file,
        working_dir=working_dir,
    )
    # 'text' and 'segments' here should already be homily-only
    return text, segments


# ---------------------------
# CLI wrapper
# ---------------------------
def generate_mdx_from_json(transcription_json_path: str) -> str:
    # Load the JSON transcription (full file, not just homily)
    with open(transcription_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Extract ONLY the homily portion + its segments
    homily_text, homily_segments = extract_homily_from_transcript(
        transcript, transcription_json_path
    )

    # Generate MDX content using only the homily
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

