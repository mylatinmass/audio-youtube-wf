"""
Lyric-alignment with Whisper (no WhisperX required)

Goal:
- You provide the *exact* Latin lyrics (as text).
- Whisper transcribes audio with word timestamps.
- We then *align the known lyrics* to Whisper’s timestamped words and
  output line-level timestamps (SRT + JSON).

Notes:
- Chant is sung, so ASR can be imperfect. We’re using alignment to “snap”
  your known text onto Whisper’s time-stamped word stream.
- For best results: use a bigger model (large), set language="la", and keep
  lyrics chunked into the same lines you want displayed.

Install:
  pip install -U openai-whisper

Usage:
  result = align_lyrics_to_audio("chant.mp3", LYRICS_TEXT, directory=".")
"""

import os
import re
import json
from datetime import timedelta
from typing import List, Dict, Any, Optional

import whisper


# ----------------------------
# Helpers
# ----------------------------

def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    base = str(timedelta(seconds=int(seconds)))
    # timedelta renders H:MM:SS for < 1 day; SRT expects HH:MM:SS
    if base.count(":") == 2 and len(base.split(":")[0]) == 1:
        base = "0" + base
    return f"{base},{ms:03d}"


def normalize_token(s: str) -> str:
    """
    Normalize for alignment:
    - lowercase
    - strip punctuation
    - collapse whitespace
    - keep only letters (Latin) and apostrophes if present
    """
    s = s.lower()
    # Replace common ecclesiastical punctuation and dashes with spaces
    s = re.sub(r"[\u2018\u2019\u201C\u201D]", "'", s)
    s = re.sub(r"[-–—]", " ", s)
    # Remove punctuation except apostrophes
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    # Remove digits/underscores (Whisper sometimes returns weird tokens)
    s = re.sub(r"[\d_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    norm = normalize_token(text)
    return [t for t in norm.split(" ") if t]


def safe_min(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def safe_max(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def similarity(a: str, b: str) -> float:
    """
    Very small fuzzy score without external deps.
    Exact match => 1.0
    Prefix-ish / small edit-ish => partial score.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # quick prefix/contain checks
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        return max(0.0, shorter / longer)
    # tiny character overlap heuristic
    aset = set(a)
    bset = set(b)
    inter = len(aset & bset)
    union = len(aset | bset)
    return inter / union if union else 0.0


# ----------------------------
# Core: Whisper -> word stream
# ----------------------------

def transcribe_with_word_timestamps(
    audio_file: str,
    model_size: str = "large",
    language: str = "la",
    prompt: str = "",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Uses openai-whisper to transcribe with word timestamps.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model = whisper.load_model(model_size)

    # Whisper’s word timestamps are best effort; for sung audio it can drift.
    # We reduce "creative" drift by turning off conditioning on previous text.
    result = model.transcribe(
        audio_file,
        verbose=verbose,
        language=language,
        word_timestamps=True,
        initial_prompt=prompt,
        condition_on_previous_text=False
    )
    return result


def extract_word_stream(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten Whisper segments into a single list of word dicts:
      {"w": normalized_word, "raw": original, "start": float, "end": float}
    """
    words: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            raw = (w.get("word") or "").strip()
            if not raw:
                continue
            norm = normalize_token(raw)
            # A "word" may normalize to multiple tokens if it had punctuation/dash.
            toks = norm.split()
            if not toks:
                continue
            if len(toks) == 1:
                words.append({"w": toks[0], "raw": raw, "start": w.get("start"), "end": w.get("end")})
            else:
                # If a single whisper word becomes multiple tokens, split time evenly.
                start = w.get("start")
                end = w.get("end")
                if start is None or end is None or end <= start:
                    # fallback: keep as one lump
                    for t in toks:
                        words.append({"w": t, "raw": raw, "start": start, "end": end})
                else:
                    dur = end - start
                    step = dur / len(toks)
                    for i, t in enumerate(toks):
                        words.append({"w": t, "raw": raw, "start": start + i * step, "end": start + (i + 1) * step})
    return words


# ----------------------------
# Alignment: known lyrics -> word stream
# ----------------------------

def align_lyrics_lines_to_words(
    lyrics_lines: List[str],
    word_stream: List[Dict[str, Any]],
    min_word_match_score: float = 0.65,
    max_skip_words: int = 40
) -> List[Dict[str, Any]]:
    """
    Greedy monotonic alignment:
    For each lyric line, we find its tokens sequentially in the word stream.
    - Allows skipping up to max_skip_words between matches.
    - Uses a small fuzzy similarity score per token.
    Returns one entry per line:
      {"line": original_line, "start": float|None, "end": float|None, "matched": int, "total": int}
    """
    aligned: List[Dict[str, Any]] = []
    i = 0  # pointer into word_stream

    for line in lyrics_lines:
        tokens = tokenize(line)
        if not tokens:
            aligned.append({"line": line, "start": None, "end": None, "matched": 0, "total": 0})
            continue

        line_start: Optional[float] = None
        line_end: Optional[float] = None
        matched = 0

        # For each expected token in the lyric line, scan forward in word_stream.
        for t in tokens:
            best_j = None
            best_score = 0.0

            scan_limit = min(len(word_stream), i + max_skip_words)
            for j in range(i, scan_limit):
                ws = word_stream[j]["w"]
                score = similarity(t, ws)
                if score > best_score:
                    best_score = score
                    best_j = j
                    if score >= 1.0:
                        break  # perfect match, stop scanning early

            if best_j is not None and best_score >= min_word_match_score:
                # accept match
                wobj = word_stream[best_j]
                line_start = safe_min(line_start, wobj.get("start"))
                line_end = safe_max(line_end, wobj.get("end"))
                matched += 1
                i = best_j + 1  # move forward
            else:
                # token not found well enough; keep going (line may still get partial timing)
                continue

        aligned.append({
            "line": line,
            "start": line_start,
            "end": line_end,
            "matched": matched,
            "total": len(tokens)
        })

    return aligned


def aligned_lines_to_srt(aligned_lines: List[Dict[str, Any]]) -> str:
    """
    Convert aligned line timings into SRT.
    - If a line has no timing, we skip it.
    - If a line has start but missing end, we give it a small default duration.
    """
    out = []
    idx = 1

    for item in aligned_lines:
        line = item["line"].rstrip()
        start = item.get("start")
        end = item.get("end")

        if start is None:
            continue
        if end is None or end <= start:
            end = start + 2.5  # default 2.5s

        out.append(str(idx))
        out.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        out.append(line)
        out.append("")  # blank line
        idx += 1

    return "\n".join(out).strip() + "\n"


# ----------------------------
# Main: align lyrics to audio
# ----------------------------

def align_lyrics_to_audio(
    audio_file: str,
    lyrics_text: str,
    directory: str = "./",
    model_size: str = "large",
    language: str = "la",
    min_word_match_score: float = 0.65,
    max_skip_words: int = 40,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Creates:
    - whisper_result.json
    - aligned_lyrics.json
    - aligned_lyrics.srt

    Returns a dict with whisper result + aligned lines.
    """
    os.makedirs(directory, exist_ok=True)

    lyrics_lines = [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]

    # Use the lyrics as a prompt to bias recognition toward your exact text.
    # Keep prompt length reasonable; Whisper will still “best-effort” match.
    prompt = (
        "This audio is a Gregorian chant in Ecclesiastical Latin. "
        "Transcribe the chant accurately. The lyrics closely follow:\n"
        + "\n".join(lyrics_lines[:50])  # cap prompt lines to avoid huge prompt
    )

    whisper_result = transcribe_with_word_timestamps(
        audio_file=audio_file,
        model_size=model_size,
        language=language,
        prompt=prompt,
        verbose=verbose
    )

    word_stream = extract_word_stream(whisper_result)

    aligned = align_lyrics_lines_to_words(
        lyrics_lines=lyrics_lines,
        word_stream=word_stream,
        min_word_match_score=min_word_match_score,
        max_skip_words=max_skip_words
    )

    srt_text = aligned_lines_to_srt(aligned)

    # Save outputs
    whisper_json_path = os.path.join(directory, "whisper_result.json")
    aligned_json_path = os.path.join(directory, "aligned_lyrics.json")
    srt_path = os.path.join(directory, "aligned_lyrics.srt")

    with open(whisper_json_path, "w", encoding="utf-8") as f:
        json.dump(whisper_result, f, ensure_ascii=False, indent=2)

    with open(aligned_json_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    return {
        "whisper_result_path": whisper_json_path,
        "aligned_json_path": aligned_json_path,
        "srt_path": srt_path,
        "aligned_lines": aligned
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    LYRICS_TEXT = """Veni, creator Spiritus,
mentes tuorum visita,
imple superna gratia,
quae tu creasti pectora.

Qui diceris Paraclitus,
altissimi donum Dei,
fons vivus, ignis, caritas,
et spiritalis unctio.

Tu, septiformis munere,
digitus paternae dexterae,
Tu rite promissum Patris,
sermone ditans guttura.

Accende lumen sensibus,
infunde amorem cordibus,
infirma nostri corporis
virtute firmans perpeti.

Hostem repellas longius
pacemque dones protinus;
ductore sic te praevio
vitemus omne noxium.

Per te sciamus da Patrem,
noscamus atque Filium;
Teque utriusque Spiritum
credamus omni tempore.

Deo Patri sit gloria,
et Filio, qui a mortuis
surrexit, ac Paraclito,
in saeculorum saecula.
Amen.
"""

    audio_file = "/Users/mainmarketing/Downloads/veni-creator.mp3"
    out = align_lyrics_to_audio(
        audio_file=audio_file,
        lyrics_text=LYRICS_TEXT,
        directory="./",
        model_size="large",
        language="la",
        min_word_match_score=0.65,
        max_skip_words=60,
        verbose=False
    )
    print("Wrote:", out["srt_path"])
