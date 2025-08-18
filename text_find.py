import re
import os
import json
from pydub.utils import mediainfo

# ─── single source of truth for your liturgical boundary ───
# matches “holy ghost amen” or “holy spirit amen”
HOMILY_MARKER = "holy ghost|spirit amen"
# seconds of silence to mark the end of the homily
SILENCE_THRESHOLD = 8.0

def normalize_text(text):
    """Remove punctuation and lowercase the text."""
    return re.sub(r'[^\w\s\|]', '', text).lower()

def parse_search_pattern(pattern):
    """
    Parse a space-delimited pattern into tokens.
    Supports:
      - Alternatives with '|'
      - Optional tokens via trailing '?'
    """
    tokens = pattern.split()
    parsed = []
    for token in tokens:
        optional = token.endswith('?')
        if optional:
            token = token[:-1]
        normalized = normalize_text(token).strip()
        if '|' in normalized:
            alts = [alt.strip() for alt in normalized.split('|')]
            parsed.append({'alternatives': alts, 'optional': optional})
        else:
            parsed.append({'word': normalized, 'optional': optional})
    return parsed

def token_matches(transcript_token, pattern_token):
    """Check if a transcript token matches a pattern token."""
    if 'word' in pattern_token:
        return transcript_token == pattern_token['word']
    return transcript_token in pattern_token['alternatives']

def match_pattern(words, i, pattern, p_index):
    """
    Recursive search: attempt to match pattern tokens starting at words[i].
    Returns the index after the last match, or None if no match.
    """
    # success if we've processed all pattern tokens
    if p_index == len(pattern):
        return i
    # if we run out of words, only succeed if remaining tokens are all optional
    if i >= len(words):
        for pt in pattern[p_index:]:
            if not pt['optional']:
                return None
        return i

    current = words[i]['token']
    pat     = pattern[p_index]

    # optional token: try skipping, then consuming if it matches
    if pat['optional']:
        skip = match_pattern(words, i, pattern, p_index + 1)
        if skip is not None:
            return skip
        if token_matches(current, pat):
            return match_pattern(words, i + 1, pattern, p_index + 1)
        return None

    # mandatory token
    if token_matches(current, pat):
        return match_pattern(words, i + 1, pattern, p_index + 1)
    return None

def find_phrase_timestamps(transcript, search_phrase, backwards=False, skip=0.0):
    """
    Returns (start, end) for the FIRST (or LAST, if backwards=True) match
    of `search_phrase` in the transcript, or (None, None) if none found.
    """
    pattern = parse_search_pattern(search_phrase)

    # flatten all words (support either segments→words or top-level words list)
    words = []
    if 'segments' in transcript:
        iterable = transcript['segments']
    elif 'words' in transcript:
        iterable = [{'words': transcript['words']}]
    else:
        iterable = []

    for seg in iterable:
        for w in seg.get('words', []):
            token = normalize_text(w['word']).strip()
            words.append({'token': token, 'start': w['start'], 'end': w['end']})

    if not words:
        return None, None

    # apply skip / backwards logic
    if backwards:
        max_end   = max(w['end'] for w in words)
        threshold = max_end - skip
        words = [w for w in words if w['end'] <= threshold]
    else:
        words = [w for w in words if w['start'] >= skip]

    # find all matches
    matches = []
    for i in range(len(words)):
        m = match_pattern(words, i, pattern, 0)
        if m is not None:
            matches.append((i, m))

    if not matches:
        return None, None

    idx, end_idx = (matches[-1] if backwards else matches[0])
    return words[idx]['start'], words[end_idx - 1]['end']

def generate_srt_file(video_segments, output_path, shift=11.0):
    """Write an SRT file from a list of (start, end, text) tuples."""
    def fmt_time(s):
        ms    = int((s % 1) * 1000)
        s_int = int(s)
        m, sec = divmod(s_int, 60)
        h, m   = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (st, en, txt) in enumerate(video_segments, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(st + shift)} --> {fmt_time(en + shift)}\n")
            f.write(f"{txt.strip()}\n\n")

# ─── Helpers to pivot around the marker ───

def find_next_word_start(transcript, timestamp):
    """
    Return the start time of the first word whose start > timestamp.
    """
    for seg in transcript.get('segments', []):
        for w in seg.get('words', []):
            if w['start'] > timestamp:
                return w['start']
    return None

def find_homily(transcript, marker=HOMILY_MARKER, silence_threshold=SILENCE_THRESHOLD, audio_file=None, working_dir=None):
    """
    Locate the homily by:
      1. Finding the opening marker ("holy ghost amen" or "holy spirit amen").
      2. Starting right after that marker.
      3. Scanning forward until there's a gap of at least `silence_threshold` seconds,
         which marks the end of the homily.
      4. If marker not found, fall back to manual start/end entry.
    Returns: first, last, homily_text, video_segments
    """
    try:
        # 1. Try finding opening marker
        fm_start, fm_end = find_phrase_timestamps(transcript, marker, backwards=False)
        if fm_end is None:
            raise RuntimeError(f"Opening marker '{marker}' not found")

        # 2. Homily starts after that 'Amen'
        first = find_next_word_start(transcript, fm_end) or fm_end

        # 3. Gather all words from 'first' onward
        words_after = []
        for seg in transcript.get('segments', []):
            for w in seg.get('words', []):
                if w['start'] >= first:
                    words_after.append(w)

        if not words_after:
            raise RuntimeError("No transcript words found after homily start")

        # 4. Detect silence gap to mark end
        last = None
        for prev, curr in zip(words_after, words_after[1:]):
            if curr['start'] - prev['end'] >= silence_threshold:
                last = prev['end']
                break
        if last is None:
            last = words_after[-1]['end']

        # Build trimmed segments
        trimmed = []
        for seg in transcript.get('segments', []):
            seg_words = [w for w in seg.get('words', []) if w['start'] >= first and w['end'] <= last]
            if not seg_words:
                continue
            trimmed.append({
                'start': seg_words[0]['start'] - first,
                'end':   seg_words[-1]['end']   - first,
                'text':  " ".join(w['word'] for w in seg_words),
                'words': seg_words
            })

        homily_words   = [w['word'] for seg in trimmed for w in seg['words']]
        homily_text    = " ".join(homily_words)
        video_segments = [(s['start'], s['end'], s['text']) for s in trimmed]

        return first, last, homily_text, video_segments

    except (UnboundLocalError, RuntimeError):
        print("🚫 Could not locate the homily section in your transcript.")
        print("   • You can now manually enter the start and end times.")

        def parse_time_input(t):
            if ":" in t:
                mins, secs = t.split(":")
                return float(mins) * 60 + float(secs)
            return float(t)

        while True:
            try:
                manual_start = float(parse_time_input(input("Enter START time (e.g., 123 or 2:03): ").strip()))
                break
            except ValueError:
                print("❌ Invalid start time. Try again.")

        while True:
            manual_end_input = input("Enter END time (or press Enter for end of file): ").strip()
            if manual_end_input == "":
                manual_end = None
                break
            try:
                manual_end = float(parse_time_input(manual_end_input))
                break
            except ValueError:
                print("❌ Invalid end time. Try again.")

        # Determine end time
        if manual_end:
            actual_end = manual_end
        else:
            try:
                if isinstance(transcript, dict) and "segments" in transcript:
                    actual_end = float(transcript["segments"][-1]["end"])
                else:
                    actual_end = float(mediainfo(audio_file)["duration"])
            except:
                actual_end = float(mediainfo(audio_file)["duration"])

        # Build segments between manual times
        if isinstance(transcript, dict) and "segments" in transcript:
            all_segments = transcript["segments"]
        elif isinstance(transcript, list):
            all_segments = transcript
        else:
            all_segments = []

        segments = [seg for seg in all_segments if float(seg.get("start", 0)) >= manual_start and float(seg.get("end", 0)) <= actual_end]
        text = " ".join(seg.get("text", "") for seg in segments)

        # Save manual JSON if working_dir provided
        if working_dir:
            homily_json_path = os.path.join(working_dir, "homily.json")
            with open(homily_json_path, "w", encoding="utf-8") as f:
                json.dump({"segments": segments, "text": text}, f, indent=2)

        return manual_start, actual_end, text, [(seg["start"], seg["end"], seg.get("text", "")) for seg in segments]
