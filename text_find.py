import re

# ─── single source of truth for your liturgical boundary ───
# this will only match “holy ghost amen” or “holy spirit amen”
HOMILY_MARKER = "holy ghost|spirit amen"

def normalize_text(text):
    """Remove punctuation and lowercase the text."""
    return re.sub(r'[^\w\s\|]', '', text).lower()

def parse_search_pattern(pattern):
    """
    Parse a space-delimited pattern into tokens.
    Supports:
      - Alternatives with ‘|’
      - Optional tokens via trailing ‘?’
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
    if 'word' in pattern_token:
        return transcript_token == pattern_token['word']
    return transcript_token in pattern_token['alternatives']

def match_pattern(words, i, pattern, p_index):
    # success if we’ve consumed the entire pattern
    if p_index == len(pattern):
        return i
    # if we run out of words, only match if all remaining tokens are optional
    if i >= len(words):
        for pt in pattern[p_index:]:
            if not pt['optional']:
                return None
        return i

    current = words[i]['token']
    pat = pattern[p_index]

    # optional token: try skipping first, then consuming if it matches
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

    # flatten all words
    words = []
    for seg in transcript.get('segments', []):
        for w in seg.get('words', []):
            token = normalize_text(w['word']).strip()
            words.append({'token': token, 'start': w['start'], 'end': w['end']})

    if not words:
        return None, None

    # apply skip / backwards logic
    if backwards:
        max_end = max(w['end'] for w in words)
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
        ms = int((s % 1) * 1000)
        s_int = int(s)
        m, sec = divmod(s_int, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (st, en, txt) in enumerate(video_segments, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(st + shift)} --> {fmt_time(en + shift)}\n")
            f.write(f"{txt.strip()}\n\n")

def find_homily(transcript, marker=HOMILY_MARKER):
    """
    Extract everything from the FIRST <marker> through the LAST <marker>.
    Returns: (first, last, homily_text, video_segments)
    """

    # ─── regression‐test: ensure bare “holy ghost” doesn’t match ───
    dummy = {'segments': [{
        'words': [
            {'word': 'Holy',  'start': 0.0, 'end': 0.5},
            {'word': 'Ghost', 'start': 0.5, 'end': 1.0}
        ]
    }]}
    bad_start, _ = find_phrase_timestamps(dummy, marker)
    assert bad_start is None, (
        f"⚠️ Your marker '{marker}' is matching bare 'Holy Ghost'—"
        "make ‘Amen’ mandatory!"
    )

    # find opening boundary
    first, _ = find_phrase_timestamps(transcript, marker, backwards=False)
    if first is None:
        raise RuntimeError(f"Opening marker '{marker}' not found")

    # find closing boundary
    _, last = find_phrase_timestamps(transcript, marker, backwards=True)
    if last is None:
        raise RuntimeError(f"Closing marker '{marker}' not found")

    # build trimmed segments
    trimmed = []
    for seg in transcript.get('segments', []):
        words = [w for w in seg['words'] if first <= w['start'] <= w['end'] <= last]
        if not words:
            continue
        trimmed.append({
            'start': words[0]['start'] - first,
            'end':   words[-1]['end']   - first,
            'text':  " ".join(w['word'] for w in words),
            'words': words
        })

    homily_words   = [w['word'] for seg in trimmed for w in seg['words']]
    homily_text    = " ".join(homily_words)
    video_segments = [(s['start'], s['end'], s['text']) for s in trimmed]

    return first, last, homily_text, video_segments