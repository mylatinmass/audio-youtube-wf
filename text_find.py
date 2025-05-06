import re
import os
import json

def normalize_text(text):
    """Remove punctuation and lowercase the text."""
    return re.sub(r'[^\w\s\|]', '', text).lower()

def parse_search_pattern(pattern):
    """
    Parse a search pattern string into a list of pattern tokens.
    
    Each token is a dict. It can have:
      - "word": a mandatory token.
      - "alternatives": a list of alternative strings.
    If the token ends with '?' then it is marked optional.
    """
    tokens = pattern.split()
    parsed = []
    for token in tokens:
        optional = False
        if token.endswith('?'):
            optional = True
            token = token[:-1]
        # Normalize the token (this removes punctuation and lowercases)
        normalized_token = normalize_text(token).strip()
        # If alternatives are provided using a pipe, split them.
        if '|' in normalized_token:
            alternatives = [alt.strip() for alt in normalized_token.split('|')]
            parsed.append({'alternatives': alternatives, 'optional': optional})
        else:
            parsed.append({'word': normalized_token, 'optional': optional})
    return parsed


def token_matches(transcript_token, pattern_token):
    """
    Check if a transcript token (a string) matches the given pattern token.
    The pattern token may specify a "word" or a list of "alternatives".
    """
    if 'word' in pattern_token:
        return transcript_token == pattern_token['word']
    elif 'alternatives' in pattern_token:
        return transcript_token in pattern_token['alternatives']
    return False

def match_pattern(words, i, pattern, p_index):
    """
    Attempt to match the pattern tokens against the transcript words (list of dicts)
    starting at index i in words and p_index in pattern.
    
    Returns the transcript index after the last matched token if successful,
    or None if the match fails.
    
    This recursive function handles optional tokens:
      - For an optional token, we try both skipping it and consuming the transcript word.
    """
    # If we've processed all pattern tokens, the match succeeds.
    if p_index == len(pattern):
        return i
    # If transcript tokens are exhausted...
    if i >= len(words):
        # Match succeeds only if the remaining pattern tokens are all optional.
        for j in range(p_index, len(pattern)):
            if not pattern[j].get('optional', False):
                return None
        return i

    current_token = words[i]['token']
    current_pattern = pattern[p_index]

    if current_pattern.get('optional', False):
        # Option 1: Skip this optional token.
        res = match_pattern(words, i, pattern, p_index + 1)
        if res is not None:
            return res
        # Option 2: If the current transcript token matches, consume it.
        if token_matches(current_token, current_pattern):
            res = match_pattern(words, i + 1, pattern, p_index + 1)
            if res is not None:
                return res
        # Otherwise, the match fails at this branch.
        return None
    else:
        # For a mandatory token, the transcript token must match.
        if token_matches(current_token, current_pattern):
            return match_pattern(words, i + 1, pattern, p_index + 1)
        else:
            return None

def seconds_to_time(seconds):
    miliseconds = int((seconds % 1) * 1000)
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)    
    return f"{minutes:02d}:{seconds:02d}:{miliseconds:03d}"

def find_phrase_timestamps(transcript, search_phrase, backwards=False, skip=0.0):
    """
    Find the start and end timestamps of the search phrase pattern in the transcript.
    
    The search_phrase supports:
      - Optional words (by appending a '?' to the token).
      - Alternatives (using '|' between acceptable tokens).
    
    Parameters:
      transcript (dict): Contains segments with word details.
      search_phrase (str): The pattern to search for.
      backwards (bool): If False (default) returns the first occurrence (after skip).
                        If True, returns the last occurrence (before transcript end minus skip).
      skip (float): In forward mode, skip all words before this start time.
                    In backward mode, skip words after (max_end - skip) seconds.
    
    Returns:
      tuple: (start_timestamp, end_timestamp) of the matched phrase, or (None, None) if not found.
    """
    # Parse the search phrase into pattern tokens.
    pattern = parse_search_pattern(search_phrase)
    
    # Flatten the transcript's words into a list with normalized tokens.
    words_list = []
    for segment in transcript.get("segments", []):
        for word_info in segment.get("words", []):
            token = normalize_text(word_info["word"]).strip()
            words_list.append({
                "token": token,
                "start": word_info["start"],
                "end": word_info["end"]
            })
    
    if not words_list:
        return None, None
    
    # Apply the skip filter.
    if backwards:
        max_end = max(word["end"] for word in words_list)
        threshold = max_end - skip
        filtered_words = [w for w in words_list if w["end"] <= threshold]
    else:
        filtered_words = [w for w in words_list if w["start"] >= skip]
    
    # Collect all matches (each as a pair of indices into filtered_words).
    matches = []
    for i in range(len(filtered_words)):
        match_end = match_pattern(filtered_words, i, pattern, 0)
        if match_end is not None:
            matches.append((i, match_end))
    
    if not matches:
        return None, None
    
    # Return the first match (forward) or the last match (backwards).
    if backwards:
        match_i, match_end = matches[-1]
    else:
        match_i, match_end = matches[0]
    
    return filtered_words[match_i]["start"], filtered_words[match_end - 1]["end"]

def find_next_word(transcript, start):
    for segment in transcript.get("segments", []):
        for word_info in segment.get("words", []):
            if word_info["start"] > start:
                return word_info["start"], word_info["end"]
    return None

def find_prev_word(transcript, end):
    for segment in reversed(transcript.get("segments", [])):
        for word_info in reversed(segment.get("words", [])):
            if word_info["end"] < end:
                return word_info["start"], word_info["end"]
    return None

def generate_srt_file(video_segments, output_path, shift=11.0):
    def format_time(seconds):
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(video_segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_time(start+shift)} --> {format_time(end+shift)}\n")
            f.write(f"{text.strip()}\n\n")
    print(f"SRT file saved at: {output_path}")

def find_homily(transcript, output_dir=None):
    # Specify the phrase you want to search.
    # search_phrase = "In the? name? of? the? Father, and? of? the? Son, and? of? the? Holy Ghost|Spirit. Amen."
    search_phrase = "Holy Ghost|Spirit. Amen."
    start, end = find_phrase_timestamps(transcript, search_phrase) # [03:10.280 --> 03:13.980] 
    
    if start is not None:
        main_start, main_end = find_next_word(transcript, end)
        first = main_start
        last = main_end
        while start is not None:
            print(f"Found phrase from {seconds_to_time(start)} to {seconds_to_time(end)}.")        
            start, end = find_phrase_timestamps(transcript, search_phrase, backwards=False, skip=end)            
            if start is not None:
                main_start, main_end = find_prev_word(transcript, start)
                last = main_end

        print(f"The main part : {first}, {last}, total {seconds_to_time(last - first)}.")

    if start is not None:
        print(f"Found phrase from {start} to {end} seconds.")
    else:
        print("Phrase not found.")

    # Extract text between 'first' and 'last' timestamps.
    homily_words = []
    #for segment in transcript.get("segments", []):
    #    for word_info in segment.get("words", []):
    #        if word_info["start"] >= first and word_info["start"] <= last:
    #            homily_words.append(word_info["word"])
    
    # Build trimmed segments based on the original transcript segments.
    trimmed_segments = []
    for segment in transcript.get("segments", []):
        # Select only those words in the segment that lie within the overall homily boundaries.
        trimmed_words = [
            word_info for word_info in segment.get("words", [])
            if word_info["start"] >= first and word_info["start"] <= last
        ]
        if trimmed_words:
            # Adjust the segment's start and end times.
            new_segment = {
                "start": trimmed_words[0]["start"] - first,
                "end": trimmed_words[-1]["end"] - first,
                "text": " ".join(word["word"] for word in trimmed_words),
                "words": trimmed_words
            }
            trimmed_segments.append(new_segment)
            homily_words.extend(word["word"] for word in trimmed_words)
    homily_text = " ".join(homily_words)
    #print(f"Homily text: {homily_text}")
    video_segments = [(seg["start"], seg["end"], seg["text"]) for seg in trimmed_segments]
    return first, last, homily_text,  video_segments


if __name__ == "__main__":
    # Determine the desktop path.
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_file = os.path.join(desktop_path, "transcription.txt")
    transcript = {}

    with open(output_file + ".json", "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Get homily timestamps, full text, and segment list
    first, last, homily_text, video_segments = find_homily(transcript)

    # Now generate the SRT using the extracted video segments
    srt_output_path = os.path.join(desktop_path, "homily_captions.srt")
    generate_srt_file(video_segments, srt_output_path)

    