def generate_video_script(transcript, homily_start, homily_end):
    """
    Generate a video script by filtering and trimming the transcript segments
    to include only words between homily_start and homily_end.
    
    The timestamps in each segment and for each word are reset so that homily_start becomes 0.
    
    Parameters:
        transcript (dict): The full transcript JSON (with a "segments" key, and each segment containing a "words" list).
        homily_start (float): The start timestamp (in seconds) of the homily.
        homily_end (float): The end timestamp (in seconds) of the homily.
    
    Returns:
        dict: A dictionary containing:
            - "segments": a list of trimmed segments with reset timestamps,
            - "homily_start": the original homily start,
            - "homily_end": the original homily end,
            - "homily_text": the concatenated homily text.
    """
    trimmed_segments = []
    homily_words = []

    # Process each segment in the transcript
    for segment in transcript.get("segments", []):
        # Filter out words that lie outside the homily boundaries.
        trimmed_words = [
            word for word in segment.get("words", [])
            if word["start"] >= homily_start and word["start"] <= homily_end
        ]
        if trimmed_words:
            # Reset timestamps for each word by subtracting homily_start.
            new_words = []
            for word in trimmed_words:
                new_word = word.copy()
                new_word["start"] = word["start"] - homily_start
                new_word["end"] = word["end"] - homily_start
                new_words.append(new_word)
            new_segment = {
                "start": new_words[0]["start"],
                "end": new_words[-1]["end"],
                "text": " ".join(word["word"] for word in new_words),
                "words": new_words
            }
            trimmed_segments.append(new_segment)
            homily_words.extend(word["word"] for word in new_words)

    homily_text = " ".join(homily_words)

    return {
        "segments": trimmed_segments,
        "homily_start": homily_start,
        "homily_end": homily_end,
        "homily_text": homily_text
    }
