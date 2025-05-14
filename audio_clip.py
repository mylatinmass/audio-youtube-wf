# audio_clip.py

from pydub import AudioSegment

def clip_audio_segment(
    input_file: str,
    start_sec: float = 0.0,
    end_sec: float = None,
    output_file: str = "output.mp3",
    add_silence_sec: float = 0.0
):
    """
    Clips an audio file:
      – starts at `start_sec` seconds into input_file,
      – ends at `end_sec` seconds (if positive), or drops the last abs(end_sec) seconds if `end_sec` is negative,
      – appends `add_silence_sec` seconds of silence at the end,
      – and exports to `output_file` (mp3).

    :param input_file:         path to source audio (any format pydub supports)
    :param start_sec:          seconds from the start to begin clipping
    :param end_sec:            if > 0, clip up to this many seconds from start;
                               if < 0, clip up to (duration – abs(end_sec)) seconds;
                               if None, clip to the very end
    :param output_file:        path where the clipped mp3 will be saved
    :param add_silence_sec:    seconds of silence to append after the clip
    """
    audio = AudioSegment.from_file(input_file)
    total_ms = len(audio)

    # Compute start and end in milliseconds
    start_ms = int(start_sec * 1000)
    if end_sec is None:
        end_ms = total_ms
    elif end_sec >= 0:
        end_ms = int(end_sec * 1000)
    else:
        # drop the last abs(end_sec) seconds
        end_ms = total_ms + int(end_sec * 1000)  # end_sec is negative

    # safety bounds
    start_ms = max(0, min(start_ms, total_ms))
    end_ms   = max(start_ms, min(end_ms, total_ms))

    segment = audio[start_ms:end_ms]

    # append silence if requested
    if add_silence_sec > 0:
        silence = AudioSegment.silent(duration=int(add_silence_sec * 1000))
        segment = segment + silence

    # export as MP3
    segment.export(output_file, format="mp3")
    return output_file
