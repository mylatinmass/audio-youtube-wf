from pydub import AudioSegment

def clip_audio_segment(
    input_file: str,
    start_sec: float = 0,
    end_sec: float = None,
    drop_last_sec: float = None,
    output_file: str = "output.mp3"
):
    """
    Clips the input audio file:
      – starting at `start_sec` seconds in,
      – ending at `end_sec` seconds (if provided),
      – or dropping the last `drop_last_sec` seconds (if provided).
    """
    audio = AudioSegment.from_file(input_file)
    start_ms = int(start_sec * 1000)

    if drop_last_sec is not None:
        end_ms = len(audio) - int(drop_last_sec * 1000)
    elif end_sec is not None:
        end_ms = int(end_sec * 1000)
    else:
        end_ms = len(audio)

    # Safety checks
    start_ms = max(0, start_ms)
    end_ms   = min(len(audio), end_ms)

    clipped = audio[start_ms:end_ms]
    clipped.export(output_file, format="mp3")
    print(f"Saved clipped audio to {output_file} "
          f"(from {start_sec}s to {end_ms/1000:.2f}s)")
    
if __name__ == "__main__":
    # Example: drop first 6.5 s and last 6.4 s
    clip_audio_segment(
        input_file="full_homily.mp3",
        start_sec=6.5,
        drop_last_sec=6.4,
        output_file="clean_homily.mp3"
    )
