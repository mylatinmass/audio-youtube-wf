import os
import whisper
import json
from datetime import timedelta

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds))) 

def audio_to_text(audio_file, directory="./"):
    # audio_file="/Users/mainmarketing/Downloads/test-audio.mp3"
    
    # Prompt the user for the audio file path.    
    if not os.path.exists(audio_file):
        print("The specified file does not exist. Please check the path and try again.")
        return
    
    # Load the Whisper model (choose a model size appropriate for your system; 'base' is a good starting point).
    print("Loading the Whisper model...")
    model = whisper.load_model("large")
    
    prompt_text = (
        "This audio is a Catholic homily delivered in a church setting. "
        "It includes common religious phrases and liturgical expressions such as "
        "'Amen', 'in the name of the Father, the Son, and the Holy Ghost', 'daily readings based on catholic liturgy' "
        "and other similar expressions. Please ensure these terms are recognized and transcribed accurately." 
        "It may also include words or phrases in Ecclesiastical Latin. This sermon is for the First Sunday of Lent in the traditional Catholic liturgical calendar."
    )
    # Transcribe the audio file. The result includes 'segments' that contain timestamped text.
    print("Transcribing the audio file. This may take a few minutes...")
    result = model.transcribe(audio_file, verbose=True, language="en", 
                              word_timestamps=True,
                              initial_prompt=prompt_text)
    
    # Process the segments to create a transcript with timestamps.
    transcript_lines = []
    for segment in result.get("segments", []):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        transcript_lines.append(f"[{start} - {end}] {text}")
    
    transcript_text = "\n".join(transcript_lines)
    
    # Determine the desktop path.
    output_file = os.path.join(directory, "transcription.txt")
    
    # Write the transcript to a text file on the desktop.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(output_file + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=4))
        
    print(f"Transcription complete. The transcript has been saved to:\n{output_file}")

    return result

if __name__ == "__main__":
    # audio_file = clean_path( input("Enter the path to the audio file (e.g., /path/to/audio.wav): ").strip())
    audio_file="/Users/mainmarketing/Downloads/sermon-02-09.mp3"
    audio_to_text(audio_file)