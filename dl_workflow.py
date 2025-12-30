from audio_to_text import audio_to_text
from text_find import find_homily, generate_srt_file
from audio_clip import clip_audio_segment
from auphonic_audio_cleaner import start_production, download_file
from video_generator import create_text_video
from video_script import generate_video_script
import os
import sys
import json
import subprocess
from PIL import Image  # kept, though we no longer use it for thumbnails
import frontmatter
from dotenv import load_dotenv
from get_token import get_and_refresh_google_user_tokens, SCOPES
from youtube import (
    youtube_list_videos,
    youtube_upload_video_with_optional_thumbnail,
    youtube_update_video,
    youtube_upload_captions
)
from googleapiclient.errors import HttpError
from txt_to_json import txt_to_json
import re
import tempfile
import shutil


# Load environment variables (if using a .env file)
load_dotenv()

# --- Front-matter helpers (drop near the top) ---
_FRONT_RE = re.compile(r"^---\s*(.*?)\s*---", re.DOTALL | re.MULTILINE)
_KV_RE = re.compile(r'^(?P<key>[A-Za-z0-9_]+):\s*"(?P<val>.*?)"\s*$', re.MULTILINE)

def parse_front_matter_block(mdx_text: str) -> dict:
    m = _FRONT_RE.search(mdx_text)
    if not m:
        return {}
    block = m.group(1)
    front = {}
    for km in _KV_RE.finditer(block):
        front[km.group("key")] = km.group("val")
    return front

def resolve_final_mdx_path_from_front(front: dict, default_dir: str) -> str:
    """
    Priority:
      1) mdx_file (keep it relative under default_dir)
      2) slug + '.mdx'
      3) 'homily.mdx'
    """
    os.makedirs(default_dir, exist_ok=True)

    if front.get("mdx_file", "").strip():
        rel = front["mdx_file"].strip().lstrip("/\\")  # keep it relative
        target = os.path.join(default_dir, rel)
    else:
        slug = (front.get("slug") or "").strip()
        if slug.startswith("/"):
            slug = slug[1:]
        base = slug if slug else "homily"
        if not base.endswith(".mdx"):
            base += ".mdx"
        target = os.path.join(default_dir, base)

    # make sure parent folder exists
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target

# Prompt user for audio file ONLY (no image)
def prompt_user():
    audio_file = clean_path(input("Enter AUDIO FILE PATH (e.g., /path/to/audio.mp3): ").strip())
    # We no longer need an image; reuse audio_file string just to keep naming flow unchanged
    image_file = audio_file
    return audio_file, image_file

def clean_path(path):
    """Removes leading/trailing spaces and handles unnecessary quotes."""
    return path.strip().strip('"').strip("'")

def main():
    audio_file, image_file = prompt_user()
    dir = os.path.dirname(audio_file)
    working_dir = os.path.join(dir, "working")
    final_output_dir = os.path.join(dir, "final")
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    image_base = os.path.splitext(os.path.basename(image_file))[0]

    # -------------------------------
    # 1. Transcribe Audio or Parse TXT
    # -------------------------------
    transcription_txt_path  = os.path.join(working_dir, "transcription.txt")
    transcription_json_path = transcription_txt_path + ".json"

    if os.path.exists(transcription_json_path):
        print("✅ JSON already exists. Loading…")
        with open(transcription_json_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

    elif os.path.exists(transcription_txt_path):
        print("⚡ TXT exists but JSON missing—parsing TXT to JSON…")
        transcript = txt_to_json(transcription_txt_path, transcription_json_path)

    else:
        print("🛠 No transcript found.  Running Whisper…")
        transcript = audio_to_text(audio_file, working_dir)

    # -------------------------------
    # 2. Extract Homily and Clip Audio
    # -------------------------------
    homily_file = os.path.join(working_dir, "homily.mp3")

    try:
        start, end, text, segments = find_homily(transcript)
    except (UnboundLocalError, RuntimeError):
        print("🚫 Could not locate the homily section in your transcript.")
        print("   • You can now manually enter the start and end times.")

        def parse_time_input(t):
            """Convert 'mm:ss' or 'ss' to float seconds."""
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

        from pydub.utils import mediainfo
        if manual_end:
            actual_end = manual_end
        else:
            try:
                if isinstance(transcript, list) and "end" in transcript[-1]:
                    actual_end = float(transcript[-1]["end"])
                elif isinstance(transcript, dict) and "segments" in transcript:
                    actual_end = float(transcript["segments"][-1]["end"])
                else:
                    actual_end = float(mediainfo(audio_file)["duration"])
            except:
                actual_end = float(mediainfo(audio_file)["duration"])

        if isinstance(transcript, dict) and "segments" in transcript:
            all_segments = transcript["segments"]
        elif isinstance(transcript, list):
            all_segments = transcript
        else:
            all_segments = []

        segments = [seg for seg in all_segments if float(seg.get("start", 0)) >= manual_start and float(seg.get("end", 0)) <= actual_end]
        text = " ".join(seg.get("text", "") for seg in segments)

        homily_json_path = os.path.join(working_dir, "homily.json")
        with open(homily_json_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments, "text": text}, f, indent=2)

        start, end = manual_start, actual_end

    if os.path.exists(homily_file):
        print("✅ Homily audio already exists; skipping clipping.")
    else:
        print(f"✂️  Clipping homily (from {start}s to {end}s) → {homily_file}")
        clip_audio_segment(audio_file, start, end, homily_file)

    # -------------------------------
    # 3. Generate Video Script JSON
    # -------------------------------
    video_script_path = os.path.join(working_dir, "video_script.json")
    if os.path.exists(video_script_path):
        print("Video script already exists; skipping generation.")
    else:
        print("Generating video script...")
        video_script = generate_video_script(transcript, start, end)
        with open(video_script_path, "w", encoding="utf-8") as f:
            json.dump(video_script, f, indent=4)
        print("Video script saved.")

    # -------------------------------
    # 4. Clean Audio
    # -------------------------------
    homily_file_clean = os.path.join(working_dir, "homily_clean.mp3")
    if os.path.exists(homily_file_clean):
        print("Cleaned homily audio already exists; skipping Auphonic step.")
    else:
        print("Starting Auphonic production…")
        uuid = start_production(homily_file)
        if not uuid:
            print("Production failed to start.")
            return
        clean_path = download_file(uuid, working_dir)
        os.rename(clean_path, homily_file_clean)
        print(f"Downloaded cleaned homily to {homily_file_clean}")

    homily_file_final = os.path.join(working_dir, "homily_final.mp3")
    if os.path.exists(homily_file_final):
        print("Final homily audio already exists; skipping clipping.")
    else:
        print("Trimming cleaned audio…")
        clip_audio_segment(
            input_file=homily_file_clean,
            start_sec=6.4,
            end_sec=-6.4,
            output_file=homily_file_final,
            add_silence_sec=1.0
        )
        print(f"Final homily audio saved as {homily_file_final}")

    # -------------------------------
    # 5. Generate Video
    # -------------------------------
    final_video_path = os.path.join(final_output_dir, f"{image_base}.mp4")
    if os.path.exists(final_video_path):
        print("Final video already exists; skipping generation.")
    else:
        video_with_text_path = os.path.join(working_dir, f"{image_base}.mp4")
        if not os.path.exists(video_with_text_path):
            print("Generating video-with-text...")
            # image_file is just a placeholder for naming; your create_text_video uses black bg now
            create_text_video(video_script_path, image_file, homily_file_final, final_video_path=video_with_text_path)

        intro_video_path = os.path.join(os.path.dirname(__file__), "mylatinmass-intro-fixed.mp4")
        if not os.path.exists(intro_video_path):
            print(f"Intro video not found at {intro_video_path}. Cannot concatenate.")
            return
        concat_list_path = os.path.join(working_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            f.write(f"file '{intro_video_path}'\n")
            f.write(f"file '{video_with_text_path}'\n")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_list_path, "-c", "copy", final_video_path
        ], check=True)
        print(f"Final video saved as {final_video_path}")

    # -------------------------------
    # 6. Generate Captions
    # -------------------------------
    srt_file = os.path.join(working_dir, "video_captions.srt")
    if os.path.exists(srt_file):
        print("SRT file already exists; skipping.")
    else:
        print("Generating SRT file…")
        generate_srt_file(segments, srt_file)


    # -------------------------------
    # 7. Generate MDX
    # -------------------------------
    temp_mdx_path = os.path.join(working_dir, "homily_temp.mdx")

    print("Generating MDX file...")
    mdx_generator_script = os.path.abspath('./mdx_generator.py')
    result = subprocess.run(
        ["python", mdx_generator_script, transcription_json_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"mdx_generator.py failed with return code {result.returncode}")

    lines_to_ignore = ("Found phrase", "The main part", "Phrase not found", "Homily text:")
    cleaned_output = []
    capture = False
    for line in result.stdout.splitlines():
        if not capture and line.strip().startswith('---'):
            capture = True
        if capture and not line.startswith(lines_to_ignore):
            cleaned_output.append(line)

    mdx_text = "\n".join(cleaned_output).strip()
    if not mdx_text:
        raise RuntimeError("MDX generation failed: empty output")

    # Write temp (atomic-ish), then decide final path from front-matter
    with open(temp_mdx_path, "w", encoding="utf-8") as mdx_file:
        mdx_file.write(mdx_text)

    front = parse_front_matter_block(mdx_text)
    final_mdx_path = resolve_final_mdx_path_from_front(front, final_output_dir)

    if os.path.exists(final_mdx_path):
        print(f"Final MDX already exists at: {final_mdx_path}; skipping write.")
    else:
        # atomic move to final path directory
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(final_mdx_path), suffix=".mdx", encoding="utf-8") as tf:
            tf.write(mdx_text)
            tmp_final = tf.name
        os.replace(tmp_final, final_mdx_path)

    print(f"✅ Clean MDX file generated at: {final_mdx_path}")


    # -------------------------------
    # 8. Load Metadata
    # -------------------------------
    post = frontmatter.load(final_mdx_path)
    metadata = post.metadata
    print("Metadata:", metadata)

    required_fields = ["title", "youtube_description", "keywords"]
    missing_fields = [field for field in required_fields if not metadata.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required MDX metadata fields: {', '.join(missing_fields)}")


    # -------------------------------
    # 9. Upload to YouTube
    # -------------------------------
    google_user_id = "102136376185174842894"
    try:
        tokens = get_and_refresh_google_user_tokens(google_user_id)

        # Pull Education settings from MDX if present
        yt_category = metadata.get("youtube_category_id", "27")  # default to Education
        yt_edu_type = metadata.get("youtube_edu_type")
        yt_edu_problems = metadata.get("youtube_edu_problems")  # expect list if present

        # Upload video. The helper handles:
        # - thumbnail search (ANY image in final folder)
        # - privacy (public if thumbnail exists else private)
        # - thumbnail set if found
        video_id = youtube_upload_video_with_optional_thumbnail(
            tokens=tokens,
            file_path=final_video_path,
            title=metadata["title"],
            description=metadata["youtube_description"],
            tags=[t.strip() for t in metadata["keywords"].split(",") if t.strip()],
            categoryId=str(yt_category),
            edu_type=yt_edu_type,
            edu_problems=yt_edu_problems,
        )

        if not video_id:
            raise RuntimeError("YouTube upload failed: no video_id returned.")

        print("✅ Video uploaded successfully. Video ID:", video_id)

        # Optional: update title/description (no thumbnail here; helper already handled it)
        update_result = youtube_update_video(
            tokens=tokens,
            video_id=video_id,
            new_description=metadata["youtube_description"],
            new_title=metadata["title"],
            new_thumbnail_path=None
        )
        print("Update Response:", update_result)

        # Upload captions
        youtube_upload_captions(tokens, video_id=video_id, caption_file_path=srt_file)

        # Save YouTube ID back into MDX
        post.metadata["media_path"] = video_id
        final_mdx_content = frontmatter.dumps(post)
        with open(final_mdx_path, "w", encoding="utf-8") as f:
            f.write(final_mdx_content)
        print(f"✅ Final MDX file updated with YouTube ID at: {final_mdx_path}")

        # Clean up temp
        if os.path.exists(temp_mdx_path):
            os.remove(temp_mdx_path)

        # Push to website repo (optional)
        website_repo = os.getenv("WEBSITE_REPO_PATH")
        if website_repo:
            dest_rel = post.metadata.get("mdx_file")
            if dest_rel:
                dest_path = os.path.join(website_repo, dest_rel)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(final_mdx_content)
                try:
                    subprocess.run(["git", "-C", website_repo, "add", dest_rel], check=True)
                    subprocess.run(["git", "-C", website_repo, "commit", "-m", f"Add {post.metadata['title']}"], check=True)
                    subprocess.run(["git", "-C", website_repo, "push"], check=True)
                    print("✅ Website content pushed successfully.")
                except subprocess.CalledProcessError as e:
                    print("⚠️ Warning: failed to push website content:", e)
            else:
                print("⚠️ mdx_file not specified in metadata; skipping website push.")
        else:
            print("ℹ️ WEBSITE_REPO_PATH not set; skipping website push.")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
