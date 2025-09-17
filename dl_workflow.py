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
from PIL import Image
import frontmatter
from dotenv import load_dotenv
from get_token import get_and_refresh_google_user_tokens, SCOPES
from youtube import youtube_list_videos, youtube_upload_video, youtube_update_video, youtube_upload_captions
from googleapiclient.errors import HttpError
from txt_to_json import txt_to_json

# Load environment variables (if using a .env file)
load_dotenv()

# Prompt user for audio file and image to complete workflow
def prompt_user():
    audio_file = clean_path(input("Enter AUDIO FILE PATH (e.g., /path/to/audio.mp3): ").strip())
    image_file = clean_path(input("Enter IMAGE FILE PATH (e.g., /path/to/image.jpg): ").strip())
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
    final_mdx_path = os.path.join(final_output_dir, "homily.mdx")

    if os.path.exists(final_mdx_path):
        print("Final MDX already exists; skipping generation.")
    else:
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
            if line.strip().startswith('---'):
                capture = True
            if capture and not line.startswith(lines_to_ignore):
                cleaned_output.append(line)

        with open(temp_mdx_path, "w", encoding="utf-8") as mdx_file:
            mdx_file.write("\n".join(cleaned_output))

        if not os.path.exists(temp_mdx_path) or os.path.getsize(temp_mdx_path) == 0:
            raise RuntimeError("MDX generation failed: file missing or empty")

        # copy to final instead of moving
        with open(temp_mdx_path, "r", encoding="utf-8") as src, open(final_mdx_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
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

        upload_response = youtube_upload_video(
            tokens,
            file_path=final_video_path,
            title=metadata["title"],
            description=metadata["youtube_description"],
            tags=metadata["keywords"].split(", "),
            categoryId="22",
            privacyStatus="public",
        )
        video_id = upload_response
        print("Video uploaded successfully. Video ID:", video_id)

        thumbnail_image_file = os.path.join(final_output_dir, "thumbnail.jpg")
        image = Image.open(image_file)
        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        image.save(thumbnail_image_file, "JPEG", quality=90)

        update_result = youtube_update_video(
            tokens,
            video_id=video_id,
            new_description=metadata["youtube_description"],
            new_title=metadata["title"],
            new_thumbnail_path=thumbnail_image_file
        )
        print("Update Response:", update_result)

        youtube_upload_captions(tokens, video_id=video_id, caption_file_path=srt_file)

        post.metadata["media_path"] = video_id
        final_mdx_content = frontmatter.dumps(post)
        with open(final_mdx_path, "w", encoding="utf-8") as f:
            f.write(final_mdx_content)
        print(f"Final MDX file updated with YouTube ID at: {final_mdx_path}")

        if os.path.exists(temp_mdx_path):
            os.remove(temp_mdx_path)

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
                    print("Website content pushed successfully.")
                except subprocess.CalledProcessError as e:
                    print("Warning: failed to push website content:", e)
            else:
                print("mdx_file not specified in metadata; skipping website push.")
        else:
            print("WEBSITE_REPO_PATH not set; skipping website push.")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()



# from audio_to_text import audio_to_text
# from text_find import find_homily, generate_srt_file
# from audio_clip import clip_audio_segment
# from auphonic_audio_cleaner import start_production, download_file
# from video_generator import create_text_video
# from video_script import generate_video_script
# import os
# import sys
# import json
# import subprocess
# from PIL import Image
# import frontmatter
# from dotenv import load_dotenv
# from get_token import get_and_refresh_google_user_tokens, SCOPES
# from youtube import youtube_list_videos, youtube_upload_video, youtube_update_video, youtube_upload_captions
# from googleapiclient.errors import HttpError
# from txt_to_json import txt_to_json

# # Load environment variables (if using a .env file)
# load_dotenv()

# # Prompt user for audio file and image to complete workflow
# def prompt_user():
#     audio_file = clean_path(input("Enter AUDIO FILE PATH (e.g., /path/to/audio.mp3): ").strip())
#     image_file = clean_path(input("Enter IMAGE FILE PATH (e.g., /path/to/image.jpg): ").strip())
#     return audio_file, image_file

# def clean_path(path):
#     """Removes leading/trailing spaces and handles unnecessary quotes."""
#     return path.strip().strip('"').strip("'")

# def main():
#     audio_file, image_file = prompt_user()
#     dir = os.path.dirname(audio_file)
#     working_dir = os.path.join(dir, "working")
#     final_output_dir = os.path.join(dir, "final")
#     os.makedirs(working_dir, exist_ok=True)
#     os.makedirs(final_output_dir, exist_ok=True)
#     image_base = os.path.splitext(os.path.basename(image_file))[0]

#     # -------------------------------
#     # 1. Transcribe Audio or Parse TXT
#     # -------------------------------
#     transcription_txt_path  = os.path.join(working_dir, "transcription.txt")
#     transcription_json_path = transcription_txt_path + ".json"

#     if os.path.exists(transcription_json_path):
#         print("✅ JSON already exists. Loading…")
#         with open(transcription_json_path, "r", encoding="utf-8") as f:
#             transcript = json.load(f)

#     elif os.path.exists(transcription_txt_path):
#         print("⚡ TXT exists but JSON missing—parsing TXT to JSON…")
#         transcript = txt_to_json(transcription_txt_path, transcription_json_path)

#     else:
#         print("🛠 No transcript found.  Running Whisper…")
#         transcript = audio_to_text(audio_file, working_dir)



#     # -------------------------------
#     # 2. Extract Homily and Clip Audio
#     # -------------------------------
#     homily_file = os.path.join(working_dir, "homily.mp3")

#     # 2a. Try to extract the timestamps & segments once
#     try:
#         start, end, text, segments = find_homily(transcript)
#     except (UnboundLocalError, RuntimeError):
#         print("🚫 Could not locate the homily section in your transcript.")
#         print("   • You can now manually enter the start and end times.")

#         def parse_time_input(t):
#             """Convert 'mm:ss' or 'ss' to float seconds."""
#             if ":" in t:
#                 mins, secs = t.split(":")
#                 return float(mins) * 60 + float(secs)
#             return float(t)

#         while True:
#             try:
#                 manual_start = float(parse_time_input(input("Enter START time (e.g., 123 or 2:03): ").strip()))
#                 break
#             except ValueError:
#                 print("❌ Invalid start time. Try again.")

#         while True:
#             manual_end_input = input("Enter END time (or press Enter for end of file): ").strip()
#             if manual_end_input == "":
#                 manual_end = None
#                 break
#             try:
#                 manual_end = float(parse_time_input(manual_end_input))
#                 break
#             except ValueError:
#                 print("❌ Invalid end time. Try again.")

#         # Determine end time
#         from pydub.utils import mediainfo
#         if manual_end:
#             actual_end = manual_end
#         else:
#             try:
#                 # Try transcript end time
#                 if isinstance(transcript, list) and "end" in transcript[-1]:
#                     actual_end = float(transcript[-1]["end"])
#                 elif isinstance(transcript, dict) and "segments" in transcript:
#                     actual_end = float(transcript["segments"][-1]["end"])
#                 else:
#                     actual_end = float(mediainfo(audio_file)["duration"])
#             except:
#                 actual_end = float(mediainfo(audio_file)["duration"])

#         # Build segments from transcript between manual times
#         if isinstance(transcript, dict) and "segments" in transcript:
#             all_segments = transcript["segments"]
#         elif isinstance(transcript, list):
#             all_segments = transcript
#         else:
#             all_segments = []

#         segments = [seg for seg in all_segments if float(seg.get("start", 0)) >= manual_start and float(seg.get("end", 0)) <= actual_end]

#         # Join text for MDX generation
#         text = " ".join(seg.get("text", "") for seg in segments)

#         # Save manual homily JSON so mdx_generator.py has correct data
#         homily_json_path = os.path.join(working_dir, "homily.json")
#         with open(homily_json_path, "w", encoding="utf-8") as f:
#             json.dump({"segments": segments, "text": text}, f, indent=2)

#         start, end = manual_start, actual_end



#     # 2b. Only run the (slow) clipping if we actually need to
#     if os.path.exists(homily_file):
#         print("✅ Homily audio already exists; skipping clipping.")
#     else:
#         print(f"✂️  Clipping homily (from {start}s to {end}s) → {homily_file}")
#         clip_audio_segment(audio_file, start, end, homily_file)


#     # -------------------------------
#     # 3. Generate Video Script JSON with trimmed segments
#     # -------------------------------
#     video_script_path = os.path.join(working_dir, "video_script.json")
#     if os.path.exists(video_script_path):
#         print("Video script already exists; skipping generation.")
#     else:
#         print("Generating video script...")
#         video_script = generate_video_script(transcript, start, end)
#         with open(video_script_path, "w", encoding="utf-8") as f:
#             json.dump(video_script, f, indent=4)
#         print("Video script saved.")


#     # -------------------------------
#     # 4a. Generate Clean Audio
#     # -------------------------------
#     homily_file_clean = os.path.join(working_dir, "homily_clean.mp3")
#     if os.path.exists(homily_file_clean):
#         print("Cleaned homily audio already exists; skipping Auphonic step.")
#     else:
#         print("Starting Auphonic production…")
#         uuid = start_production(homily_file)
#         if not uuid:
#             print("Production failed to start.")
#             return
#         clean_path = download_file(uuid, working_dir)
#         # ensure it lands at homily_file_clean
#         os.rename(clean_path, homily_file_clean)
#         print(f"Downloaded cleaned homily to {homily_file_clean}")

#     # -------------------------------
#     # 4b. Clip Final Audio
#     # -------------------------------
#     homily_file_final = os.path.join(working_dir, "homily_final.mp3")
#     if os.path.exists(homily_file_final):
#         print("Final homily audio already exists; skipping clipping.")
#     else:
#         print("Trimming cleaned audio (removing 6.45 s at start and end)…")
#         clip_audio_segment(
#             input_file=homily_file_clean,   # e.g. working/homily_clean.mp3
#             start_sec=6.4,                  # drop first ~6.4 s
#             end_sec=-6.4,                   # drop last ~6.4 s
#             output_file=homily_file_final,  # e.g. working/homily_final.mp3
#             add_silence_sec=1.0             # append 1 s of silence
#         )

#         print(f"Final homily audio saved as {homily_file_final}")


#     # -------------------------------
#     # 5. Generate Text Video
#     # -------------------------------
#     final_video_path = os.path.join(final_output_dir, f"{image_base}.mp4")
#     if os.path.exists(final_video_path):
#         print("Final video already exists; skipping video generation and concatenation.")
#     else:
#         # Step 5: Generate text video if necessary.
#         video_with_text_path = os.path.join(working_dir, f"{image_base}.mp4")
#         if os.path.exists(video_with_text_path):
#             print("Video-with-text already exists; skipping rendering.")
#         else:
#             print("Generating video-with-text...")
#             create_text_video(video_script_path, image_file, homily_file_final, final_video_path=video_with_text_path)

#         # Step 6: Concatenate the intro video with the generated video-with-text.
#         intro_video_path = os.path.join(os.path.dirname(__file__), "mylatinmass-intro-fixed.mp4")
#         if not os.path.exists(intro_video_path):
#             print(f"Intro video not found at {intro_video_path}. Cannot concatenate.")
#             return
#         print("Concatenating intro video with video-with-text using FFmpeg (concat demuxer)...")
#         concat_list_path = os.path.join(working_dir, "concat_list.txt")
#         with open(concat_list_path, "w", encoding="utf-8") as f:
#             f.write(f"file '{intro_video_path}'\n")
#             f.write(f"file '{video_with_text_path}'\n")
#         subprocess.run([
#             "ffmpeg",
#             "-f", "concat",
#             "-safe", "0",
#             "-i", concat_list_path,
#             "-c", "copy",
#             final_video_path
#         ], check=True)
#         print(f"Final concatenated video saved as {final_video_path}")

#     # -------------------------------
#     # 7. Generate SRT Captions file
#     # -------------------------------
#     srt_file = os.path.join(working_dir, "video_captions.srt")
#     if os.path.exists(srt_file):
#         print("SRT file already exists; skipping generation.")
#     else:
#         print("Generating SRT file...")
#         generate_srt_file(segments, srt_file)
        

#     # -------------------------------
#     # 8. Generate Temp MDX file
#     # -------------------------------
#     temp_mdx_path = os.path.join(working_dir, "homily_temp.mdx")
#     final_mdx_path = os.path.join(final_output_dir, "homily.mdx")

#     if os.path.exists(final_mdx_path):
#         print("Final MDX already exists; skipping generation.")
#     elif os.path.exists(temp_mdx_path):
#         print("Temporary MDX already exists; using existing file.")
#     else:
#         print("Generating MDX file...")
#         mdx_generator_script = os.path.abspath('./mdx_generator.py')
#         result = subprocess.run(
#             ["python", mdx_generator_script, transcription_json_path],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )

#         if result.returncode != 0:
#             print(result.stderr)
#             raise RuntimeError(f"mdx_generator.py failed with return code {result.returncode}")

#         lines_to_ignore = ("Found phrase", "The main part", "Phrase not found", "Homily text:")
#         cleaned_output = []
#         capture = False
#         for line in result.stdout.splitlines():
#             if line.strip().startswith('---'):
#                 capture = True
#             if capture and not line.startswith(lines_to_ignore):
#                 cleaned_output.append(line)

#         with open(temp_mdx_path, "w", encoding="utf-8") as mdx_file:
#             mdx_file.write('\n'.join(cleaned_output))

#         if not os.path.exists(temp_mdx_path) or os.path.getsize(temp_mdx_path) == 0:
#             raise RuntimeError(f"MDX generation failed: {temp_mdx_path} is missing or empty")

#         os.replace(temp_mdx_path, final_mdx_path)
#         print(f"✅ Clean MDX file successfully generated at: {final_mdx_path}")


#      # Load the temporary MDX file
#     post = frontmatter.load(temp_mdx_path)

#     # Access the frontmatter metadata (returns a dictionary)
#     metadata = post.metadata
#     print("Metadata:", metadata)

#     required_fields = ["title", "youtube_description", "keywords"]
#     missing_fields = [field for field in required_fields if not metadata.get(field)]
#     if missing_fields:
#         missing_str = ", ".join(missing_fields)
#         raise ValueError(f"Missing required MDX metadata fields: {missing_str}. Please fix the MDX file before uploading.")

#      # Ensure required keys exist before proceeding
#     required_keys = ["title", "youtube_description", "keywords"]
#     missing_keys = [k for k in required_keys if not metadata.get(k)]
#     if missing_keys:
#         raise KeyError(f"Missing required metadata in MDX: {', '.join(missing_keys)}")


#     # -------------------------------
#     # 9. Upload to YouTube
#     # -------------------------------
#     google_user_id = "102136376185174842894"
#     try:
#         tokens = get_and_refresh_google_user_tokens(google_user_id)

#         # 1. List Videos
#         # videos = youtube_list_videos(tokens)
#         # print("Uploaded videos:")
#         # for item in videos:
#         #     title = item["snippet"]["title"]
#         #     video_id = item["contentDetails"]["videoId"]
#         #     print(f"Title: {title} (ID: {video_id})")
        
#         # 2. Upload Video
#         # Uncomment and update the following lines to test video upload.
#         upload_response = youtube_upload_video(
#             tokens,
#             file_path=final_video_path,
#             title=metadata["title"],
#             description=metadata["youtube_description"],
#             tags=metadata["keywords"].split(", "),
#             categoryId="22",  # People & Blogs category
#             privacyStatus="public",
#         )
#         #print("Upload Response Video ID:", upload_response)
#         video_id = upload_response
#         print("Video uploaded successfully. Video ID:", video_id)

#         # 3. Update Video (metadata and/or thumbnail)
#         # Uncomment and update the following lines to test updating a video.
#         # open image file and reduce size
#         thumbnail_image_file = os.path.join(final_output_dir, "thumbnail.jpg")
#         image = Image.open(image_file)
#         image = image.resize((1280, 720), Image.Resampling.LANCZOS)
#         image.save(thumbnail_image_file, "JPEG", quality=90)
#         # Upload thumbnail
#         update_result = youtube_update_video(
#             tokens,
#             video_id=video_id,
#             new_description=metadata["youtube_description"],
#             new_title=metadata["title"],
#             new_thumbnail_path=thumbnail_image_file
            
#         )
#         print("Update Response:", update_result)

#         # 4. Upload Captions (Uncomment to test)
#         youtube_upload_captions(
#             tokens,
#             video_id=video_id,
#             caption_file_path=srt_file
#         )

#          # Finalize MDX with YouTube video ID
#         post.metadata["media_path"] = video_id
#         final_mdx_content = frontmatter.dumps(post)
#         with open(final_mdx_path, "w", encoding="utf-8") as f:
#             f.write(final_mdx_content)
#         print(f"Final MDX file saved at: {final_mdx_path}")
#         if os.path.exists(temp_mdx_path):
#             os.remove(temp_mdx_path)

#         # Push MDX to website repository if configured
#         website_repo = os.getenv("WEBSITE_REPO_PATH")
#         if website_repo:
#             dest_rel = post.metadata.get("mdx_file")
#             if dest_rel:
#                 dest_path = os.path.join(website_repo, dest_rel)
#                 os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#                 with open(dest_path, "w", encoding="utf-8") as f:
#                     f.write(final_mdx_content)
#                 try:
#                     subprocess.run(["git", "-C", website_repo, "add", dest_rel], check=True)
#                     subprocess.run(["git", "-C", website_repo, "commit", "-m", f"Add {post.metadata['title']}",], check=True)
#                     subprocess.run(["git", "-C", website_repo, "push"], check=True)
#                     print("Website content pushed successfully.")
#                 except subprocess.CalledProcessError as e:
#                     print("Warning: failed to push website content:", e)
#             else:
#                 print("mdx_file not specified in metadata; skipping website push.")
#         else:
#             print("WEBSITE_REPO_PATH not set; skipping website push.")
            
#     except Exception as e:
#         print("Error:", e)

# if __name__ == "__main__":
#     main()
