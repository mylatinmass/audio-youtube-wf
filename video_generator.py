#!/opt/anaconda3/envs/dlindo/bin/python
import json
import os
import numpy as np
from moviepy import *
from PIL import Image, ImageDraw, ImageFont
import subprocess  # for potential FFmpeg operations (used later in workflow)

os.environ['IMAGEMAGICK_BINARY'] = '/usr/local/bin/magick'

# Update your file paths as needed
json_path = "/Users/mainmarketing/Desktop/March 30th homily/working/transcription.txt.json"
background_image_path = "/Users/mainmarketing/Downloads/DALL·E Sower Illustration Feb 24.webp"
# Final audio clip: this should be the homily_clean.mp3 in your working folder.
audio_file_path = "/Users/mainmarketing/Desktop/March 30th homily/working/homily_clean.mp3"
video_size = (video_width, video_height) = (2560, 1440)

def clean_path(path):
    return path.strip().strip("'").strip('"')

def wrap_text(text, font, max_width):
    lines, words, line = [], text.split(), ""
    for word in words:
        test_line = line + " " + word if line else word
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def generate_subtitle_overlay_clips(segments, video_size):
    video_width, video_height = video_size
    total_duration = segments[-1][1]
    reserved = 2  # seconds reserved for the outro
    font_path = "Arial Bold.ttf"
    font_size = int(video_width * 0.0375)
    font = ImageFont.truetype(font_path, font_size)
    text_box_width = int(video_width * 0.6)
    left_margin = int(video_width * 0.2)
    subtitle_clips = []
    num_segments = len(segments)
    
    for i, (start, end, sentence) in enumerate(segments):
        if i == 0:
            actual_start = 0
            if num_segments > 1:
                duration = segments[1][0] - 0
            else:
                duration = total_duration - reserved
        else:
            actual_start = start
            if i < num_segments - 1:
                duration = segments[i+1][0] - start
            else:
                duration = max(end - start - reserved, 0)
        
        bg_img = Image.open(background_image_path).convert("RGBA")
        bg_img = bg_img.resize(video_size, Image.Resampling.LANCZOS)
        overlay = Image.new("RGBA", video_size, (0, 0, 0, 150))
        img = Image.alpha_composite(bg_img, overlay)
        
        wrapped_text = wrap_text(sentence, font, text_box_width)
        line_height = (font.getbbox("A")[3] - font.getbbox("A")[1]) + int(font_size * 0.5)
        text_block_height = len(wrapped_text) * line_height
        y_start = (video_height - text_block_height) // 2
        
        draw = ImageDraw.Draw(img)
        for j, line in enumerate(wrapped_text):
            bbox_line = font.getbbox(line)
            text_width = bbox_line[2] - bbox_line[0]
            x_position = left_margin + (text_box_width - text_width) // 2
            draw.text((x_position, y_start + j * line_height), line, font=font, fill="white")
        
        clip = ImageClip(np.array(img), duration=duration).with_start(actual_start)
        subtitle_clips.append(clip)
    
    # Append the outro clip.
    outro_frames = generate_subtitle_background_image_outro(video_size, duration=reserved, fps=24)
    outro_clip = ImageSequenceClip([np.array(frame) for frame in outro_frames], fps=24)
    outro_clip = outro_clip.with_start(total_duration - reserved)
    subtitle_clips.append(outro_clip)
    
    return subtitle_clips

def generate_banner_text_clip(video_size, duration):
    video_width, video_height = video_size
    banner_width, banner_height = int(video_width * 0.5), int(video_height * 0.12)
    
    def make_frame(t):
        if t < 4:
            return np.array(Image.new("RGBA", video_size, (0, 0, 0, 0)))
        t_adj = t - 4
        progress = min(max(t_adj / 1, 0), 1)
        first_color = (255, int(progress * 255), int(progress * 255))
        middle_color = (255, int(progress * 255), 0)
        last_color = (255, int(progress * 255), int(progress * 255))
        text_canvas = Image.new("RGBA", (banner_width, banner_height), (0, 0, 0, 0))
        draw_text = ImageDraw.Draw(text_canvas)
        banner_font = ImageFont.truetype("Times New Roman.ttf", int(banner_height * 0.7))
        first_part = "MY"
        middle_part = "LATIN"
        last_part = "MASS.COM"
        start_x = int(banner_width * 0.035)
        start_y = int(banner_height * 0.075)
        draw_text.text((start_x, start_y), first_part, fill=first_color, font=banner_font)
        first_bbox = banner_font.getbbox(first_part)
        first_width = first_bbox[2] - first_bbox[0]
        draw_text.text((start_x + first_width, start_y), middle_part, fill=middle_color, font=banner_font)
        middle_bbox = banner_font.getbbox(middle_part)
        middle_width = middle_bbox[2] - middle_bbox[0]
        draw_text.text((start_x + first_width + middle_width, start_y), last_part, fill=last_color, font=banner_font)
        canvas = Image.new("RGBA", video_size, (0, 0, 0, 0))
        canvas.paste(text_canvas, (0, 0), text_canvas)
        return np.array(canvas)
    
    from moviepy import VideoClip
    text_clip = VideoClip(make_frame, duration=duration)
    return text_clip

def generate_banner_background_clip(video_size, duration):
    video_width, video_height = video_size
    banner_width, banner_height = int(video_width * 0.5), int(video_height * 0.12)
    banner_img = Image.new("RGBA", (banner_width, banner_height), (255, 0, 0, 255))
    canvas = Image.new("RGBA", video_size, (0, 0, 0, 0))
    canvas.paste(banner_img, (0, 0), banner_img)
    background_clip = ImageClip(np.array(canvas), duration=duration).with_start(3).with_position(
        lambda t: (-banner_width + (banner_width * t / 0.75) if t < 0.75 else 0, 0))
    return background_clip

def generate_subtitle_background_image_intro(video_size, duration, fps=24):
    bg_img = Image.open(background_image_path).convert("RGBA")
    bg_img = bg_img.resize(video_size, Image.Resampling.LANCZOS)
    num_frames = int(duration * fps)
    frames = []
    for i in range(num_frames):
        current_time = i / fps
        if current_time <= 1:
            current_alpha = int(255 - (105 * (current_time / 1)))
        else:
            current_alpha = 150
        overlay = Image.new("RGBA", video_size, (0, 0, 0, current_alpha))
        frame = Image.alpha_composite(bg_img, overlay)
        frames.append(frame)
    return frames

def generate_subtitle_background_image_outro(video_size, duration=1, fps=24):
    bg_img = Image.open(background_image_path).convert("RGBA")
    bg_img = bg_img.resize(video_size, Image.Resampling.LANCZOS)
    num_frames = int(duration * fps)
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 1
        current_alpha = int(150 + (255 - 150) * t)
        overlay = Image.new("RGBA", video_size, (0, 0, 0, current_alpha))
        frame = Image.alpha_composite(bg_img, overlay)
        frames.append(frame)
    return frames

def parse_json_transcript(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    segments = [(seg["start"], seg["end"], seg["text"]) for seg in data["segments"]]
    return segments

def filter_homily_segments(transcript_json, homily_start, homily_end):
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered_segments = [
        seg for seg in data["segments"]
        if seg["start"] >= homily_start and seg["end"] <= homily_end
    ]
    data["segments"] = filtered_segments
    filtered_json_path = transcript_json.replace(".json", "_homily.json")
    with open(filtered_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return filtered_json_path

def create_text_video(json_path, bg_img_path, audio_path, final_video_path=None, video_size=video_size):
    global background_image_path, audio_file_path
    background_image_path = bg_img_path
    audio_file_path = audio_path

    try:
        print("Starting video creation...")
        print("Parsing transcript...")
        segments = parse_json_transcript(json_path)
        total_duration = segments[-1][1]
        print(f"Transcript parsed successfully. Total segments: {len(segments)}, Total duration: {total_duration} seconds.")
        
        print("Generating banner text clip...")
        banner_text_clip = generate_banner_text_clip(video_size, total_duration)
        print("Generating subtitle overlay clips...")
        subtitle_clips = generate_subtitle_overlay_clips(segments, video_size)
        print("Generating banner background clip...")
        banner_background_clip = generate_banner_background_clip(video_size, total_duration)
        
        print("Compositing main video clip...")
        main_composite = CompositeVideoClip(
            subtitle_clips + [banner_background_clip, banner_text_clip],
            size=video_size
        )
        
        print("Generating intro frames...")
        intro_frames = generate_subtitle_background_image_intro(video_size, duration=1, fps=24)
        intro_clip = ImageSequenceClip([np.array(frame) for frame in intro_frames], fps=24)
        
        print("Concatenating intro and main composite clips...")
        final_video = concatenate_videoclips([intro_clip, main_composite])
        
        print("Outputting Full Video (video-with-text)...")
        # Save the video-with-text (without concatenating an intro video) into the working folder.
        output_path = final_video_path if final_video_path else os.path.join(os.path.dirname(json_path), "video-with-text.mp4")
        print("Loading and integrating audio clip...")
        silence = AudioClip(lambda t: 0, duration=1, fps=44100).with_duration(1)
        audio_clip = AudioFileClip(audio_file_path)
        delayed_audio = concatenate_audioclips([silence, audio_clip])
        final_video = final_video.with_audio(delayed_audio)
        
        print("Writing video-with-text file... This may take a while.")
        final_video.write_videofile(output_path, fps=24, codec="libx264", logger="bar", audio_codec="aac", audio_bitrate="192k")
        print(f"Video-with-text saved as {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Call create_text_video without concatenating the intro video.
    create_text_video(
        clean_path(json_path),
        clean_path(background_image_path),
        clean_path(audio_file_path)
    )
