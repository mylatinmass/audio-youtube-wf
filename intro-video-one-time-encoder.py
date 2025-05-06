#!/usr/bin/env python3
import subprocess
import os
import sys

def reencode_intro(input_path, output_path):
    """
    Re-encodes the intro video to match the desired properties:
      - Resolution: 2560x1440
      - Frame rate: 24 fps
      - Video codec: H.264 (libx264) with high profile, veryfast preset, CRF 18, yuv420p pixel format
      - Audio codec: AAC, 44100 Hz, stereo, 192k bitrate
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "scale=2560:1440,fps=24",  # set resolution and frame rate
        "-c:v", "libx264",
        "-profile:v", "high",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        output_path
    ]
    
    print("Running FFmpeg command:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Re-encoded intro video saved as: {output_path}")

def main():
    # Input intro video (adjust path if needed)
    input_path = "/Users/mainmarketing/Desktop/Homily Audio File to Video/mylatinmass-intro.mp4"
    # Output fixed intro video
    output_path = "mylatinmass-intro-fixed.mp4"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    reencode_intro(input_path, output_path)

if __name__ == "__main__":
    main()
