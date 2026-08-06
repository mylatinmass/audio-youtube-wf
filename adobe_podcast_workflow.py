#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv


TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
FRONT_RE = re.compile(r"^---\s*(.*?)\s*---", re.DOTALL | re.MULTILINE)
KV_RE = re.compile(r'^(?P<key>[A-Za-z0-9_]+):\s*"(?P<val>.*?)"\s*$', re.MULTILINE)


def clean_path(value: str) -> Path:
    return Path(value.strip().strip('"').strip("'")).expanduser().resolve()


def srt_time_to_seconds(value: str) -> float:
    hms, ms = value.split(",")
    hours, minutes, seconds = [int(part) for part in hms.split(":")]
    return hours * 3600 + minutes * 60 + seconds + int(ms) / 1000.0


def seconds_to_srt_time(value: float) -> str:
    value = max(0.0, float(value))
    total_ms = int(round(value * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    minutes_total, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def caption_text_for_mdx(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 ._'’-]{0,40}:", text):
            continue
        text = re.sub(r"^[A-Za-z][A-Za-z0-9 ._'’-]{0,40}:\s*", "", text).strip()
        if text:
            cleaned.append(text)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def parse_srt(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    segments: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        timing_index = next((i for i, line in enumerate(lines) if TIME_RE.search(line)), None)
        if timing_index is None:
            continue

        match = TIME_RE.search(lines[timing_index])
        if not match:
            continue

        text = caption_text_for_mdx(lines[timing_index + 1 :])
        if not text:
            continue

        segments.append(
            {
                "start": srt_time_to_seconds(match.group("start")),
                "end": srt_time_to_seconds(match.group("end")),
                "text": text,
            }
        )

    if not segments:
        raise RuntimeError(f"No caption segments found in SRT: {path}")

    return segments


def wrap_caption_text(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=42, break_long_words=False)) or text


def write_shifted_srt(input_path: Path, output_path: Path, shift_seconds: float) -> None:
    segments = parse_srt(input_path)
    lines: List[str] = []
    for index, segment in enumerate(segments, 1):
        start = seconds_to_srt_time(float(segment["start"]) + shift_seconds)
        end = seconds_to_srt_time(float(segment["end"]) + shift_seconds)
        lines.extend(
            [
                str(index),
                f"{start} --> {end}",
                wrap_caption_text(str(segment["text"])),
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def probe(path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=index,codec_type,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def first_stream(info: Dict[str, Any], codec_type: str) -> Dict[str, Any]:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    raise RuntimeError(f"Missing {codec_type} stream.")


def fps_to_float(value: str) -> float:
    if "/" not in value:
        return float(value)
    num, den = value.split("/", 1)
    den_f = float(den)
    return float(num) / den_f if den_f else 30.0


def render_video_with_intro(intro_path: Path, video_path: Path, output_path: Path) -> None:
    info = probe(video_path)
    video_stream = first_stream(info, "video")
    audio_stream = first_stream(info, "audio")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = fps_to_float(str(video_stream.get("r_frame_rate") or "30/1"))
    sample_rate = int(audio_stream.get("sample_rate") or 48000)

    vf_intro = f"scale={width}:{height}:flags=lanczos,fps={fps},format=yuv420p,setsar=1"
    vf_main = f"scale={width}:{height}:flags=lanczos,fps={fps},format=yuv420p,setsar=1"
    af = f"aformat=sample_rates={sample_rate}:channel_layouts=stereo"
    filter_complex = (
        f"[0:v]{vf_intro}[v0];"
        f"[1:v]{vf_main}[v1];"
        f"[0:a]{af}[a0];"
        f"[1:a]{af}[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(intro_path),
            "-i",
            str(video_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def intro_duration(path: Path) -> float:
    info = probe(path)
    return float(info.get("format", {}).get("duration") or 0.0)


def parse_front_matter(mdx_text: str) -> Dict[str, str]:
    match = FRONT_RE.search(mdx_text)
    if not match:
        return {}
    return {m.group("key"): m.group("val") for m in KV_RE.finditer(match.group(1))}


def resolve_mdx_output_path(front: Dict[str, str], output_dir: Path, fallback_stem: str) -> Path:
    mdx_file = str(front.get("mdx_file") or "").strip().lstrip("/\\")
    if mdx_file:
        return output_dir / mdx_file

    slug = str(front.get("slug") or fallback_stem).strip().lstrip("/\\") or fallback_stem
    if not slug.endswith(".mdx"):
        slug += ".mdx"
    return output_dir / slug


def generate_mdx(homily_json_path: Path, output_dir: Path, fallback_stem: str) -> Path:
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()

    from mdx_generator import generate_mdx_from_json

    mdx_text = generate_mdx_from_json(str(homily_json_path))
    front = parse_front_matter(mdx_text)
    mdx_path = resolve_mdx_output_path(front, output_dir, fallback_stem)
    mdx_path.parent.mkdir(parents=True, exist_ok=True)
    mdx_path.write_text(mdx_text.strip() + "\n", encoding="utf-8")
    return mdx_path


def default_output_dir(video_path: Path) -> Path:
    if video_path.parent.name == "working":
        return video_path.parent.parent / "final"
    return video_path.parent / "final"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an existing Adobe Podcast MP4 plus SRT to add the MyLatinMass intro and generate MDX."
    )
    parser.add_argument("--video", required=True, help="Existing Adobe Podcast MP4.")
    parser.add_argument("--srt", required=True, help="SRT transcript/captions for the existing MP4.")
    parser.add_argument("--thumbnail", required=True, help="Finished thumbnail image.")
    parser.add_argument("--output-dir", help="Defaults to ../final when video is in a working folder.")
    parser.add_argument(
        "--intro",
        default=str(Path(__file__).with_name("mylatinmass-intro-fixed.mp4")),
        help="Intro MP4 to prepend.",
    )
    parser.add_argument("--force-video", action="store_true", help="Re-render the intro-prefixed video if it exists.")
    parser.add_argument("--skip-mdx", action="store_true", help="Only create the video/captions/JSON assets.")
    args = parser.parse_args()

    video_path = clean_path(args.video)
    srt_path = clean_path(args.srt)
    thumbnail_path = clean_path(args.thumbnail)
    intro_path = clean_path(args.intro)
    output_dir = clean_path(args.output_dir) if args.output_dir else default_output_dir(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (video_path, srt_path, thumbnail_path, intro_path):
        if not path.exists():
            raise FileNotFoundError(path)

    segments = parse_srt(srt_path)
    homily_text = "\n\n".join(segment["text"] for segment in segments)
    stem = video_path.stem

    homily_json_path = output_dir / f"{stem}-homily.json"
    homily_json_path.write_text(
        json.dumps(
            {
                "homily_text": homily_text,
                "homily_segments": segments,
                "source_video": str(video_path),
                "source_srt": str(srt_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    intro_seconds = intro_duration(intro_path)
    shifted_srt_path = output_dir / f"{stem}-captions-with-intro.srt"
    write_shifted_srt(srt_path, shifted_srt_path, intro_seconds)

    final_video_path = output_dir / f"{stem}-with-intro.mp4"
    if args.force_video or not final_video_path.exists():
        render_video_with_intro(intro_path, video_path, final_video_path)

    final_thumbnail_path = output_dir / thumbnail_path.name
    if thumbnail_path.resolve() != final_thumbnail_path.resolve():
        shutil.copy2(thumbnail_path, final_thumbnail_path)

    mdx_path = None
    if not args.skip_mdx:
        mdx_path = generate_mdx(homily_json_path, output_dir, stem)

    print(json.dumps({
        "final_video": str(final_video_path),
        "shifted_srt": str(shifted_srt_path),
        "homily_json": str(homily_json_path),
        "thumbnail": str(final_thumbnail_path),
        "mdx": str(mdx_path) if mdx_path else "",
        "intro_seconds": intro_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
