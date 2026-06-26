import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from shorts_fxn.step_01_load_homily import (
    clean_text,
    load_homily_from_folder,
)


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def segment_line(segment: Dict[str, Any]) -> str:
    start = format_time(segment["start"])
    end = format_time(segment["end"])
    text = clean_text(segment.get("text", ""))
    return f"[{start}-{end}] {text}"


def build_clip_source_text(result: Dict[str, Any]) -> str:
    paths = result["paths"]
    segments: List[Dict[str, Any]] = result["segments"]
    duration = float(segments[-1]["end"] - segments[0]["start"])

    lines = [
        "GPT SHORTS CLIP SOURCE",
        "",
        "Use this timed transcript to choose coherent YouTube Shorts from the homily.",
        "",
        "Return CSV only, with exactly these columns:",
        "Start,End,Title",
        "",
        "Rules:",
        "- Start and End must come from the transcript timestamps below.",
        "- Every clip must be between 15 and 90 seconds.",
        "- Do not create clips beyond the homily duration.",
        "- Prefer complete Catholic ideas, exhortations, stories, moral points, and theological points.",
        "- Titles should be clear, short, and useful as YouTube Shorts titles.",
        "- If enough material exists, provide at least 5 clips.",
        "- More clips are fine only when each section makes sense by itself.",
        "",
        f"Homily folder: {paths['root']}",
        f"Source JSON: {paths['homily_json']}",
        f"Homily duration: {format_time(duration)}",
        "",
        "TIMED TRANSCRIPT",
        "",
    ]

    lines.extend(segment_line(segment) for segment in segments)
    lines.append("")

    return "\n".join(lines)


def export_clip_source(homily_folder: str | Path, output_path: str | Path | None = None) -> Path:
    result = load_homily_from_folder(
        homily_folder=homily_folder,
        create_folders=True,
    )

    output_folder = Path(result["paths"]["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    if output_path:
        destination = Path(output_path).expanduser().resolve()
    else:
        destination = output_folder / "gpt_clip_source.txt"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_clip_source_text(result), encoding="utf-8")

    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "homily_root": str(result["paths"]["root"]),
                "source_json": str(result["paths"]["homily_json"]),
                "source_audio": str(result["paths"]["homily_audio"] or ""),
                "segments": len(result["segments"]),
                "duration": result["segments"][-1]["end"] - result["segments"][0]["start"],
                "text_export": str(destination),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a GPT-friendly timed transcript for manual Shorts clip selection."
    )
    parser.add_argument(
        "homily_folder",
        nargs="?",
        help="Homily folder or direct homily JSON path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .txt path. Defaults to 'Video Clips/gpt_clip_source.txt'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    homily_folder = args.homily_folder

    if not homily_folder:
        homily_folder = input("Enter path to homily folder or homily JSON: ").strip().strip('"').strip("'")

    if not homily_folder:
        raise ValueError("A homily folder or homily JSON path is required.")

    destination = export_clip_source(homily_folder, args.output)

    print(f"Exported GPT clip source: {destination}")
    print(f"Upload/provide this text file to GPT, then save its CSV answer as manual_clips.csv.")


if __name__ == "__main__":
    main()
