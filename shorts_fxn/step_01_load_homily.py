import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_homily_json(homily_folder: str | Path) -> Path:
    """
    Step #1A:
    Find the homily JSON file from a folder.

    Accepted folder structures:

    homily-folder/
    ├── working/
    │   ├── video_script.json
    │   └── homily.json

    Or:

    homily-folder/
    ├── homily.json
    └── video_script.json

    You can also pass the direct JSON file path.
    """

    folder = Path(homily_folder).expanduser().resolve()

    if folder.is_file() and folder.suffix.lower() == ".json":
        return folder

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    preferred_paths = [
        folder / "working" / "video_script.json",
        folder / "working" / "homily.json",
        folder / "video_script.json",
        folder / "homily.json",
    ]

    for path in preferred_paths:
        if path.exists():
            return path

    # Fallback: search all JSON files and find one with timestamped segments.
    for path in folder.rglob("*.json"):
        try:
            data = load_json(path)
            if data.get("segments") or data.get("homily_segments"):
                return path
        except Exception:
            continue

    raise FileNotFoundError(
        f"No usable homily JSON found in: {folder}. "
        "Expected a JSON file with 'segments' or 'homily_segments'."
    )


def find_homily_audio(homily_folder: str | Path, audio_path: Optional[str | Path] = None) -> Optional[Path]:
    """
    Step #1B:
    Find the homily audio file.

    This is optional for Step #1 because Step #2 only needs the transcript.
    Step #4 will need the audio.
    """

    if audio_path:
        audio = Path(audio_path).expanduser().resolve()
        if not audio.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio}")
        return audio

    folder = Path(homily_folder).expanduser().resolve()

    if folder.is_file():
        folder = folder.parent

    preferred_paths = [
        folder / "working" / "homily_final.mp3",
        folder / "homily_final.mp3",
        folder / "working" / "homily.mp3",
        folder / "homily.mp3",
    ]

    for path in preferred_paths:
        if path.exists():
            return path

    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in audio_extensions:
            return path

    return None


def get_homily_root(homily_json_path: str | Path) -> Path:
    """
    If JSON is inside working/, root is the parent folder.

    Example:
    homily-folder/working/homily.json
    root = homily-folder
    """

    path = Path(homily_json_path).expanduser().resolve()

    if path.parent.name.lower() == "working":
        return path.parent.parent

    return path.parent


def get_output_folder(homily_json_path: str | Path) -> Path:
    """
    Output folder for analysis, images, video, captions, etc.
    """

    root = get_homily_root(homily_json_path)
    return root / "Video Clips"


def get_workflow_paths(
    homily_folder: str | Path,
    audio_path: Optional[str | Path] = None,
) -> Dict[str, Optional[Path]]:
    """
    Step #1C:
    Resolve the main paths for the workflow.
    """

    homily_json = find_homily_json(homily_folder)
    homily_audio = find_homily_audio(homily_folder, audio_path)
    root = get_homily_root(homily_json)
    output_folder = get_output_folder(homily_json)

    paths: Dict[str, Optional[Path]] = {
        "root": root,
        "homily_json": homily_json,
        "homily_audio": homily_audio,
        "output_folder": output_folder,
        "shorts_analysis": output_folder / "shorts_analysis.json",
        "shorts_manifest": output_folder / "shorts_manifest.json",
        "image_candidates": output_folder / "image_candidates.json",
        "upload_metadata": output_folder / "upload_metadata.json",
        "images_dir": output_folder / "images",
        "audio_dir": output_folder / "audio",
        "videos_dir": output_folder / "videos",
        "captions_dir": output_folder / "captions",
    }

    return paths


def ensure_output_folders(paths: Dict[str, Optional[Path]]) -> None:
    """
    Create the folders needed later in the workflow.
    """

    folders = [
        paths.get("output_folder"),
        paths.get("images_dir"),
        paths.get("audio_dir"),
        paths.get("videos_dir"),
        paths.get("captions_dir"),
    ]

    for folder in folders:
        if folder:
            folder.mkdir(parents=True, exist_ok=True)


def normalize_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Step #1D:
    Normalize transcript segments.

    Accepts:
    - homily_segments
    - segments

    Returns:
    [
      {
        "index": 1,
        "start": 0.0,
        "end": 4.78,
        "text": "Both the Epistle and Gospel..."
      }
    ]
    """

    raw_segments = data.get("homily_segments") or data.get("segments") or []

    if not raw_segments:
        raise ValueError("Homily JSON must contain 'homily_segments' or 'segments'.")

    segments: List[Dict[str, Any]] = []

    for i, segment in enumerate(raw_segments, start=1):
        text = clean_text(segment.get("text", ""))

        if not text:
            continue

        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
        except Exception:
            continue

        if end <= start:
            continue

        normalized = {
            "index": i,
            "start": start,
            "end": end,
            "text": text,
        }

        # Keep word timestamps if they exist.
        if segment.get("words"):
            normalized["words"] = segment.get("words")

        segments.append(normalized)

    if not segments:
        raise ValueError("No valid timestamped segments found.")

    return segments


def load_homily_from_folder(
    homily_folder: str | Path,
    audio_path: Optional[str | Path] = None,
    create_folders: bool = True,
) -> Dict[str, Any]:
    """
    Main Step #1 function.

    Input:
    - homily folder path
    - optional audio path

    Output:
    {
      "paths": {...},
      "homily_data": {...},
      "segments": [...]
    }
    """

    paths = get_workflow_paths(homily_folder, audio_path)

    if create_folders:
        ensure_output_folders(paths)

    homily_json = paths["homily_json"]

    if not homily_json:
        raise FileNotFoundError("Could not resolve homily JSON path.")

    homily_data = load_json(homily_json)
    segments = normalize_segments(homily_data)

    return {
        "paths": paths,
        "homily_data": homily_data,
        "segments": segments,
    }


def print_loaded_homily_summary(result: Dict[str, Any]) -> None:
    paths = result["paths"]
    segments = result["segments"]

    start = segments[0]["start"]
    end = segments[-1]["end"]
    duration = end - start

    print()
    print("Homily Loaded")
    print("-" * 60)
    print(f"Root:          {paths['root']}")
    print(f"JSON:          {paths['homily_json']}")
    print(f"Audio:         {paths['homily_audio'] or 'Not found yet'}")
    print(f"Output Folder: {paths['output_folder']}")
    print(f"Segments:      {len(segments)}")
    print(f"Duration:      {int(duration // 60)}:{int(duration % 60):02d}")
    print("-" * 60)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step #1: Load homily script from folder.")
    parser.add_argument("homily_folder", help="Homily folder or direct JSON path.")
    parser.add_argument("--audio", default=None, help="Optional direct path to audio file.")

    args = parser.parse_args()

    result = load_homily_from_folder(
        homily_folder=args.homily_folder,
        audio_path=args.audio,
        create_folders=True,
    )

    print_loaded_homily_summary(result)