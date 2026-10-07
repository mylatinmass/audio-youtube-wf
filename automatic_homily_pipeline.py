import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from shorts_youtube_upload import find_related_video_id_from_mdx


SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def clean_path(value: str) -> str:
    return str(value or "").strip().strip('"').strip("'")


def resolve_audio_path(value: Optional[str]) -> Path:
    if not value:
        value = input("Enter AUDIO FILE PATH: ")

    audio_path = Path(clean_path(value)).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio type {audio_path.suffix}. Expected one of: {supported}")
    return audio_path


def manifest_path(audio_path: Path) -> Path:
    return audio_path.parent / "working" / "automatic_pipeline.json"


def write_manifest(audio_path: Path, status: str, stage: str, **extra) -> None:
    path = manifest_path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_file": str(audio_path),
        "status": status,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def run_stage(name: str, command: List[str], audio_path: Path, dry_run: bool = False) -> None:
    print()
    print("=" * 88)
    print(name)
    print("=" * 88)
    print("Command:", " ".join(command))

    if dry_run:
        return

    write_manifest(audio_path, status="running", stage=name)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=SCRIPT_DIR, check=True, env=env)
    write_manifest(audio_path, status="running", stage=f"{name} complete")


def build_long_video_command(audio_path: Path) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "dl_workflow.py"),
        str(audio_path),
        "--non-interactive",
    ]


def build_shorts_command(args: argparse.Namespace, audio_path: Path) -> List[str]:
    root = audio_path.parent
    homily_audio = root / "working" / "homily_final.mp3"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "shorts_workflow.py"),
        str(root),
        "--audio",
        str(homily_audio),
        "--select-all",
        "--no-manual-clips-prompt",
        "--no-background-music",
        "--min-clips",
        str(args.min_clips),
        "--max-clips",
        str(args.max_clips),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.force_analysis:
        command.append("--force-analysis")
    if args.force_images:
        command.append("--force-images")
    if args.force_render:
        command.append("--force-render")
    return command


def build_upload_command(
    args: argparse.Namespace,
    clips_root: Path,
    related_video_id: str,
) -> List[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "shorts_youtube_upload.py"),
        str(clips_root),
        "--related-video-id",
        related_video_id,
        "--no-related-video-prompt",
        "--title-hashtags",
        str(args.title_hashtags),
    ]
    if args.google_user_id:
        command.extend(["--google-user-id", args.google_user_id])
    if args.schedule_start_date:
        command.extend(["--schedule-start-date", args.schedule_start_date])
    if args.no_volume_hashtags:
        command.append("--no-volume-hashtags")
    if args.refresh_hashtag_volume:
        command.append("--refresh-hashtag-volume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete homily production unattended after receiving one audio file: "
            "full video, thumbnail, YouTube upload, website MDX, Shorts, and scheduling."
        )
    )
    parser.add_argument("audio_file", nargs="?", help="Source homily audio. Prompts once when omitted.")
    parser.add_argument("--min-clips", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--model", help="Optional model override for Shorts analysis.")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--force-images", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--google-user-id", default=os.getenv("GOOGLE_USER_ID"))
    parser.add_argument("--schedule-start-date", help="First eligible Shorts date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--title-hashtags", type=int, default=2)
    parser.add_argument("--no-volume-hashtags", action="store_true")
    parser.add_argument("--refresh-hashtag-volume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print all stages without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_clips < 1:
        raise ValueError("--min-clips must be at least 1.")
    if args.max_clips < args.min_clips:
        raise ValueError("--max-clips must be greater than or equal to --min-clips.")
    if args.title_hashtags < 0:
        raise ValueError("--title-hashtags cannot be negative.")

    audio_path = resolve_audio_path(args.audio_file)
    root = audio_path.parent
    clips_root = root / "Video Clips"

    try:
        run_stage(
            "STAGE 1/3: Full homily production and publishing",
            build_long_video_command(audio_path),
            audio_path,
            dry_run=args.dry_run,
        )

        homily_audio = root / "working" / "homily_final.mp3"
        if not args.dry_run and not homily_audio.is_file():
            raise FileNotFoundError(f"Full workflow did not create the Shorts source audio: {homily_audio}")

        run_stage(
            "STAGE 2/3: Shorts discovery, selection, and rendering",
            build_shorts_command(args, audio_path),
            audio_path,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            related_video_id = "FULL_VIDEO_ID_FROM_MDX"
        else:
            related_video_id = find_related_video_id_from_mdx(str(clips_root))
            if not related_video_id:
                raise RuntimeError(
                    "The full homily YouTube ID was not found in the generated MDX. "
                    "Shorts were not uploaded because they could not be linked reliably."
                )

        run_stage(
            "STAGE 3/3: Scheduled Shorts upload",
            build_upload_command(args, clips_root, related_video_id),
            audio_path,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            write_manifest(
                audio_path,
                status="complete",
                stage="complete",
                full_video_id=related_video_id,
                clips_root=str(clips_root),
            )

        print()
        print("Automatic homily pipeline complete." if not args.dry_run else "Dry run complete.")
        print(f"Production folder: {root}")
        if not args.dry_run:
            print(f"Full homily: https://www.youtube.com/watch?v={related_video_id}")
            print("All pending Shorts were uploaded with scheduled public release times.")

    except Exception as exc:
        if not args.dry_run:
            write_manifest(audio_path, status="failed", stage="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
