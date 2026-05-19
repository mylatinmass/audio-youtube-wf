import argparse
from pathlib import Path
from typing import Optional

from shorts_fxn.step_01_load_homily import (
    load_homily_from_folder,
    print_loaded_homily_summary,
)

from shorts_fxn.step_02_identify_shorts import (
    identify_usable_shorts_from_folder,
)

from shorts_fxn.step_03_parse_selection import (
    load_json,
    prompt_user_for_selection,
    select_clips_from_analysis_file,
)

from shorts_fxn.step_04_render_with_images import (
    run_step_04_render_with_images,
)


def resolve_analysis_path(homily_result: dict) -> Path:
    paths = homily_result["paths"]

    analysis_path = paths.get("shorts_analysis")

    if not analysis_path:
        raise FileNotFoundError("Could not resolve shorts_analysis.json path.")

    return Path(analysis_path).expanduser().resolve()


def resolve_audio_path(homily_result: dict, audio_arg: Optional[str] = None) -> Path:
    if audio_arg:
        audio_path = Path(audio_arg).expanduser().resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        return audio_path

    paths = homily_result["paths"]
    audio_path = paths.get("homily_audio")

    if not audio_path:
        raise FileNotFoundError(
            "No homily audio found. Pass it manually with --audio '/path/to/homily_final.mp3'"
        )

    audio_path = Path(audio_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    return audio_path


def run_step_1_load(args: argparse.Namespace) -> dict:
    print()
    print("STEP 1: Loading homily")
    print("=" * 80)

    result = load_homily_from_folder(
        homily_folder=args.homily_folder,
        audio_path=args.audio,
        create_folders=True,
    )

    print_loaded_homily_summary(result)

    return result


def run_step_2_analyze(args: argparse.Namespace) -> dict:
    print()
    print("STEP 2: Identifying usable Shorts")
    print("=" * 80)

    analysis = identify_usable_shorts_from_folder(
        homily_folder=args.homily_folder,
        min_clips=args.min_clips,
        max_clips=args.max_clips,
        model=args.model,
        force=args.force_analysis,
    )

    return analysis


def run_step_3_select(args: argparse.Namespace, analysis_path: Path) -> dict:
    print()
    print("STEP 3: Selecting Shorts")
    print("=" * 80)

    analysis = load_json(analysis_path)

    selection = args.select

    if not selection:
        selection = prompt_user_for_selection(analysis)

    updated_analysis = select_clips_from_analysis_file(
        shorts_analysis_path=analysis_path,
        selection=selection,
    )

    return updated_analysis


def run_step_4_render(
    args: argparse.Namespace,
    analysis_path: Path,
    audio_path: Path,
) -> list:
    print()
    print("STEP 4: Rendering selected Shorts")
    print("=" * 80)

    rendered_clips = run_step_04_render_with_images(
        shorts_analysis_path=analysis_path,
        source_audio=audio_path,
        bg_audio_dir=args.bg_audio_dir,
        force_images=args.force_images,
        force_render=args.force_render,
        allow_ai_fallback=not args.no_ai_fallback,
        max_workers=args.max_workers,
    )

    return rendered_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Shorts workflow: load homily, identify clips, select IDs, and render videos."
    )

    parser.add_argument(
        "homily_folder",
        nargs="?",
        default=None,
        help="Homily folder or direct homily JSON path.",
    )

    parser.add_argument(
        "--audio",
        default=None,
        help="Optional direct path to homily audio file.",
    )

    parser.add_argument(
        "--select",
        default=None,
        help='Clip IDs to render, example: "1, 2-5, 7, 9-12". If omitted, you will be prompted.',
    )

    parser.add_argument(
        "--min-clips",
        type=int,
        default=4,
        help="Minimum number of Shorts to try identifying.",
    )

    parser.add_argument(
        "--max-clips",
        type=int,
        default=12,
        help="Maximum number of Shorts to identify.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override for Step #2.",
    )

    parser.add_argument(
        "--force-analysis",
        action="store_true",
        help="Force Step #2 to regenerate shorts_analysis.json.",
    )

    parser.add_argument(
        "--force-images",
        action="store_true",
        help="Force Step #4 to search/generate images again.",
    )

    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Force Step #4 to render videos again.",
    )

    parser.add_argument(
        "--no-ai-fallback",
        action="store_true",
        help="Do not generate AI images if public-domain art is not found.",
    )

    parser.add_argument(
        "--bg-audio-dir",
        default=None,
        help="Optional folder containing background music files.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Parallel image lookup/generation workers for Step #4.",
    )

    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Run only a specific step or the full workflow.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not args.homily_folder:
        args.homily_folder = input("Enter path to homily folder or homily JSON: ").strip().strip('"').strip("'")

    if not args.homily_folder:
        raise ValueError("A homily folder or homily JSON path is required.")

    if args.min_clips < 1:
        raise ValueError("--min-clips must be at least 1.")

    if args.max_clips < args.min_clips:
        raise ValueError("--max-clips must be greater than or equal to --min-clips.")

    homily_result = None
    analysis_path = None
    audio_path = None
    
    if args.step in {"1", "all"}:
        homily_result = run_step_1_load(args)

        if args.step == "1":
            return

    if homily_result is None:
        homily_result = load_homily_from_folder(
            homily_folder=args.homily_folder,
            audio_path=args.audio,
            create_folders=True,
        )

    analysis_path = resolve_analysis_path(homily_result)

    if args.step in {"2", "all"}:
        run_step_2_analyze(args)

        if args.step == "2":
            return

    if args.step in {"3", "all"}:
        if not analysis_path.exists():
            raise FileNotFoundError(
                f"shorts_analysis.json does not exist yet: {analysis_path}. Run Step #2 first."
            )

        run_step_3_select(args, analysis_path)

        if args.step == "3":
            return

    if args.step in {"4", "all"}:
        if not analysis_path.exists():
            raise FileNotFoundError(
                f"shorts_analysis.json does not exist yet: {analysis_path}. Run Step #2 first."
            )

        audio_path = resolve_audio_path(homily_result, args.audio)

        run_step_4_render(
            args=args,
            analysis_path=analysis_path,
            audio_path=audio_path,
        )

        return


if __name__ == "__main__":
    main()