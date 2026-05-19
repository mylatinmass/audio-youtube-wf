import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_selection(selection: str, max_id: int) -> Set[int]:
    """
    Converts user input like:

    1, 2-5, 7, 9-12

    into:

    {1, 2, 3, 4, 5, 7, 9, 10, 11, 12}
    """

    selection = str(selection or "").strip()

    if not selection:
        return set()

    selected_ids: Set[int] = set()

    parts = [
        part.strip()
        for part in selection.split(",")
        if part.strip()
    ]

    for part in parts:
        # Range like 2-5
        if "-" in part:
            match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)

            if not match:
                raise ValueError(f"Invalid range selection: {part}")

            start = int(match.group(1))
            end = int(match.group(2))

            if end < start:
                start, end = end, start

            for number in range(start, end + 1):
                if 1 <= number <= max_id:
                    selected_ids.add(number)

            continue

        # Single ID like 7
        if not part.isdigit():
            raise ValueError(f"Invalid selection item: {part}")

        number = int(part)

        if 1 <= number <= max_id:
            selected_ids.add(number)

    return selected_ids


def get_max_clip_id(analysis: Dict[str, Any]) -> int:
    clips = analysis.get("clips", [])

    if not clips:
        return 0

    return max(int(clip.get("id", 0)) for clip in clips)


def update_selected_clips(
    analysis: Dict[str, Any],
    selected_ids: Set[int],
) -> Dict[str, Any]:
    """
    Updates the selected field for each clip.

    Selected IDs become:
    selected: true

    All others become:
    selected: false
    """

    clips = analysis.get("clips", [])

    for clip in clips:
        clip_id = int(clip.get("id", 0))
        clip["selected"] = clip_id in selected_ids

    analysis["selected_ids"] = sorted(selected_ids)
    analysis["selected_count"] = len(selected_ids)

    return analysis


def print_selection_summary(analysis: Dict[str, Any]) -> None:
    clips = analysis.get("clips", [])

    selected = [clip for clip in clips if clip.get("selected")]
    not_selected = [clip for clip in clips if not clip.get("selected")]

    print()
    print("Shorts Selection Updated")
    print("-" * 80)

    if selected:
        print("Selected:")
        for clip in selected:
            print(
                f"{clip['id']}. "
                f"{clip.get('time', '')} "
                f"({clip.get('length', '')}) - "
                f"{clip.get('title', '')}"
            )
    else:
        print("No clips selected.")

    print()
    print(f"Selected clips:     {len(selected)}")
    print(f"Not selected clips: {len(not_selected)}")
    print("-" * 80)
    print()


def select_clips_from_analysis_file(
    shorts_analysis_path: str | Path,
    selection: str,
) -> Dict[str, Any]:
    """
    Main Step #3 function.

    Input:
    - path to shorts_analysis.json
    - user selection string like "1, 2-5, 7"

    Output:
    - updates shorts_analysis.json
    - returns updated analysis
    """

    shorts_analysis_path = Path(shorts_analysis_path).expanduser().resolve()

    if not shorts_analysis_path.exists():
        raise FileNotFoundError(f"shorts_analysis.json not found: {shorts_analysis_path}")

    analysis = load_json(shorts_analysis_path)

    max_id = get_max_clip_id(analysis)

    if max_id <= 0:
        raise ValueError("No clips found in shorts_analysis.json.")

    selected_ids = parse_selection(selection, max_id=max_id)

    analysis = update_selected_clips(
        analysis=analysis,
        selected_ids=selected_ids,
    )

    save_json(shorts_analysis_path, analysis)

    print_selection_summary(analysis)

    return analysis


def prompt_user_for_selection(analysis: Dict[str, Any]) -> str:
    clips = analysis.get("clips", [])

    print()
    print("Available Shorts")
    print("-" * 100)
    print(f"{'ID':<4} {'Time':<13} {'Length':<8} {'Strength':<10} Title")
    print("-" * 100)

    for clip in clips:
        print(
            f"{clip.get('id', ''):<4} "
            f"{clip.get('time', ''):<13} "
            f"{clip.get('length', ''):<8} "
            f"{clip.get('strength', ''):<10} "
            f"{clip.get('title', '')}"
        )

    print("-" * 100)
    print()
    print("Enter IDs like:")
    print("1, 2-5, 7, 9-12")
    print()

    return input("Select clips to generate: ").strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step #3: Select Shorts to generate.")
    parser.add_argument(
        "shorts_analysis",
        help="Path to shorts_analysis.json.",
    )
    parser.add_argument(
        "--select",
        default=None,
        help='Selection string, example: "1, 2-5, 7". If omitted, you will be prompted.',
    )

    args = parser.parse_args()

    analysis_path = Path(args.shorts_analysis).expanduser().resolve()

    if not analysis_path.exists():
        raise FileNotFoundError(f"shorts_analysis.json not found: {analysis_path}")

    analysis = load_json(analysis_path)

    selection = args.select

    if not selection:
        selection = prompt_user_for_selection(analysis)

    select_clips_from_analysis_file(
        shorts_analysis_path=analysis_path,
        selection=selection,
    )