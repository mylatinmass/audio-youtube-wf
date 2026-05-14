import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from googleapiclient.errors import HttpError

from get_token import get_and_refresh_google_user_tokens
from thumbnail_generator import is_valid_youtube_thumbnail
from youtube import youtube_update_video, youtube_upload_captions, youtube_upload_video


DEFAULT_GOOGLE_USER_ID = "102136376185174842894"
SHORTS_BASE_TAGS = [
    "Catholic",
    "Traditional Catholic",
    "Latin Mass",
    "Catholic homily",
    "YouTube Shorts",
    "Shorts",
]


def clean_path(path: str) -> str:
    return str(path or "").strip().strip('"').strip("'")


def dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_video_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(clean_path(path)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Video file does not exist: {resolved}")
    if os.path.splitext(resolved)[1].lower() != ".mp4":
        raise ValueError(f"Expected an MP4 video: {resolved}")
    return resolved


def clips_root_from_video(video_path: str) -> str:
    video_dir = os.path.dirname(video_path)
    if os.path.basename(video_dir) == "uploaded":
        video_dir = os.path.dirname(video_dir)
    if os.path.basename(video_dir) != "videos":
        raise ValueError(
            "Expected the short to live inside a Video Clips/videos folder: "
            + video_path
        )
    clips_root = os.path.dirname(video_dir)
    metadata_path = os.path.join(clips_root, "upload_metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing Shorts upload metadata: {metadata_path}")
    return clips_root


def find_clip_for_video(upload_metadata: Dict[str, Any], video_path: str) -> Dict[str, Any]:
    requested = os.path.abspath(video_path)
    requested_name = os.path.basename(requested)
    for clip in upload_metadata.get("clips") or []:
        known_path = os.path.abspath(str(clip.get("video_path") or ""))
        if known_path == requested or os.path.basename(known_path) == requested_name:
            return clip
    raise ValueError(f"Could not find video in upload_metadata.json: {video_path}")


def shorts_title(title: str) -> str:
    title = str(title or "Catholic Homily Short").strip()
    suffix = " #Shorts"
    if "#shorts" in title.lower():
        return title[:100]
    if len(title) + len(suffix) <= 100:
        return title + suffix
    return title[: 100 - len(suffix)].rstrip() + suffix


def shorts_description(clip: Dict[str, Any], source: Dict[str, Any]) -> str:
    description = str(clip.get("description") or clip.get("theme") or "").strip()
    title = str(clip.get("title") or "").strip()
    source_mdx = str((source or {}).get("mdx") or "").strip()

    lines: List[str] = []
    if description:
        lines.append(description)
    elif title:
        lines.append(title)

    lines.append("From a traditional Catholic homily.")
    if source_mdx:
        lines.append(f"Source notes: {os.path.basename(source_mdx)}")
    lines.append("#Shorts #Catholic #TraditionalCatholic #LatinMass")
    return "\n\n".join(line for line in lines if line).strip()


def shorts_tags(clip: Dict[str, Any]) -> List[str]:
    clip_tags = [str(tag) for tag in (clip.get("tags") or [])]
    return dedupe_keep_order(clip_tags + SHORTS_BASE_TAGS)[:500]


def upload_record_path(clips_root: str) -> str:
    return os.path.join(clips_root, "shorts_youtube_uploads.json")


def load_upload_records(clips_root: str) -> Dict[str, Any]:
    path = upload_record_path(clips_root)
    if os.path.isfile(path):
        return load_json(path)
    return {"uploads": []}


def already_uploaded(records: Dict[str, Any], clip_id: str, video_path: str) -> Optional[Dict[str, Any]]:
    requested_name = os.path.basename(video_path)
    for record in records.get("uploads") or []:
        if record.get("clip_id") == clip_id or os.path.basename(str(record.get("original_video_path") or "")) == requested_name:
            return record
    return None


def set_thumbnail(tokens: Dict[str, Any], video_id: str, thumbnail_path: str) -> bool:
    if not thumbnail_path or not os.path.isfile(thumbnail_path):
        print(f"No thumbnail file found: {thumbnail_path}")
        return False
    if not is_valid_youtube_thumbnail(thumbnail_path):
        print(f"Thumbnail is not a valid YouTube upload image: {thumbnail_path}")
        return False
    result = youtube_update_video(
        tokens=tokens,
        video_id=video_id,
        new_thumbnail_path=thumbnail_path,
    )
    return bool(result and result.get("thumbnail_response") is not None)


def move_uploaded_video(video_path: str) -> str:
    video_dir = os.path.dirname(video_path)
    if os.path.basename(video_dir) == "uploaded":
        return video_path
    uploaded_dir = os.path.join(video_dir, "uploaded")
    os.makedirs(uploaded_dir, exist_ok=True)
    destination = os.path.join(uploaded_dir, os.path.basename(video_path))
    if os.path.abspath(video_path) == os.path.abspath(destination):
        return destination
    if os.path.exists(destination):
        base, ext = os.path.splitext(destination)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = f"{base}-{stamp}{ext}"
    shutil.move(video_path, destination)
    return destination


def update_metadata_video_path(metadata_path: str, clip_id: str, new_video_path: str) -> None:
    metadata = load_json(metadata_path)
    for clip in metadata.get("clips") or []:
        if clip.get("id") == clip_id:
            clip["video_path"] = new_video_path
            break
    save_json(metadata_path, metadata)


def upload_short(video_path: str, google_user_id: str, move_after_upload: bool = True) -> Dict[str, Any]:
    video_path = resolve_video_path(video_path)
    clips_root = clips_root_from_video(video_path)
    metadata_path = os.path.join(clips_root, "upload_metadata.json")
    metadata = load_json(metadata_path)
    source = metadata.get("source") or {}
    clip = find_clip_for_video(metadata, video_path)
    clip_id = str(clip.get("id") or os.path.splitext(os.path.basename(video_path))[0])

    records = load_upload_records(clips_root)
    previous = already_uploaded(records, clip_id, video_path)
    if previous:
        raise RuntimeError(
            f"{clip_id} already appears uploaded as YouTube video {previous.get('youtube_video_id')}. "
            "Remove the record from shorts_youtube_uploads.json if you intentionally need to re-upload."
        )

    title = shorts_title(str(clip.get("title") or "Catholic Homily Short"))
    description = shorts_description(clip, source)
    tags = shorts_tags(clip)
    thumbnail_path = str(clip.get("thumbnail_path") or "")
    captions_path = str(clip.get("captions_path") or "")

    tokens = get_and_refresh_google_user_tokens(google_user_id)
    video_id = youtube_upload_video(
        tokens=tokens,
        file_path=video_path,
        title=title,
        description=description,
        tags=tags,
        categoryId="27",
        privacyStatus="private",
        edu_type=None,
        edu_problems=None,
        chapters=None,
    )
    if not video_id:
        raise RuntimeError("YouTube upload failed: no video_id returned.")

    thumbnail_set = False
    try:
        thumbnail_set = set_thumbnail(tokens, video_id, thumbnail_path)
    except HttpError as exc:
        print(f"YouTube rejected the custom thumbnail for {video_id}: {exc}")

    captions_uploaded = False
    if captions_path and os.path.isfile(captions_path):
        captions_uploaded = bool(
            youtube_upload_captions(
                tokens=tokens,
                video_id=video_id,
                caption_file_path=captions_path,
                language="en",
                name="English Captions",
            )
        )

    final_video_path = video_path
    if move_after_upload:
        final_video_path = move_uploaded_video(video_path)
        update_metadata_video_path(metadata_path, clip_id, final_video_path)

    record = {
        "clip_id": clip_id,
        "youtube_video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "privacy_status": "private",
        "is_short": True,
        "original_video_path": video_path,
        "video_path": final_video_path,
        "thumbnail_path": thumbnail_path,
        "thumbnail_set": thumbnail_set,
        "captions_path": captions_path,
        "captions_uploaded": captions_uploaded,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    records.setdefault("uploads", []).append(record)
    save_json(upload_record_path(clips_root), records)
    return record


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    parser = argparse.ArgumentParser(description="Upload generated homily Shorts to YouTube as private drafts.")
    parser.add_argument("video", help="Path to a generated Shorts MP4 inside Video Clips/videos.")
    parser.add_argument("--google-user-id", default=os.getenv("GOOGLE_USER_ID", DEFAULT_GOOGLE_USER_ID))
    parser.add_argument("--no-move", action="store_true", help="Do not move the uploaded MP4 into videos/uploaded.")
    args = parser.parse_args()

    record = upload_short(
        video_path=args.video,
        google_user_id=args.google_user_id,
        move_after_upload=not args.no_move,
    )
    print("\nUploaded Shorts draft:")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
