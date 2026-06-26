import argparse
import json
import os
import random
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageFilter, ImageOps

from get_token import get_and_refresh_google_user_tokens
from thumbnail_generator import MAX_THUMBNAIL_BYTES, THUMBNAIL_SIZE, is_valid_youtube_thumbnail
from youtube import (
    build_youtube_service,
    finalize_youtube_description,
    youtube_update_video,
    youtube_upload_captions,
    youtube_upload_video,
)


DEFAULT_GOOGLE_USER_ID = "102136376185174842894"
DEFAULT_SCHEDULE_TIMEZONE = "America/New_York"
DEFAULT_SCHEDULE_WINDOW_START = "16:00"
DEFAULT_SCHEDULE_WINDOW_END = "20:00"
DEFAULT_SCHEDULE_DAYS = 5
HASHTAG_VOLUME_CACHE_MAX_AGE_DAYS = 30
SHORTS_BASE_TAGS = [
    "Catholic",
    "Christian",
    "Faith",
    "Prayer",
    "Worship",
    "Motivation",
    "Traditional Catholic",
    "Latin Mass",
    "Traditional Mass",
    "Catholic Mass",
    "Catholic homily",
    "YouTube Shorts",
    "Shorts",
]

BROAD_HASHTAGS = [
    "#catholic",
    "#christian",
    "#faith",
    "#prayer",
    "#catholicfaith",
    "#traditionalcatholic",
]

CONTENT_HASHTAG_RULES = [
    ({"mary", "our lady", "blessed virgin", "immaculate heart"}, "#mary", 9),
    ({"joseph", "saint joseph", "st joseph"}, "#saintjoseph", 9),
    ({"holy family", "mary and joseph", "nativity"}, "#holyfamily", 9),
    ({"jesus", "christ", "sacred heart", "our lord"}, "#jesus", 8),
    ({"sacred heart"}, "#sacredheart", 8),
    ({"rosary", "devotion", "pray", "prayer"}, "#catholicprayer", 8),
    ({"mercy", "compassion", "charity", "kindness"}, "#compassion", 8),
    ({"motivation", "courage", "persevere", "strength"}, "#christianmotivation", 7),
    ({"sin", "temptation", "conversion", "repent", "repentance"}, "#conversion", 8),
    ({"hope", "saved", "salvation", "soul"}, "#hope", 7),
    ({"salvation", "saved", "soul"}, "#salvation", 7),
    ({"pride", "humility", "humble"}, "#humility", 8),
    ({"cross", "suffering", "sacrifice", "passion"}, "#cross", 8),
    ({"mass", "eucharist", "communion", "altar", "chalice"}, "#eucharist", 8),
    ({"mass", "latin mass", "traditional mass", "altar"}, "#catholicmass", 7),
    ({"latin mass", "traditional mass", "sspx", "pius x"}, "#latinmass", 8),
    ({"traditional", "tradition", "latin mass", "traditional mass", "sspx"}, "#traditionalmass", 7),
    ({"gospel", "scripture", "bible", "parable"}, "#gospel", 7),
    ({"bible", "scripture"}, "#bible", 6),
    ({"saint", "saints"}, "#saints", 7),
    ({"father", "priest", "homily", "sermon"}, "#homily", 7),
]


class NoPendingVideosError(RuntimeError):
    pass


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


def slugify(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value or "").strip("-").lower()
    return value or fallback


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


def clips_root_from_folder(folder_path: str) -> str:
    folder_path = os.path.abspath(os.path.expanduser(clean_path(folder_path)))
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    candidates = [
        folder_path,
        os.path.join(folder_path, "Video Clips"),
    ]

    if os.path.basename(folder_path) == "videos":
        candidates.append(os.path.dirname(folder_path))
    elif os.path.basename(folder_path) == "uploaded":
        videos_dir = os.path.dirname(folder_path)
        if os.path.basename(videos_dir) == "videos":
            candidates.append(os.path.dirname(videos_dir))

    for candidate in candidates:
        metadata_path = os.path.join(candidate, "upload_metadata.json")
        if os.path.isfile(metadata_path):
            return candidate

    raise FileNotFoundError(
        "Could not find upload_metadata.json. Pass a short MP4, a homily folder, "
        "a Video Clips folder, or a Video Clips/videos folder."
    )


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
    title = re.sub(r"(?i)\s*#shorts?\b", "", title).strip()
    return title[:100].rstrip()


def youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def shorts_description(
    clip: Dict[str, Any],
    source: Dict[str, Any],
    related_video_id: str = "",
    hashtags: Optional[List[str]] = None,
) -> str:
    description = str(clip.get("description") or clip.get("theme") or "").strip()
    title = str(clip.get("title") or "").strip()
    source_mdx = str((source or {}).get("mdx") or "").strip()

    lines: List[str] = []
    if description:
        lines.append(description)
    elif title:
        lines.append(title)

    if related_video_id:
        lines.append(f"Watch the full homily:\n{youtube_watch_url(related_video_id)}")

    lines.append("From a traditional Catholic homily.")
    if source_mdx:
        lines.append(f"Source notes: {os.path.basename(source_mdx)}")
    lines.append(" ".join(hashtags or shorts_hashtags(clip)))
    return "\n\n".join(line for line in lines if line).strip()


def shorts_hashtag_text(clip: Dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in [
            clip.get("title"),
            clip.get("description"),
            clip.get("theme"),
            clip.get("power_quote"),
            clip.get("why_it_works"),
            clip.get("source_text"),
            " ".join(str(item) for item in (clip.get("image_search_terms") or [])),
            " ".join(str(item) for item in (clip.get("keywords") or [])),
        ]
    ).lower()


def hashtag_match_count(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def candidate_hashtag_relevance(clip: Dict[str, Any]) -> Dict[str, int]:
    text = shorts_hashtag_text(clip)
    scored: Dict[str, int] = {}

    for index, hashtag in enumerate(BROAD_HASHTAGS):
        scored[hashtag] = max(scored.get(hashtag, 0), 40 - index)

    for terms, hashtag, weight in CONTENT_HASHTAG_RULES:
        match_count = hashtag_match_count(text, terms)

        if match_count:
            scored[hashtag] = max(scored.get(hashtag, 0), 100 + int(weight) * 10 + match_count)

    return scored


def hashtag_volume_cache_path(clips_root: str) -> str:
    return os.path.join(clips_root, "hashtag_volume_cache.json")


def load_hashtag_volume_cache(cache_path: Optional[str]) -> Dict[str, Any]:
    if cache_path and os.path.isfile(cache_path):
        try:
            return load_json(cache_path)
        except Exception:
            return {"hashtags": {}}
    return {"hashtags": {}}


def cache_entry_is_fresh(entry: Dict[str, Any]) -> bool:
    updated_at = str(entry.get("updated_at") or "")

    if not updated_at:
        return False

    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False

    return updated >= datetime.now(timezone.utc) - timedelta(days=HASHTAG_VOLUME_CACHE_MAX_AGE_DAYS)


def fetch_hashtag_volume(youtube: Any, hashtag: str) -> Optional[int]:
    try:
        response = (
            youtube.search()
            .list(
                part="id",
                q=hashtag,
                type="video",
                maxResults=1,
                order="relevance",
            )
            .execute()
        )
        return int((response.get("pageInfo") or {}).get("totalResults") or 0)
    except HttpError as exc:
        print(f"Could not fetch hashtag volume for {hashtag}: {exc}")
        return None
    except Exception as exc:
        print(f"Could not fetch hashtag volume for {hashtag}: {exc}")
        return None


def hashtag_volumes(
    hashtags: List[str],
    tokens: Optional[Dict[str, Any]] = None,
    cache_path: Optional[str] = None,
    refresh: bool = False,
) -> Dict[str, int]:
    if not cache_path:
        return {}

    cache = load_hashtag_volume_cache(cache_path)
    cached_hashtags = cache.setdefault("hashtags", {})
    volumes: Dict[str, int] = {}
    youtube = None
    changed = False

    for hashtag in hashtags:
        key = hashtag.lower()
        entry = cached_hashtags.get(key) or {}

        if not refresh and entry and cache_entry_is_fresh(entry):
            volumes[hashtag] = int(entry.get("total_results") or 0)
            continue

        if not tokens:
            continue

        if youtube is None:
            youtube = build_youtube_service(tokens)

        volume = fetch_hashtag_volume(youtube, hashtag)

        if volume is None:
            if entry:
                volumes[hashtag] = int(entry.get("total_results") or 0)
            continue

        volumes[hashtag] = volume
        cached_hashtags[key] = {
            "hashtag": hashtag,
            "total_results": volume,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        changed = True

    if changed:
        save_json(cache_path, cache)

    return volumes


def shorts_hashtags(
    clip: Dict[str, Any],
    limit: int = 10,
    tokens: Optional[Dict[str, Any]] = None,
    cache_path: Optional[str] = None,
    use_volume: bool = True,
    refresh_volume: bool = False,
) -> List[str]:
    scored = candidate_hashtag_relevance(clip)
    candidates = list(scored.keys())
    volumes = hashtag_volumes(
        candidates,
        tokens=tokens,
        cache_path=cache_path,
        refresh=refresh_volume,
    ) if use_volume else {}

    if volumes:
        ranked = sorted(
            scored.items(),
            key=lambda item: (-(volumes.get(item[0], -1)), -item[1], item[0]),
        )
    else:
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))

    return dedupe_keep_order([hashtag for hashtag, _score in ranked])[:limit]


def shorts_tags(clip: Dict[str, Any], hashtags: Optional[List[str]] = None) -> List[str]:
    clip_tags = [str(tag) for tag in (clip.get("tags") or [])]
    hashtag_tags = [tag.lstrip("#") for tag in (hashtags or shorts_hashtags(clip))]
    return dedupe_keep_order(clip_tags + hashtag_tags + SHORTS_BASE_TAGS)[:500]


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


def uploadable_video_path(clips_root: str, video_path: str, allow_reupload: bool) -> str:
    video_path = os.path.abspath(str(video_path or ""))
    if allow_reupload and f"{os.sep}uploaded{os.sep}" in video_path:
        fresh_path = os.path.join(clips_root, "videos", os.path.basename(video_path))
        if os.path.isfile(fresh_path):
            return fresh_path
    return video_path


def pending_videos_from_folder(folder_path: str, allow_reupload: bool = False) -> List[str]:
    clips_root = clips_root_from_folder(folder_path)
    metadata = load_json(os.path.join(clips_root, "upload_metadata.json"))
    records = load_upload_records(clips_root)
    pending: List[str] = []
    skipped_uploaded = 0

    for clip in metadata.get("clips") or []:
        clip_id = str(clip.get("id") or "")
        video_path = uploadable_video_path(clips_root, str(clip.get("video_path") or ""), allow_reupload)
        if not video_path:
            continue
        if not allow_reupload and f"{os.sep}uploaded{os.sep}" in video_path:
            skipped_uploaded += 1
            continue
        if not allow_reupload and already_uploaded(records, clip_id, video_path):
            skipped_uploaded += 1
            continue
        if not os.path.isfile(video_path):
            print(f"Skipping missing video for {clip_id or 'unknown clip'}: {video_path}")
            continue
        pending.append(resolve_video_path(video_path))

    if not pending:
        message = f"No pending Shorts videos found in: {clips_root}"
        if skipped_uploaded:
            message += (
                f"\n{skipped_uploaded} clip(s) already appear in shorts_youtube_uploads.json "
                "or have been moved to videos/uploaded."
                "\nUse --reupload if these are regenerated videos that should be uploaded as new drafts."
            )
        raise NoPendingVideosError(message)
    return pending


def resolve_upload_targets(target: str, allow_reupload: bool = False) -> List[str]:
    resolved = os.path.abspath(os.path.expanduser(clean_path(target)))
    if os.path.isfile(resolved):
        return [resolve_video_path(resolved)]
    if os.path.isdir(resolved):
        return pending_videos_from_folder(resolved, allow_reupload=allow_reupload)
    raise FileNotFoundError(f"Upload target does not exist: {resolved}")


def parse_hhmm(value: str) -> time:
    try:
        hour, minute = str(value or "").strip().split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except Exception as exc:
        raise ValueError(f"Expected time in HH:MM format, got: {value}") from exc


def parse_schedule_start_date(value: Optional[str], tz: ZoneInfo) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()

    return (datetime.now(tz) + timedelta(days=1)).date()


def random_schedule_times(
    count: int,
    start_date: date,
    days: int = DEFAULT_SCHEDULE_DAYS,
    window_start: str = DEFAULT_SCHEDULE_WINDOW_START,
    window_end: str = DEFAULT_SCHEDULE_WINDOW_END,
    timezone_name: str = DEFAULT_SCHEDULE_TIMEZONE,
) -> List[datetime]:
    if count <= 0:
        return []

    tz = ZoneInfo(timezone_name)
    start_time = parse_hhmm(window_start)
    end_time = parse_hhmm(window_end)
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute

    if end_minutes <= start_minutes:
        raise ValueError("--schedule-window-end must be later than --schedule-window-start.")

    days = max(1, int(days))
    now = datetime.now(tz)
    scheduled: List[datetime] = []

    for _ in range(count):
        for _attempt in range(100):
            day = start_date + timedelta(days=random.randrange(days))
            minute = random.randrange(start_minutes, end_minutes + 1)
            publish_at = datetime.combine(
                day,
                time(hour=minute // 60, minute=minute % 60),
                tzinfo=tz,
            )

            if publish_at > now + timedelta(minutes=15):
                scheduled.append(publish_at)
                break
        else:
            fallback_day = max(start_date, (now + timedelta(days=1)).date())
            minute = random.randrange(start_minutes, end_minutes + 1)
            scheduled.append(
                datetime.combine(
                    fallback_day,
                    time(hour=minute // 60, minute=minute % 60),
                    tzinfo=tz,
                )
            )

    return sorted(scheduled)


def youtube_publish_at(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def find_mdx_files_for_clips_root(clips_root: str) -> List[str]:
    homily_root = os.path.dirname(os.path.abspath(clips_root))
    candidates: List[str] = []

    for root, _dirs, files in os.walk(homily_root):
        if f"{os.sep}Video Clips{os.sep}" in root:
            continue

        for filename in files:
            if filename.lower().endswith(".mdx"):
                candidates.append(os.path.join(root, filename))

    candidates.sort(
        key=lambda path: (
            0 if f"{os.sep}final{os.sep}" in path else 1,
            len(path),
            path.lower(),
        )
    )
    return candidates


def frontmatter_block(text: str) -> str:
    if not text.startswith("---"):
        return ""

    match = re.match(r"^---\s*\n(.*?)\n---\s*", text, flags=re.DOTALL)
    return match.group(1) if match else ""


def frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*['\"]?([^'\"\n]+)['\"]?\s*$", frontmatter)
    return clean_path(match.group(1)) if match else ""


def extract_youtube_video_id(value: str) -> str:
    value = clean_path(value)

    if not value:
        return ""

    for pattern in [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{6,})",
        r"^[A-Za-z0-9_-]{8,}$",
    ]:
        match = re.search(pattern, value)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    return ""


def find_related_video_id_from_mdx(clips_root: str) -> str:
    for mdx_path in find_mdx_files_for_clips_root(clips_root):
        try:
            text = open(mdx_path, "r", encoding="utf-8").read()
        except Exception:
            continue

        fm = frontmatter_block(text)
        if not fm:
            continue

        for key in [
            "youtube_video_id",
            "youtube_id",
            "youtubeId",
            "video_id",
            "videoId",
            "media_path",
        ]:
            video_id = extract_youtube_video_id(frontmatter_value(fm, key))

            if video_id:
                return video_id

        video_id = extract_youtube_video_id(fm)

        if video_id:
            return video_id

    return ""


def youtube_upload_scheduled_short_video(
    tokens: Dict[str, Any],
    file_path: str,
    title: str,
    description: str,
    tags: Optional[List[str]],
    publish_at: datetime,
    category_id: str = "27",
) -> Optional[str]:
    youtube = build_youtube_service(tokens)
    publish_at_utc = youtube_publish_at(publish_at)

    final_description = finalize_youtube_description(
        description=description,
        tags=tags,
        chapters=None,
        edu_type=None,
        edu_problems=None,
    )

    body = {
        "snippet": {
            "title": title,
            "description": final_description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at_utc,
        },
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    try:
        print("Uploading scheduled public Shorts video...")
        print(f"File: {file_path}")
        print(f"Title: {title}")
        print(f"Category ID: {category_id}")
        print("Privacy Status: private until scheduled public release")
        print(f"Scheduled Publish At: {publish_at_utc}")
        if tags:
            print(f"Tags: {tags}")

        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None

        while response is None:
            status, response = request.next_chunk()

            if status:
                print(f"Uploaded {int(status.progress() * 100)}%...")

        print("Scheduled public Shorts upload complete!")
        return response.get("id") if response else None

    except HttpError as exc:
        print(f"An HTTP error {exc.resp.status} occurred:\n{exc.content}")
        return None


def upload_thumbnail_path(clips_root: str, clip: Dict[str, Any]) -> str:
    clip_id = str(clip.get("id") or "clip")
    title = str(clip.get("title") or clip_id)
    return os.path.join(clips_root, "thumbnails", f"{clip_id}-{slugify(title, clip_id)}.jpg")


def cover_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    target_ratio = target_width / target_height
    width, height = image.size
    ratio = width / height

    if ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image.resize(size, Image.Resampling.LANCZOS)


def save_compressed_jpeg(image: Image.Image, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for quality in range(92, 54, -4):
        image.save(output_path, format="JPEG", quality=quality, optimize=True, progressive=True)
        if os.path.getsize(output_path) <= MAX_THUMBNAIL_BYTES:
            return output_path
    raise RuntimeError(f"Could not compress YouTube thumbnail below {MAX_THUMBNAIL_BYTES} bytes: {output_path}")


def ensure_upload_thumbnail(clips_root: str, clip: Dict[str, Any]) -> str:
    source_path = os.path.abspath(str(clip.get("thumbnail_path") or ""))
    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Missing source thumbnail image for {clip.get('id')}: {source_path}")

    output_path = upload_thumbnail_path(clips_root, clip)
    if (
        os.path.isfile(output_path)
        and is_valid_youtube_thumbnail(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(source_path)
    ):
        return output_path

    image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    background = cover_crop(image, THUMBNAIL_SIZE).filter(ImageFilter.GaussianBlur(18))
    background = ImageOps.autocontrast(background)

    overlay = Image.new("RGB", THUMBNAIL_SIZE, (0, 0, 0))
    background = Image.blend(background, overlay, 0.18)

    contained = image.copy()
    contained.thumbnail((int(THUMBNAIL_SIZE[0] * 0.72), THUMBNAIL_SIZE[1]), Image.Resampling.LANCZOS)
    x = (THUMBNAIL_SIZE[0] - contained.width) // 2
    y = (THUMBNAIL_SIZE[1] - contained.height) // 2
    background.paste(contained, (x, y))

    return save_compressed_jpeg(background, output_path)


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


def clips_root_from_target(target: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(clean_path(target)))
    if os.path.isfile(resolved):
        return clips_root_from_video(resolve_video_path(resolved))
    if os.path.isdir(resolved):
        return clips_root_from_folder(resolved)
    raise FileNotFoundError(f"Upload target does not exist: {resolved}")


def repair_uploaded_thumbnails(target: str, google_user_id: str) -> List[Dict[str, Any]]:
    clips_root = clips_root_from_target(target)
    metadata = load_json(os.path.join(clips_root, "upload_metadata.json"))
    clips_by_id = {
        str(clip.get("id")): clip
        for clip in metadata.get("clips") or []
        if clip.get("id")
    }
    records = load_upload_records(clips_root)
    uploads = records.get("uploads") or []

    resolved_target = os.path.abspath(os.path.expanduser(clean_path(target)))
    if os.path.isfile(resolved_target):
        target_name = os.path.basename(resolved_target)
        uploads = [
            record
            for record in uploads
            if os.path.basename(str(record.get("video_path") or "")) == target_name
            or os.path.basename(str(record.get("original_video_path") or "")) == target_name
        ]

    if not uploads:
        raise RuntimeError(f"No uploaded Shorts records found for: {target}")

    tokens = get_and_refresh_google_user_tokens(google_user_id)
    results: List[Dict[str, Any]] = []
    for record in uploads:
        clip_id = str(record.get("clip_id") or "")
        video_id = str(record.get("youtube_video_id") or "")
        clip = clips_by_id.get(clip_id)
        if not clip or not video_id:
            print(f"Skipping incomplete upload record for {clip_id or 'unknown clip'}.")
            continue

        thumbnail_path = ensure_upload_thumbnail(clips_root, clip)
        thumbnail_set = set_thumbnail(tokens, video_id, thumbnail_path)
        record["thumbnail_path"] = thumbnail_path
        record["thumbnail_set"] = thumbnail_set
        record["thumbnail_repaired_at"] = datetime.now(timezone.utc).isoformat()
        results.append(
            {
                "clip_id": clip_id,
                "youtube_video_id": video_id,
                "thumbnail_path": thumbnail_path,
                "thumbnail_set": thumbnail_set,
            }
        )

    save_json(upload_record_path(clips_root), records)
    return results


def upload_short(
    video_path: str,
    google_user_id: str,
    move_after_upload: bool = True,
    allow_reupload: bool = False,
    scheduled_publish_at: Optional[datetime] = None,
    related_video_id: str = "",
    add_related_video_link: bool = True,
    use_volume_hashtags: bool = True,
    refresh_hashtag_volume: bool = False,
) -> Dict[str, Any]:
    video_path = resolve_video_path(video_path)
    clips_root = clips_root_from_video(video_path)
    metadata_path = os.path.join(clips_root, "upload_metadata.json")
    metadata = load_json(metadata_path)
    source = metadata.get("source") or {}
    clip = find_clip_for_video(metadata, video_path)
    clip_id = str(clip.get("id") or os.path.splitext(os.path.basename(video_path))[0])

    records = load_upload_records(clips_root)
    previous = already_uploaded(records, clip_id, video_path)
    if previous and not allow_reupload:
        raise RuntimeError(
            f"{clip_id} already appears uploaded as YouTube video {previous.get('youtube_video_id')}. "
            "Remove the record from shorts_youtube_uploads.json if you intentionally need to re-upload."
        )

    if not add_related_video_link:
        related_video_id = ""
    elif related_video_id:
        related_video_id = extract_youtube_video_id(related_video_id)
    else:
        related_video_id = find_related_video_id_from_mdx(clips_root)

    tokens = get_and_refresh_google_user_tokens(google_user_id)

    title = shorts_title(str(clip.get("title") or "Catholic Homily Short"))
    hashtags = shorts_hashtags(
        clip,
        tokens=tokens,
        cache_path=hashtag_volume_cache_path(clips_root),
        use_volume=use_volume_hashtags,
        refresh_volume=refresh_hashtag_volume,
    )
    description = shorts_description(
        clip,
        source,
        related_video_id=related_video_id,
        hashtags=hashtags,
    )
    tags = shorts_tags(clip, hashtags=hashtags)
    thumbnail_path = ensure_upload_thumbnail(clips_root, clip)
    captions_path = str(clip.get("captions_path") or "")

    if scheduled_publish_at:
        video_id = youtube_upload_scheduled_short_video(
            tokens=tokens,
            file_path=video_path,
            title=title,
            description=description,
            tags=tags,
            publish_at=scheduled_publish_at,
            category_id="27",
        )
    else:
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
        "privacy_status": "scheduled_public" if scheduled_publish_at else "private",
        "scheduled_publish_at": scheduled_publish_at.isoformat() if scheduled_publish_at else "",
        "related_video_id": related_video_id,
        "related_video_url": youtube_watch_url(related_video_id) if related_video_id else "",
        "hashtags": hashtags,
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
    parser = argparse.ArgumentParser(description="Upload generated homily Shorts to YouTube as private drafts or scheduled public releases.")
    parser.add_argument(
        "target",
        nargs="?",
        help="A generated Shorts MP4, homily folder, Video Clips folder, or Video Clips/videos folder.",
    )
    parser.add_argument("--google-user-id", default=os.getenv("GOOGLE_USER_ID", DEFAULT_GOOGLE_USER_ID))
    parser.add_argument("--no-move", action="store_true", help="Do not move the uploaded MP4 into videos/uploaded.")
    parser.add_argument("--thumbnail-only", action="store_true", help="Reapply upload-safe thumbnails to already uploaded Shorts without uploading videos.")
    parser.add_argument("--reupload", action="store_true", help="Upload videos even when the clip id already appears in shorts_youtube_uploads.json.")
    parser.add_argument("--schedule", action="store_true", help="Schedule Shorts for public release at random times instead of leaving them as private drafts.")
    parser.add_argument("--schedule-start-date", help="First local date eligible for scheduling, in YYYY-MM-DD. Defaults to tomorrow.")
    parser.add_argument("--schedule-days", type=int, default=DEFAULT_SCHEDULE_DAYS, help="Number of days to randomly distribute scheduled Shorts across. Default: 5.")
    parser.add_argument("--schedule-window-start", default=DEFAULT_SCHEDULE_WINDOW_START, help="Earliest local publish time, HH:MM. Default: 16:00.")
    parser.add_argument("--schedule-window-end", default=DEFAULT_SCHEDULE_WINDOW_END, help="Latest local publish time, HH:MM. Default: 20:00.")
    parser.add_argument("--schedule-timezone", default=DEFAULT_SCHEDULE_TIMEZONE, help="IANA timezone for scheduling. Default: America/New_York.")
    parser.add_argument("--related-video-id", help="Optional full homily YouTube video ID or URL. Defaults to the MDX media_path/youtube id when found.")
    parser.add_argument("--no-related-video-link", action="store_true", help="Do not add the full homily link to Shorts descriptions.")
    parser.add_argument("--no-volume-hashtags", action="store_true", help="Use relevance-ranked hashtags without checking YouTube search volume.")
    parser.add_argument("--refresh-hashtag-volume", action="store_true", help="Refresh cached YouTube hashtag volume estimates before ranking hashtags.")
    args = parser.parse_args()

    if not args.target:
        args.target = clean_path(input("Enter Shorts video or folder path to upload: "))

    if args.thumbnail_only:
        records = repair_uploaded_thumbnails(args.target, args.google_user_id)
        print("\nRepaired Shorts thumbnail(s):")
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return

    try:
        videos = resolve_upload_targets(args.target, allow_reupload=args.reupload)
    except NoPendingVideosError as exc:
        print(str(exc))
        return

    scheduled_times: List[Optional[datetime]] = [None] * len(videos)

    if args.schedule:
        schedule_tz = ZoneInfo(args.schedule_timezone)
        schedule_start = parse_schedule_start_date(args.schedule_start_date, schedule_tz)
        scheduled_times = random_schedule_times(
            count=len(videos),
            start_date=schedule_start,
            days=args.schedule_days,
            window_start=args.schedule_window_start,
            window_end=args.schedule_window_end,
            timezone_name=args.schedule_timezone,
        )

    print(f"Uploading {len(videos)} Shorts draft(s)...")

    if args.schedule:
        print(
            "Scheduling randomly between "
            f"{args.schedule_window_start} and {args.schedule_window_end} "
            f"{args.schedule_timezone} across {args.schedule_days} day(s)."
        )

    related_video_id = clean_path(args.related_video_id or "")

    records = []
    for video_path, scheduled_publish_at in zip(videos, scheduled_times):
        print(f"\nUploading: {video_path}")
        if scheduled_publish_at:
            print(f"Scheduled local publish time: {scheduled_publish_at.isoformat()}")
        records.append(
            upload_short(
                video_path=video_path,
                google_user_id=args.google_user_id,
                move_after_upload=not args.no_move,
                allow_reupload=args.reupload,
                scheduled_publish_at=scheduled_publish_at,
                related_video_id=related_video_id,
                add_related_video_link=not args.no_related_video_link,
                use_volume_hashtags=not args.no_volume_hashtags,
                refresh_hashtag_volume=args.refresh_hashtag_volume,
            )
        )

    print("\nUploaded Shorts draft(s):")
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
