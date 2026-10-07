import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydub import AudioSegment
import numpy as np


CANVAS_SIZE = (1080, 1920)
IMAGE_SIZE = CANVAS_SIZE
IMAGE_RATIO = CANVAS_SIZE[0] / CANVAS_SIZE[1]
IMAGE_TOP = 0
GRADIENT_TOP = 1280
GRADIENT_BOTTOM = CANVAS_SIZE[1]
GRADIENT_MAX_ALPHA = 165
FPS = 30

TEXT_SAFE_LEFT = 74
TEXT_SAFE_RIGHT = 74
TEXT_SAFE_TOP = 760
TEXT_SAFE_BOTTOM = 760
CAPTION_MAX_LINES = 2
CAPTION_MAX_WORDS = 7
CAPTION_MAX_CHARS = 54

BACKGROUND_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
DEFAULT_BG_MUSIC_GAIN_DB = -22.0
DEFAULT_BG_MUSIC_FADE_MS = 2500
MAX_BG_MUSIC_SECONDS = 55.0

REQUEST_TIMEOUT = 18
MIN_PUBLIC_DOMAIN_SCORE = 7.0
NGA_OBJECTS_CSV_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv"
NGA_PUBLISHED_IMAGES_CSV_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"
NGA_INDEX_VERSION = 2
CATHOLIC_TRADITION_GALLERY_URL = "https://www.catholictradition.org/Galleries/gallery-one.htm"
CATHOLIC_TRADITION_INDEX_VERSION = 1
CATHOLIC_TRADITION_MAX_GALLERY_PAGES = 70

PAINTING_TERMS = {
    "painting",
    "painted",
    "oil",
    "tempera",
    "watercolor",
    "watercolour",
    "fresco",
    "canvas",
    "panel",
    "altarpiece",
    "manuscript illumination",
    "illumination",
    "gouache",
    "pastel",
}

SACRED_CATHOLIC_TERMS = {
    "adoration",
    "angel",
    "annunciation",
    "apostle",
    "baptism",
    "blessed virgin",
    "calvary",
    "christ",
    "christ child",
    "christian",
    "church",
    "coronation of the virgin",
    "cross",
    "crucifixion",
    "deposition",
    "ecce homo",
    "eucharist",
    "evangelist",
    "flight into egypt",
    "holy family",
    "holy spirit",
    "infant jesus",
    "jesus",
    "john the baptist",
    "last judgment",
    "last supper",
    "madonna",
    "magi",
    "martyr",
    "mass",
    "obedience",
    "nativity",
    "passion",
    "penance",
    "peter",
    "pentecost",
    "pieta",
    "prayer",
    "sacrament",
    "sacraments",
    "resurrection",
    "saint",
    "souls",
    "st.",
    "st ",
    "trinity",
    "virgin",
}

NON_PAINTING_TERMS = {
    "amulet",
    "artifact",
    "badge",
    "bas-relief",
    "bowl",
    "bronze",
    "bust",
    "ceramic",
    "coin",
    "daguerreotype",
    "engraving",
    "etching",
    "figurine",
    "fragment",
    "furniture",
    "glass",
    "installation",
    "jar",
    "jewelry",
    "lithograph",
    "medal",
    "metal",
    "metalwork",
    "object",
    "ornament",
    "photograph",
    "poster",
    "print",
    "relief",
    "scarab",
    "sculpture",
    "statuette",
    "statue",
    "stone",
    "terracotta",
    "textile",
    "vessel",
    "woodcut",
}

SECULAR_ARTWORK_TERMS = {
    "abstract",
    "advertisement",
    "allegory",
    "battle",
    "cityscape",
    "costume",
    "fashion",
    "landscape",
    "mythological",
    "nude",
    "portrait",
    "poster",
    "seascape",
    "still life",
}

CATHOLIC_TRADITION_SKIP_TERMS = {
    "aspirations",
    "back",
    "banner",
    "bar",
    "desktop",
    "directory",
    "divider",
    "download",
    "email",
    "forward",
    "gem",
    "home",
    "icon",
    "quote",
    "scenic",
    "sculpture",
    "sources",
    "stained glass",
    "text",
    "wallpaper",
    "window",
}

TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL = (
    "Traditional Catholic visual guardrails: use pre-1962 Catholic sacred art references, "
    "traditional vestments, Latin Mass-era devotional imagery, modest reverence, and timeless church interiors. "
    "Avoid modern liturgical settings, modern vestments, celebrity-like clergy portraits, contemporary church architecture, "
    "political symbols, caricature, satire, text, logos, and watermarks."
)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slugify(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return value or fallback


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_folder(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def artwork_identity(candidate: Optional[Dict[str, Any]]) -> str:
    if not candidate:
        return ""

    source = clean_text(candidate.get("source", "")).lower()

    for field in ["image_id", "object_id", "source_url", "image_url"]:
        value = clean_text(candidate.get(field, ""))

        if value:
            return f"{source}:{field}:{value}"

    return ""


def reserve_artwork_key(
    candidate: Dict[str, Any],
    used_artwork_keys: Optional[set[str]],
    lock: Optional[threading.Lock],
) -> Tuple[bool, str]:
    key = artwork_identity(candidate)

    if not key or used_artwork_keys is None:
        return True, key

    if lock:
        with lock:
            if key in used_artwork_keys:
                return False, key
            used_artwork_keys.add(key)
            return True, key

    if key in used_artwork_keys:
        return False, key

    used_artwork_keys.add(key)
    return True, key


def release_artwork_key(
    key: str,
    used_artwork_keys: Optional[set[str]],
    lock: Optional[threading.Lock],
) -> None:
    if not key or used_artwork_keys is None:
        return

    if lock:
        with lock:
            used_artwork_keys.discard(key)
            return

    used_artwork_keys.discard(key)


def reserve_existing_artwork_key(
    key: str,
    used_artwork_keys: Optional[set[str]],
    lock: Optional[threading.Lock],
) -> bool:
    key = clean_text(key)

    if not key or used_artwork_keys is None:
        return True

    if lock:
        with lock:
            if key in used_artwork_keys:
                return False
            used_artwork_keys.add(key)
            return True

    if key in used_artwork_keys:
        return False

    used_artwork_keys.add(key)
    return True


def existing_artwork_key_from_meta(meta: Dict[str, Any]) -> str:
    return clean_text(meta.get("artwork_key", "")) or artwork_identity(meta.get("artwork"))


def format_timestamp(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))

    if millis >= 1000:
        whole += 1
        millis -= 1000

    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    sep = "," if srt else "."

    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def resolve_video_clips_folder(shorts_analysis_path: str | Path) -> Path:
    path = Path(shorts_analysis_path).expanduser().resolve()

    if path.is_dir():
        return path

    return path.parent


def resolve_output_paths(shorts_analysis_path: str | Path) -> Dict[str, Path]:
    video_clips_folder = resolve_video_clips_folder(shorts_analysis_path)

    paths = {
        "video_clips_folder": video_clips_folder,
        "shorts_analysis": video_clips_folder / "shorts_analysis.json",
        "shorts_manifest": video_clips_folder / "shorts_manifest.json",
        "image_candidates": video_clips_folder / "image_candidates.json",
        "upload_metadata": video_clips_folder / "upload_metadata.json",
        "images_dir": video_clips_folder / "images",
        "audio_dir": video_clips_folder / "audio",
        "videos_dir": video_clips_folder / "videos",
        "captions_dir": video_clips_folder / "captions",
        "artwork_credits": video_clips_folder / "artwork_credits.json",
    }

    for key in ["images_dir", "audio_dir", "videos_dir", "captions_dir"]:
        ensure_folder(paths[key])

    return paths


def get_selected_clips(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = [clip for clip in analysis.get("clips", []) if clip.get("selected")]

    if not selected:
        raise ValueError("No clips selected. Run Step #3 first and set selected clips to true.")

    selected.sort(key=lambda clip: int(clip.get("id", 0)))
    return selected


def get_clip_paths(clip: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Path]:
    clip_id = f"clip-{int(clip['id']):02d}"
    slug = slugify(clip.get("title", ""), clip_id)

    return {
        "image": paths["images_dir"] / f"{clip_id}-{slug}.jpg",
        "audio": paths["audio_dir"] / f"{clip_id}.mp3",
        "video": paths["videos_dir"] / f"{clip_id}-{slug}.mp4",
        "captions": paths["captions_dir"] / f"{clip_id}.srt",
        "image_meta": paths["images_dir"] / f"{clip_id}-{slug}.image.json",
    }


def load_existing_artwork_keys_for_other_clips(
    paths: Dict[str, Path],
    selected_clip_ids: set[int],
) -> set[str]:
    keys: set[str] = set()

    for meta_path in paths["images_dir"].glob("*.image.json"):
        try:
            meta = load_json(meta_path)
        except Exception:
            continue

        try:
            clip_id = int(meta.get("clip_id", 0))
        except Exception:
            clip_id = 0

        if clip_id in selected_clip_ids:
            continue

        key = existing_artwork_key_from_meta(meta)

        if key:
            keys.add(key)

    return keys


def build_search_terms(clip: Dict[str, Any]) -> List[str]:
    terms = []
    priority_terms = []

    for value in clip.get("image_search_terms", []):
        value = clean_text(value)
        if value:
            terms.append(value)

    title = clean_text(clip.get("title", ""))
    main_idea = clean_text(clip.get("main_idea", ""))
    image_idea = clean_text(clip.get("image_idea", ""))
    power_quote = clean_text(clip.get("power_quote", ""))

    for value in [title, main_idea, image_idea, power_quote]:
        if value:
            terms.append(value)
            terms.append(f"{value} painting")

    combined_clip_text = " ".join([title, image_idea, power_quote]).lower()

    catholic_fallbacks = [
        f"{title} Catholic painting",
        f"{title} sacred art",
        f"{title} Renaissance painting",
        f"{title} Baroque painting",
        f"{title} Biblical painting",
        "Christ carrying the cross",
        "Crucifixion Renaissance painting",
        "Last Supper chalice",
        "Mass of Saint Gregory",
        "saint in prayer",
        "Good Samaritan painting",
    ]

    if "mary" in combined_clip_text and "joseph" in combined_clip_text:
        priority_terms = [
            "Virgin Mary Saint Joseph oil painting",
            "Virgin Mary Saint Joseph painting",
            "Holy Family oil painting",
            "Holy Family painting",
            "The Holy Family Renaissance painting",
            "Marriage of the Virgin painting",
            "Saint Joseph Virgin Mary Child Jesus painting",
            "Nativity Virgin Mary Saint Joseph painting",
            "Flight into Egypt Holy Family painting",
        ]

    terms = priority_terms + terms + catholic_fallbacks

    unique = []
    seen = set()

    for term in terms:
        term = clean_text(term)
        key = term.lower()

        if key and key not in seen:
            unique.append(term)
            seen.add(key)

    return unique[:16]


def candidate_media_text(candidate: Dict[str, Any]) -> str:
    fields = [
        "title",
        "medium",
        "classification",
        "object_type",
        "department",
        "artwork_type",
    ]
    return " ".join(clean_text(candidate.get(field, "")) for field in fields).lower()


def candidate_subject_text(candidate: Dict[str, Any]) -> str:
    fields = [
        "title",
        "artist",
        "date",
        "medium",
        "classification",
        "object_type",
        "department",
        "artwork_type",
        "assistive_text",
    ]
    return " ".join(clean_text(candidate.get(field, "")) for field in fields).lower()


def candidate_is_painting(candidate: Dict[str, Any]) -> bool:
    text = candidate_media_text(candidate)

    if not text.strip():
        return False

    if any(term in text for term in NON_PAINTING_TERMS):
        return False

    return any(term in text for term in PAINTING_TERMS)


def candidate_is_catholic_painting(candidate: Dict[str, Any]) -> bool:
    if not candidate_is_painting(candidate):
        return False

    subject_text = candidate_subject_text(candidate)

    if not subject_text:
        return False

    has_sacred_subject = any(term in subject_text for term in SACRED_CATHOLIC_TERMS)

    if not has_sacred_subject:
        return False

    if any(term in subject_text for term in SECULAR_ARTWORK_TERMS):
        title = clean_text(candidate.get("title", "")).lower()
        sacred_in_title = any(term in title for term in SACRED_CATHOLIC_TERMS)

        if not sacred_in_title:
            return False

    return True


def score_artwork(candidate: Dict[str, Any], clip: Dict[str, Any], search_term: str) -> float:
    if not candidate_is_catholic_painting(candidate):
        return -100.0

    title = clean_text(candidate.get("title", "")).lower()
    artist = clean_text(candidate.get("artist", "")).lower()
    source = clean_text(candidate.get("source", "")).lower()
    term = clean_text(search_term).lower()

    clip_title = clean_text(clip.get("title", "")).lower()
    image_idea = clean_text(clip.get("image_idea", "")).lower()
    power_quote = clean_text(clip.get("power_quote", "")).lower()

    sacred_terms = [
        "christ",
        "jesus",
        "cross",
        "crucifixion",
        "virgin",
        "mary",
        "saint",
        "apostle",
        "mass",
        "eucharist",
        "chalice",
        "altar",
        "prayer",
        "angel",
        "samaritan",
        "martyr",
        "passion",
        "last supper",
        "sacrifice",
    ]

    bad_terms = [
        "modern",
        "poster",
        "photograph",
        "abstract",
        "installation",
        "fashion",
        "advertisement",
    ]

    score = 0.0

    if candidate.get("image_url"):
        score += 2.0

    if candidate.get("public_domain") or candidate.get("approved_for_use"):
        score += 3.0

    score += 3.0

    if source in {"the met", "art institute of chicago", "rijksmuseum", "national gallery of art"}:
        score += 0.7

    for word in sacred_terms:
        if word in title:
            score += 1.2
        if word in term:
            score += 0.5
        if word in image_idea:
            score += 0.6
        if word in power_quote:
            score += 0.4

    for word in re.findall(r"[a-zA-Z]+", clip_title):
        if len(word) > 4 and word in title:
            score += 0.5

    for word in re.findall(r"[a-zA-Z]+", term):
        if len(word) > 4 and word in title:
            score += 0.7

    if "mary" in clip_title and "joseph" in clip_title:
        joseph_context_terms = [
            "joseph",
            "holy family",
            "nativity",
            "flight into egypt",
            "rest on the flight",
            "marriage of the virgin",
            "adoration of the shepherds",
        ]

        if any(word in title for word in joseph_context_terms):
            score += 5.0
        else:
            score -= 12.0

    if "unknown" not in artist and artist:
        score += 0.3

    for word in bad_terms:
        if word in title:
            score -= 2.0

    subject_text = candidate_subject_text(candidate)

    if any(term in subject_text for term in SACRED_CATHOLIC_TERMS):
        score += 4.0

    if any(term in title for term in SACRED_CATHOLIC_TERMS):
        score += 3.0

    return round(score, 2)


def requests_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def requests_get_text(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text
    except Exception:
        return None


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    temp_path.replace(destination)


def catholic_tradition_cache_dir() -> Path:
    configured = os.getenv("CATHOLIC_TRADITION_CACHE_DIR", "").strip()

    if configured:
        return ensure_folder(configured)

    return ensure_folder(Path.home() / "Library" / "Caches" / "homily-shorts" / "catholic-tradition")


def is_catholic_tradition_url(url: str) -> bool:
    return urlparse(url).netloc.lower().endswith("catholictradition.org")


def is_html_page_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith((".htm", ".html"))


def is_image_file_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith((".jpg", ".jpeg", ".png"))


def clean_gallery_label(value: str, fallback: str = "Traditional Catholic Image") -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" -:\t\r\n")
    value = re.sub(r"^(image|download)\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    return value or fallback


def should_skip_catholic_tradition_item(title: str, gallery_title: str = "") -> bool:
    text = f"{title} {gallery_title}".lower()

    if not text.strip():
        return True

    return any(term in text for term in CATHOLIC_TRADITION_SKIP_TERMS)


class CatholicTraditionLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}

        if tag.lower() == "a" and attr_map.get("href"):
            self._current = {
                "url": urljoin(self.base_url, attr_map["href"]),
                "text": "",
            }
            return

        if tag.lower() == "img":
            src = attr_map.get("src", "")

            if src:
                self.links.append(
                    {
                        "url": urljoin(self.base_url, src),
                        "text": attr_map.get("alt", ""),
                    }
                )

            if self._current is not None and attr_map.get("alt"):
                self._current["text"] = clean_text(
                    f"{self._current.get('text', '')} {attr_map.get('alt', '')}"
                )

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current["text"] = clean_text(f"{self._current.get('text', '')} {data}")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current is not None:
            self.links.append(self._current)
            self._current = None


def parse_catholic_tradition_links(url: str, html: str) -> List[Dict[str, str]]:
    parser = CatholicTraditionLinkParser(url)
    parser.feed(html or "")
    return parser.links


def build_catholic_tradition_index() -> List[Dict[str, Any]]:
    index_path = catholic_tradition_cache_dir() / "catholic_tradition_images.json"

    if index_path.exists():
        try:
            cached = load_json(index_path)

            if cached.get("version") == CATHOLIC_TRADITION_INDEX_VERSION:
                return cached.get("items", [])
        except Exception:
            pass

    print("Building Catholic Tradition image index...")

    root_html = requests_get_text(CATHOLIC_TRADITION_GALLERY_URL)

    if not root_html:
        return []

    root_links = parse_catholic_tradition_links(CATHOLIC_TRADITION_GALLERY_URL, root_html)
    gallery_pages: List[Dict[str, str]] = []
    seen_pages = set()

    for link in root_links:
        url = link.get("url", "")

        if not is_catholic_tradition_url(url) or not is_html_page_url(url):
            continue

        title = clean_gallery_label(link.get("text", ""), fallback=Path(urlparse(url).path).stem)

        if should_skip_catholic_tradition_item(title) or url in seen_pages:
            continue

        gallery_pages.append({"url": url, "title": title})
        seen_pages.add(url)

        if len(gallery_pages) >= CATHOLIC_TRADITION_MAX_GALLERY_PAGES:
            break

    items: List[Dict[str, Any]] = []
    seen_images = set()

    for gallery in gallery_pages:
        gallery_url = gallery["url"]
        gallery_title = gallery["title"]
        html = requests_get_text(gallery_url)

        if not html:
            continue

        for link in parse_catholic_tradition_links(gallery_url, html):
            image_url = link.get("url", "")

            if not is_catholic_tradition_url(image_url) or not is_image_file_url(image_url):
                continue

            title = clean_gallery_label(link.get("text", ""), fallback=Path(urlparse(image_url).path).stem)

            if should_skip_catholic_tradition_item(title, gallery_title) or image_url in seen_images:
                continue

            seen_images.add(image_url)
            search_blob = clean_text(
                " ".join(
                    [
                        title,
                        gallery_title,
                        Path(urlparse(image_url).path).stem.replace("-", " "),
                    ]
                )
            ).lower()

            items.append(
                {
                    "source": "Catholic Tradition",
                    "image_id": hashlib.sha1(image_url.encode("utf-8")).hexdigest(),
                    "title": title,
                    "artist": "",
                    "date": "",
                    "medium": "traditional Catholic painting",
                    "classification": "painting",
                    "object_type": "sacred art",
                    "department": gallery_title,
                    "source_url": gallery_url,
                    "image_url": image_url,
                    "license": "Traditional Catholic image",
                    "public_domain": False,
                    "approved_for_use": True,
                    "gallery_title": gallery_title,
                    "assistive_text": search_blob,
                    "search_blob": search_blob,
                }
            )

    save_json(
        index_path,
        {
            "version": CATHOLIC_TRADITION_INDEX_VERSION,
            "created_at": time.time(),
            "source": CATHOLIC_TRADITION_GALLERY_URL,
            "items": items,
        },
    )

    return items


def nga_cache_dir() -> Path:
    configured = os.getenv("NGA_OPENDATA_CACHE_DIR", "").strip()

    if configured:
        return ensure_folder(configured)

    return ensure_folder(Path.home() / "Library" / "Caches" / "homily-shorts" / "nga-opendata")


def ensure_nga_cache_file(filename: str, url: str) -> Path:
    path = nga_cache_dir() / filename

    if path.exists() and path.stat().st_size > 0:
        return path

    print(f"Downloading NGA Open Data cache: {filename}")
    download_file(url, path)
    return path


def nga_iiif_image_url(iiif_url: str) -> str:
    iiif_url = clean_text(iiif_url).rstrip("/")

    if not iiif_url:
        return ""

    return f"{iiif_url}/full/1600,/0/default.jpg"


def text_tokens(*values: str) -> List[str]:
    stopwords = {
        "about",
        "above",
        "after",
        "again",
        "against",
        "among",
        "because",
        "before",
        "being",
        "between",
        "catholic",
        "christian",
        "clip",
        "from",
        "into",
        "painting",
        "sacred",
        "short",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "through",
        "with",
        "would",
    }
    tokens = []

    for value in values:
        for token in re.findall(r"[a-zA-Z]+", clean_text(value).lower()):
            if len(token) >= 4 and token not in stopwords:
                tokens.append(token)

    return list(dict.fromkeys(tokens))


def build_nga_index() -> List[Dict[str, Any]]:
    objects_path = ensure_nga_cache_file("objects.csv", NGA_OBJECTS_CSV_URL)
    images_path = ensure_nga_cache_file("published_images.csv", NGA_PUBLISHED_IMAGES_CSV_URL)
    index_path = nga_cache_dir() / "nga_painting_index.json"

    newest_source_mtime = max(objects_path.stat().st_mtime, images_path.stat().st_mtime)

    if index_path.exists() and index_path.stat().st_mtime >= newest_source_mtime:
        try:
            cached = load_json(index_path)

            if cached.get("version") == NGA_INDEX_VERSION:
                return cached.get("items", [])
        except Exception:
            pass

    print("Building NGA artwork search index...")

    primary_images: Dict[str, Dict[str, Any]] = {}

    with open(images_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if clean_text(row.get("openaccess", "")) != "1":
                continue

            if clean_text(row.get("viewtype", "")).lower() != "primary":
                continue

            object_id = clean_text(row.get("depictstmsobjectid", ""))
            image_url = nga_iiif_image_url(row.get("iiifurl", ""))

            if not object_id or not image_url:
                continue

            existing = primary_images.get(object_id)
            sequence = clean_text(row.get("sequence", ""))

            if existing and sequence and clean_text(existing.get("sequence", "")) <= sequence:
                continue

            primary_images[object_id] = {
                "image_id": clean_text(row.get("uuid", "")),
                "image_url": image_url,
                "thumbnail_url": clean_text(row.get("iiifthumburl", "")),
                "sequence": sequence,
                "assistive_text": clean_text(row.get("assistivetext", "")),
            }

    items: List[Dict[str, Any]] = []

    with open(objects_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            object_id = clean_text(row.get("objectid", ""))
            image = primary_images.get(object_id)

            if not image:
                continue

            candidate = {
                "source": "National Gallery of Art",
                "object_id": object_id,
                "image_id": image.get("image_id", ""),
                "title": clean_text(row.get("title", "Untitled")),
                "artist": clean_text(row.get("attribution", "Unknown artist")),
                "date": clean_text(row.get("displaydate", "")),
                "medium": clean_text(row.get("medium", "")),
                "classification": clean_text(row.get("classification", "")),
                "object_type": clean_text(row.get("subclassification", "")),
                "department": clean_text(row.get("departmentabbr", "")),
                "source_url": f"https://www.nga.gov/collection/art-object-page.{object_id}.html",
                "image_url": image.get("image_url", ""),
                "thumbnail_url": image.get("thumbnail_url", ""),
                "license": "National Gallery of Art Open Access image / CC0 collection data",
                "public_domain": True,
                "assistive_text": image.get("assistive_text", ""),
            }

            if not candidate_is_catholic_painting(candidate):
                continue

            candidate["search_blob"] = clean_text(
                " ".join(
                    [
                        candidate["title"],
                        candidate["artist"],
                        candidate["date"],
                        candidate["medium"],
                        candidate["classification"],
                        candidate["object_type"],
                        candidate["assistive_text"],
                    ]
                )
            ).lower()
            items.append(candidate)

    save_json(
        index_path,
        {
            "version": NGA_INDEX_VERSION,
            "created_at": time.time(),
            "source": "NationalGalleryOfArt/opendata",
            "items": items,
        },
    )

    return items


def search_met(term: str, clip: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
    results = []

    search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    search_data = requests_get_json(
        search_url,
        params={
            "q": term,
            "hasImages": "true",
        },
    )

    object_ids = (search_data or {}).get("objectIDs") or []

    for object_id in object_ids[:max_results]:
        object_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        obj = requests_get_json(object_url)

        if not obj:
            continue

        image_url = obj.get("primaryImage") or obj.get("primaryImageSmall")

        if not image_url:
            continue

        public_domain = bool(obj.get("isPublicDomain"))

        if not public_domain:
            continue

        candidate = {
            "source": "The Met",
            "title": clean_text(obj.get("title", "Untitled")),
            "artist": clean_text(obj.get("artistDisplayName", "Unknown artist")),
            "date": clean_text(obj.get("objectDate", "")),
            "medium": clean_text(obj.get("medium", "")),
            "classification": clean_text(obj.get("classification", "")),
            "object_type": clean_text(obj.get("objectName", "")),
            "department": clean_text(obj.get("department", "")),
            "source_url": clean_text(obj.get("objectURL", "")),
            "image_url": image_url,
            "license": "Public Domain / Open Access",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_artic(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    results = []

    search_url = "https://api.artic.edu/api/v1/artworks/search"
    data = requests_get_json(
        search_url,
        params={
            "q": term,
            "limit": max_results,
            "fields": (
                "id,title,artist_display,image_id,is_public_domain,date_display,thumbnail,"
                "medium_display,classification_titles,artwork_type_title,category_titles"
            ),
            "query[term][is_public_domain]": "true",
        },
    )

    for item in (data or {}).get("data", []):
        image_id = item.get("image_id")

        if not image_id:
            continue

        if not item.get("is_public_domain"):
            continue

        image_url = f"https://www.artic.edu/iiif/2/{image_id}/full/1600,/0/default.jpg"

        candidate = {
            "source": "Art Institute of Chicago",
            "title": clean_text(item.get("title", "Untitled")),
            "artist": clean_text(item.get("artist_display", "Unknown artist")),
            "date": clean_text(item.get("date_display", "")),
            "medium": clean_text(item.get("medium_display", "")),
            "classification": clean_text(", ".join(item.get("classification_titles") or [])),
            "object_type": clean_text(item.get("artwork_type_title", "")),
            "department": clean_text(", ".join(item.get("category_titles") or [])),
            "source_url": f"https://www.artic.edu/artworks/{item.get('id')}",
            "image_url": image_url,
            "license": "Public Domain / CC0 when marked public domain",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_rijksmuseum(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Requires RIJKSMUSEUM_API_KEY in .env.

    Example:
    RIJKSMUSEUM_API_KEY=your_key_here
    """

    api_key = os.getenv("RIJKSMUSEUM_API_KEY", "").strip()

    if not api_key:
        return []

    results = []

    search_url = "https://www.rijksmuseum.nl/api/en/collection"
    data = requests_get_json(
        search_url,
        params={
            "key": api_key,
            "q": term,
            "imgonly": "True",
            "ps": max_results,
            "format": "json",
        },
    )

    for item in (data or {}).get("artObjects", []):
        web_image = item.get("webImage") or {}
        image_url = web_image.get("url")

        if not image_url:
            continue

        candidate = {
            "source": "Rijksmuseum",
            "title": clean_text(item.get("title", "Untitled")),
            "artist": clean_text(item.get("principalOrFirstMaker", "Unknown artist")),
            "date": "",
            "medium": "",
            "classification": clean_text(", ".join(item.get("classification") or [])),
            "object_type": clean_text(", ".join(item.get("objectTypes") or [])),
            "department": "",
            "source_url": clean_text(item.get("links", {}).get("web", "")),
            "image_url": image_url,
            "license": "Rijksmuseum image/open data. Verify object rights if needed.",
            "public_domain": True,
            "search_term": term,
        }

        candidate["score"] = score_artwork(candidate, clip, term)
        results.append(candidate)

    return results


def search_nga(term: str, clip: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
    try:
        index = build_nga_index()
    except Exception as exc:
        print(f"NGA Open Data search unavailable: {exc}")
        return []

    tokens = text_tokens(
        term,
        clip.get("title", ""),
        clip.get("main_idea", ""),
        clip.get("image_idea", ""),
        clip.get("power_quote", ""),
    )
    term_tokens = set(text_tokens(term))
    results = []

    for item in index:
        blob = clean_text(item.get("search_blob", "")).lower()

        if tokens and not any(token in blob for token in tokens):
            continue

        candidate = dict(item)
        title = clean_text(candidate.get("title", "")).lower()
        token_hits = sum(1 for token in tokens if token in blob)
        title_hits = sum(1 for token in term_tokens if token in title)

        candidate["search_term"] = term
        candidate["score"] = round(
            score_artwork(candidate, clip, term)
            + min(4.0, token_hits * 0.45)
            + min(2.5, title_hits * 0.75),
            2,
        )
        candidate.pop("search_blob", None)
        results.append(candidate)

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


def search_catholic_tradition(term: str, clip: Dict[str, Any], max_results: int = 12) -> List[Dict[str, Any]]:
    try:
        index = build_catholic_tradition_index()
    except Exception as exc:
        print(f"Catholic Tradition image search unavailable: {exc}")
        return []

    tokens = text_tokens(
        term,
        clip.get("title", ""),
        clip.get("main_idea", ""),
        clip.get("image_idea", ""),
        clip.get("power_quote", ""),
    )
    term_tokens = set(text_tokens(term))
    results = []

    for item in index:
        blob = clean_text(item.get("search_blob", "")).lower()

        if tokens and not any(token in blob for token in tokens):
            continue

        candidate = dict(item)
        title = clean_text(candidate.get("title", "")).lower()
        gallery_title = clean_text(candidate.get("gallery_title", "")).lower()
        token_hits = sum(1 for token in tokens if token in blob)
        title_hits = sum(1 for token in term_tokens if token in title)
        gallery_hits = sum(1 for token in tokens if token in gallery_title)

        candidate["search_term"] = term
        candidate["score"] = round(
            score_artwork(candidate, clip, term)
            + 5.0
            + min(6.0, token_hits * 0.65)
            + min(3.0, title_hits * 0.9)
            + min(3.0, gallery_hits * 0.9),
            2,
        )
        candidate.pop("search_blob", None)
        results.append(candidate)

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


def search_public_domain_art_for_clip(
    clip: Dict[str, Any],
    used_artwork_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    all_candidates = []
    search_terms = build_search_terms(clip)

    for term in search_terms:
        all_candidates.extend(search_catholic_tradition(term, clip))
        all_candidates.extend(search_nga(term, clip))
        all_candidates.extend(search_met(term, clip))
        all_candidates.extend(search_artic(term, clip))
        all_candidates.extend(search_rijksmuseum(term, clip))

        all_candidates = [c for c in all_candidates if candidate_is_catholic_painting(c)]
        if used_artwork_keys is not None:
            all_candidates = [
                c
                for c in all_candidates
                if not artwork_identity(c) or artwork_identity(c) not in used_artwork_keys
            ]

        strong_candidates = [
            c for c in all_candidates if c.get("score", 0) >= MIN_PUBLIC_DOMAIN_SCORE
        ]

        if strong_candidates:
            break

    all_candidates.sort(key=lambda item: item.get("score", 0), reverse=True)

    return {
        "clip_id": clip.get("id"),
        "clip_title": clip.get("title"),
        "search_terms": search_terms,
        "candidates": all_candidates[:20],
        "best_candidate": all_candidates[0] if all_candidates else None,
    }


def download_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    urls = [url]

    if is_catholic_tradition_url(url):
        headers["Referer"] = CATHOLIC_TRADITION_GALLERY_URL

        if url.startswith("http://"):
            urls.insert(0, "https://" + url[len("http://"):])
        elif url.startswith("https://"):
            urls.append("http://" + url[len("https://"):])

    last_exc: Optional[Exception] = None

    for candidate_url in dict.fromkeys(urls):
        try:
            response = requests.get(candidate_url, timeout=REQUEST_TIMEOUT, headers=headers)
            response.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Could not download image: {url}")

    try:
        image = Image.open(BytesIO(response.content))
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Downloaded image was not readable: {url}") from exc

    return ImageOps.exif_transpose(image).convert("RGB")


def center_crop_to_ratio(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    current_ratio = width / height

    if current_ratio > ratio:
        new_width = int(height * ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))

    new_height = int(width / ratio)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def save_cropped_image(image: Image.Image, output_path: Path) -> Path:
    image = center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92, optimize=True, progressive=True)

    return output_path


def find_font(bold: bool = True, serif: bool = False) -> str:
    serif_candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]

    sans_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    candidates = serif_candidates if serif else sans_candidates

    if not bold:
        candidates = [p for p in candidates if "Bold" not in p] + candidates

    for path in candidates:
        if os.path.exists(path):
            return path

    raise RuntimeError("Could not find a usable TrueType font.")


def load_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(find_font(bold=bold, serif=serif), size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = re.split(r"\s+", str(text or "").strip())
    lines = []
    line = ""

    for word in words:
        trial = f"{line} {word}".strip()
        bbox = font.getbbox(trial)

        if bbox[2] - bbox[0] <= max_width:
            line = trial
            continue

        if line:
            lines.append(line)

        line = word

    if line:
        lines.append(line)

    return lines


def normalize_caption_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    return clean_text(text)


def make_placeholder_image(clip: Dict[str, Any], output_path: Path) -> Path:
    image = Image.new("RGB", IMAGE_SIZE, (18, 18, 18))
    draw = ImageDraw.Draw(image)

    title_font = load_font(58, bold=True, serif=True)
    small_font = load_font(30, bold=False, serif=False)

    title = clean_text(clip.get("title", "Catholic Short")).upper()
    image_idea = clean_text(clip.get("image_idea", ""))

    lines = wrap_text(title, title_font, IMAGE_SIZE[0] - 120)[:4]

    y = 220

    for line in lines:
        bbox = title_font.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x + 2, y + 2), line, font=title_font, fill=(0, 0, 0))
        draw.text((x, y), line, font=title_font, fill=(235, 235, 235))
        y += 80

    small_lines = wrap_text(image_idea or "No image found", small_font, IMAGE_SIZE[0] - 160)[:4]
    y = IMAGE_SIZE[1] - 310

    for line in small_lines:
        bbox = small_font.getbbox(line)
        x = (IMAGE_SIZE[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=small_font, fill=(180, 180, 180))
        y += 42

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)

    return output_path


def generate_ai_image_for_clip(clip: Dict[str, Any], output_path: Path) -> Optional[Path]:
    """
    Optional AI image fallback.

    Requires:
    OPENAI_API_KEY

    This function uses the current OpenAI Python client image API.
    If generation fails, it returns None and the app creates a placeholder.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = (
            "Create a 4:5 vertical image for a Catholic YouTube Short.\n\n"
            f"Title: {clip.get('title', '')}\n"
            f"Image idea: {clip.get('image_idea', '')}\n"
            f"Power quote: {clip.get('power_quote', '')}\n\n"
            "Create this as a painted sacred-art image only, not a sculpture, statue, object, artifact, photo, or museum display. "
            "Use light watercolor with ink details, reverent sacred art tone, cinematic composition. "
            f"{TRADITIONAL_CATHOLIC_IMAGE_GUARDRAIL} "
            "No text, no typography, no subtitles, no logo, no watermark."
        )

        model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=os.getenv("OPENAI_SHORTS_IMAGE_SIZE", "1024x1536"),
            n=1,
        )

        item = response.data[0]

        if getattr(item, "b64_json", None):
            import base64

            image_bytes = base64.b64decode(item.b64_json)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return save_cropped_image(image, output_path)

        if getattr(item, "url", None):
            image = download_image(item.url)
            return save_cropped_image(image, output_path)

        return None

    except Exception as exc:
        print(f"AI image generation failed for clip {clip.get('id')}: {exc}")
        return None


def ensure_image_for_clip(
    clip: Dict[str, Any],
    paths: Dict[str, Path],
    force_image: bool = False,
    allow_ai_fallback: bool = True,
    used_artwork_keys: Optional[set[str]] = None,
    artwork_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    clip_paths = get_clip_paths(clip, paths)
    image_path = clip_paths["image"]
    image_meta_path = clip_paths["image_meta"]

    if image_path.exists() and image_meta_path.exists() and not force_image:
        existing_meta = load_json(image_meta_path)
        existing_key = existing_artwork_key_from_meta(existing_meta)
        reserved = reserve_existing_artwork_key(
            existing_key,
            used_artwork_keys,
            artwork_lock,
        )

        if reserved:
            if existing_key and not existing_meta.get("artwork_key"):
                existing_meta["artwork_key"] = existing_key
                save_json(image_meta_path, existing_meta)

            return existing_meta

        print(
            f"Existing image for clip {clip.get('id')} duplicates another selected clip; finding a new one."
        )

    print(f"Finding image for clip {clip.get('id')}: {clip.get('title')}")

    image_search_result = search_public_domain_art_for_clip(
        clip,
        used_artwork_keys=used_artwork_keys,
    )
    candidates = [
        candidate
        for candidate in image_search_result.get("candidates", [])
        if candidate.get("score", 0) >= MIN_PUBLIC_DOMAIN_SCORE
    ]

    for candidate in candidates:
        reserved, artwork_key = reserve_artwork_key(candidate, used_artwork_keys, artwork_lock)

        if not reserved:
            continue

        try:
            image = download_image(candidate["image_url"])
            save_cropped_image(image, image_path)

            meta = {
                "clip_id": clip.get("id"),
                "clip_title": clip.get("title"),
                "image_path": str(image_path),
                "image_source_type": (
                    "catholic_tradition_gallery"
                    if clean_text(candidate.get("source", "")).lower() == "catholic tradition"
                    else "public_domain"
                ),
                "artwork": candidate,
                "artwork_key": artwork_key,
                "all_candidates": image_search_result.get("candidates", []),
                "created_at": time.time(),
            }

            save_json(image_meta_path, meta)
            return meta

        except Exception as exc:
            source = clean_text(candidate.get("source", "artwork"))
            print(f"Artwork image failed for clip {clip.get('id')} ({source}): {exc}")
            release_artwork_key(artwork_key, used_artwork_keys, artwork_lock)

    if allow_ai_fallback:
        generated = generate_ai_image_for_clip(clip, image_path)

        if generated:
            meta = {
                "clip_id": clip.get("id"),
                "clip_title": clip.get("title"),
                "image_path": str(generated),
                "image_source_type": "ai_generated",
                "artwork": None,
                "artwork_key": "",
                "all_candidates": image_search_result.get("candidates", []),
                "created_at": time.time(),
            }

            save_json(image_meta_path, meta)
            return meta

    make_placeholder_image(clip, image_path)

    meta = {
        "clip_id": clip.get("id"),
        "clip_title": clip.get("title"),
        "image_path": str(image_path),
        "image_source_type": "placeholder",
        "artwork": None,
        "artwork_key": "",
        "all_candidates": image_search_result.get("candidates", []),
        "created_at": time.time(),
    }

    save_json(image_meta_path, meta)
    return meta


def make_background(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = center_crop_to_ratio(image, IMAGE_RATIO)
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", CANVAS_SIZE, "black")
    canvas.paste(image, (0, IMAGE_TOP))

    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gradient_height = GRADIENT_BOTTOM - GRADIENT_TOP

    for offset in range(gradient_height):
        alpha = int(GRADIENT_MAX_ALPHA * ((offset + 1) / gradient_height))
        y = GRADIENT_TOP + offset
        draw.line([(0, y), (CANVAS_SIZE[0], y)], fill=(0, 0, 0, alpha))

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def text_overlay(text: str, title: bool = False) -> Image.Image:
    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = 54
    font = load_font(font_size, bold=True, serif=False)

    safe_left = TEXT_SAFE_LEFT
    safe_right = CANVAS_SIZE[0] - TEXT_SAFE_RIGHT
    safe_top = TEXT_SAFE_TOP
    safe_bottom = CANVAS_SIZE[1] - TEXT_SAFE_BOTTOM
    safe_width = safe_right - safe_left
    safe_height = safe_bottom - safe_top

    max_lines = CAPTION_MAX_LINES
    lines = wrap_text(text, font, safe_width)[:max_lines]

    def metrics() -> Tuple[int, List[int], List[int]]:
        gap = int(font_size * 0.32)
        widths = []
        heights = []

        for line in lines:
            bbox = font.getbbox(line)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])

        return sum(heights) + gap * max(0, len(lines) - 1), widths, heights

    total_height, widths, heights = metrics()

    while (len(wrap_text(text, font, safe_width)) > max_lines or total_height > safe_height) and font_size > 36:
        font_size -= 4
        font = load_font(font_size, bold=True, serif=False)
        lines = wrap_text(text, font, safe_width)[:max_lines]
        total_height, widths, heights = metrics()

    line_gap = int(font_size * 0.32)
    y = safe_top + max(0, (safe_height - total_height) // 2)

    fill = (255, 255, 255, 255)
    shadow = (0, 0, 0, 220)

    for line, width, height in zip(lines, widths, heights):
        x = safe_left + (safe_width - width) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_gap

    return overlay


def cut_audio_clip(source_audio: str | Path, start: float, end: float, output_path: Path) -> Path:
    source_audio = Path(source_audio).expanduser().resolve()

    if not source_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {source_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(source_audio)
    start_ms = int(float(start) * 1000)
    end_ms = int(float(end) * 1000)

    audio[start_ms:end_ms].export(output_path, format="mp3")

    return output_path


def get_audio_duration_seconds(source_audio: str | Path) -> float:
    source_audio = Path(source_audio).expanduser().resolve()

    if not source_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {source_audio}")

    return len(AudioSegment.from_file(source_audio)) / 1000.0


def validate_clips_against_audio_duration(
    clips: List[Dict[str, Any]],
    source_audio: str | Path,
    tolerance_seconds: float = 0.25,
) -> None:
    audio_duration = get_audio_duration_seconds(source_audio)
    invalid = []

    for clip in clips:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", start))

        if start >= audio_duration - tolerance_seconds:
            invalid.append(
                f"{clip.get('id')}. {clip.get('title')} starts at {format_timestamp(start)}, "
                f"but audio ends at {format_timestamp(audio_duration)}"
            )
            continue

        if end > audio_duration + tolerance_seconds:
            invalid.append(
                f"{clip.get('id')}. {clip.get('title')} ends at {format_timestamp(end)}, "
                f"but audio ends at {format_timestamp(audio_duration)}"
            )

    if invalid:
        details = "\n".join(f"- {item}" for item in invalid)
        raise ValueError(
            "Selected clip range(s) exceed the source audio duration. "
            "Use a clip table that matches this exact homily audio, or select only valid rows.\n"
            f"{details}"
        )


def list_background_audio_files(bg_audio_dir: str | Path) -> List[Path]:
    bg_audio_dir = Path(bg_audio_dir).expanduser().resolve()

    if not bg_audio_dir.exists() or not bg_audio_dir.is_dir():
        return []

    files = []

    for path in bg_audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() in BACKGROUND_AUDIO_EXTENSIONS:
            files.append(path)

    return sorted(files)


def loop_audio_to_duration(audio: AudioSegment, duration_ms: int) -> AudioSegment:
    if len(audio) <= 0:
        raise RuntimeError("Background audio file has no duration.")

    if len(audio) >= duration_ms:
        return audio[:duration_ms]

    repeats = (duration_ms // len(audio)) + 1
    return (audio * repeats)[:duration_ms]


def mix_background_music(
    speech_path: Path,
    bg_audio_files: List[Path],
    gain_db: float,
    fade_ms: int,
) -> Optional[Path]:
    if not bg_audio_files:
        return None

    bg_path = random.choice(bg_audio_files)

    speech = AudioSegment.from_file(speech_path)
    duration_ms = len(speech)

    if duration_ms <= 0:
        return None

    music = AudioSegment.from_file(bg_path)
    music = loop_audio_to_duration(music, duration_ms)
    music = music.set_frame_rate(speech.frame_rate).set_channels(speech.channels)
    music = music + float(gain_db)

    fade_ms = max(0, min(int(fade_ms), duration_ms // 2))

    if fade_ms:
        music = music.fade_in(fade_ms).fade_out(fade_ms)

    mixed = speech.overlay(music)
    mixed.export(speech_path, format="mp3")

    return bg_path


def resolve_timed_transcript_path(paths: Dict[str, Path]) -> Optional[Path]:
    video_clips_folder = paths["video_clips_folder"]
    homily_folder = video_clips_folder.parent

    candidates = [
        homily_folder / "working" / "video_script.json",
        homily_folder / "working" / "homily.json",
        homily_folder / "video_script.json",
        homily_folder / "homily.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_timed_words(paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    transcript_path = resolve_timed_transcript_path(paths)

    if not transcript_path:
        return []

    try:
        data = load_json(transcript_path)
    except Exception:
        return []

    words: List[Dict[str, Any]] = []

    for segment in data.get("homily_segments") or data.get("segments") or []:
        segment_start = float(segment.get("start", 0.0) or 0.0)

        for raw_word in segment.get("words") or []:
            text = clean_text(raw_word.get("word", ""))

            if not text:
                continue

            try:
                start = float(raw_word.get("start"))
                end = float(raw_word.get("end"))
            except (TypeError, ValueError):
                continue

            # Some transcribers store word offsets relative to the segment.
            if start < segment_start - 1.0 and segment_start > 0:
                start += segment_start
                end += segment_start

            if end <= start:
                continue

            words.append(
                {
                    "word": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def words_for_clip(words: List[Dict[str, Any]], clip: Dict[str, Any]) -> List[Dict[str, Any]]:
    start = float(clip["start"])
    end = float(clip["end"])

    return [
        word
        for word in words
        if float(word["end"]) > start and float(word["start"]) < end
    ]


def words_to_caption_text(words: List[Dict[str, Any]]) -> str:
    return normalize_caption_text(" ".join(clean_text(word.get("word", "")) for word in words))


def create_word_caption_groups(
    clip: Dict[str, Any],
    timed_words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    clip_words = words_for_clip(timed_words, clip)

    if not clip_words:
        return []

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    groups: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return

        groups.append(
            {
                "start": round(max(clip_start, float(current[0]["start"])), 3),
                "end": round(min(clip_end, float(current[-1]["end"])), 3),
                "text": words_to_caption_text(current),
            }
        )
        current.clear()

    for word in clip_words:
        word_text = clean_text(word.get("word", ""))

        if not word_text:
            continue

        previous = current[-1] if current else None
        current_text = words_to_caption_text(current) if current else ""
        next_text = normalize_caption_text(f"{current_text} {word_text}".strip())
        gap = float(word["start"]) - float(previous["end"]) if previous else 0.0
        current_duration = float(previous["end"]) - float(current[0]["start"]) if previous else 0.0
        previous_ends_sentence = bool(previous and re.search(r"[.!?][\"')\]]?$", clean_text(previous["word"])))

        should_flush = bool(
            current
            and (
                len(current) >= CAPTION_MAX_WORDS
                or len(next_text) > CAPTION_MAX_CHARS
                or gap > 0.65
                or (previous_ends_sentence and current_duration >= 0.8)
            )
        )

        if should_flush:
            flush()

        current.append(word)

    flush()

    return [
        group
        for group in groups
        if group.get("text") and float(group["end"]) > float(group["start"])
    ]


def create_caption_groups(
    clip: Dict[str, Any],
    timed_words: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if timed_words:
        groups = create_word_caption_groups(clip, timed_words)

        if groups:
            return groups

    return create_basic_caption_groups(clip)


def create_basic_caption_groups(clip: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simple captions using source_text.
    Fallback only; word-level captions are preferred when transcript timings exist.
    """

    start = float(clip["start"])
    end = float(clip["end"])
    duration = end - start

    text = clean_text(clip.get("source_text", ""))

    if not text:
        text = clean_text(clip.get("power_quote", ""))

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [clean_text(s) for s in sentences if clean_text(s)]

    if not sentences:
        return []

    groups = []
    current_time = start
    seconds_per_group = max(2.0, min(4.0, duration / max(1, len(sentences))))

    for sentence in sentences:
        group_end = min(end, current_time + seconds_per_group)

        groups.append(
            {
                "start": round(current_time, 3),
                "end": round(group_end, 3),
                "text": sentence[:120],
            }
        )

        current_time = group_end

        if current_time >= end:
            break

    return groups


def write_srt(
    clip: Dict[str, Any],
    output_path: Path,
    timed_words: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    groups = clip.get("caption_groups") or create_caption_groups(clip, timed_words)

    lines = []
    clip_start = float(clip["start"])

    for index, group in enumerate(groups, 1):
        text = clean_text(group.get("text", ""))

        if not text:
            continue

        start = max(0.0, float(group["start"]) - clip_start)
        end = max(start + 0.2, float(group["end"]) - clip_start)

        lines.extend(
            [
                str(index),
                f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}",
                text,
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return output_path


def render_video_clip(
    clip: Dict[str, Any],
    source_audio: str | Path,
    paths: Dict[str, Path],
    timed_words: Optional[List[Dict[str, Any]]] = None,
    bg_audio_files: Optional[List[Path]] = None,
    bg_music_gain_db: float = DEFAULT_BG_MUSIC_GAIN_DB,
    bg_music_fade_ms: int = DEFAULT_BG_MUSIC_FADE_MS,
    force: bool = False,
) -> Dict[str, Any]:
    clip_paths = get_clip_paths(clip, paths)

    image_path = clip_paths["image"]
    audio_path = clip_paths["audio"]
    video_path = clip_paths["video"]
    captions_path = clip_paths["captions"]

    if video_path.exists() and not force:
        clip["output_paths"] = {k: str(v) for k, v in clip_paths.items()}
        return clip

    if not image_path.exists():
        raise FileNotFoundError(f"Image missing for clip {clip.get('id')}: {image_path}")

    print(f"Rendering clip {clip.get('id')}: {clip.get('title')}")
    duration = float(clip["end"]) - float(clip["start"])

    bg_audio_path = None

    if bg_audio_files and duration <= MAX_BG_MUSIC_SECONDS:
        cut_audio_clip(
            source_audio=source_audio,
            start=float(clip["start"]),
            end=float(clip["end"]),
            output_path=audio_path,
        )
        bg_audio_path = mix_background_music(
            speech_path=audio_path,
            bg_audio_files=bg_audio_files,
            gain_db=bg_music_gain_db,
            fade_ms=bg_music_fade_ms,
        )
    elif bg_audio_files:
        print(
            f"Skipping background music for clip {clip.get('id')} "
            f"because duration is {duration:.2f}s."
        )

    clip["caption_groups"] = create_caption_groups(clip, timed_words)
    write_srt(clip, captions_path, timed_words=timed_words)

    background = np.array(make_background(image_path))
    layers = [ImageClip(background, duration=duration)]

    clip_start = float(clip["start"])

    for group in clip.get("caption_groups", []):
        text = clean_text(group.get("text", ""))

        if not text:
            continue

        rel_start = max(0.0, float(group["start"]) - clip_start)
        rel_end = min(duration, max(rel_start + 0.3, float(group["end"]) - clip_start))

        caption_img = np.array(text_overlay(text, title=False))

        layers.append(
            ImageClip(caption_img, duration=rel_end - rel_start).with_start(rel_start)
        )

    if bg_audio_path:
        audio = AudioFileClip(str(audio_path))
    else:
        audio = AudioFileClip(str(source_audio)).subclipped(float(clip["start"]), float(clip["end"]))

    video = CompositeVideoClip(layers, size=CANVAS_SIZE).with_audio(audio).with_duration(duration)

    video_path.parent.mkdir(parents=True, exist_ok=True)

    video.write_videofile(
        str(video_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        threads=4,
        logger="bar",
    )

    audio.close()
    video.close()

    clip["background_audio"] = str(bg_audio_path) if bg_audio_path else ""
    clip["output_paths"] = {k: str(v) for k, v in clip_paths.items()}

    return clip


def write_manifest_and_metadata(
    selected_clips: List[Dict[str, Any]],
    image_meta_by_clip_id: Dict[int, Dict[str, Any]],
    paths: Dict[str, Path],
    source_audio: str | Path,
) -> None:
    manifest = {
        "version": 1,
        "source_audio": str(Path(source_audio).expanduser().resolve()),
        "clips": selected_clips,
    }

    upload_metadata = {
        "clips": [],
    }

    artwork_credits = {
        "clips": [],
    }

    for clip in selected_clips:
        clip_id = int(clip["id"])
        image_meta = image_meta_by_clip_id.get(clip_id, {})
        artwork = image_meta.get("artwork")
        artwork_source = clean_text((artwork or {}).get("source", ""))
        suppress_credit = artwork_source.lower() == "catholic tradition"

        upload_metadata["clips"].append(
            {
                "id": clip_id,
                "title": clip.get("title", ""),
                "description": clip.get("why_it_works", ""),
                "power_quote": clip.get("power_quote", ""),
                "tags": [
                    "Catholic homily",
                    "Traditional Catholic",
                    "Catholic Shorts",
                    "Latin Mass",
                ],
                "start": clip.get("start"),
                "end": clip.get("end"),
                "duration": clip.get("length_seconds"),
                "video_path": clip.get("output_paths", {}).get("video", ""),
                "thumbnail_path": clip.get("output_paths", {}).get("image", ""),
                "captions_path": clip.get("output_paths", {}).get("captions", ""),
            }
        )

        artwork_credits["clips"].append(
            {
                "id": clip_id,
                "clip_title": clip.get("title", ""),
                "image_source_type": image_meta.get("image_source_type", ""),
                "artwork_title": (artwork or {}).get("title", ""),
                "artist": "" if suppress_credit else (artwork or {}).get("artist", ""),
                "source": "" if suppress_credit else artwork_source,
                "source_url": "" if suppress_credit else (artwork or {}).get("source_url", ""),
                "license": "" if suppress_credit else (artwork or {}).get("license", ""),
                "image_path": image_meta.get("image_path", ""),
                "artwork_key": image_meta.get("artwork_key", ""),
            }
        )

    save_json(paths["shorts_manifest"], manifest)
    save_json(paths["upload_metadata"], upload_metadata)
    save_json(paths["artwork_credits"], artwork_credits)

    print(f"Saved manifest: {paths['shorts_manifest']}")
    print(f"Saved upload metadata: {paths['upload_metadata']}")
    print(f"Saved artwork credits: {paths['artwork_credits']}")


def run_step_04_render_with_images(
    shorts_analysis_path: str | Path,
    source_audio: str | Path,
    bg_audio_dir: Optional[str | Path] = None,
    force_images: bool = False,
    force_render: bool = False,
    allow_ai_fallback: bool = True,
    max_workers: int = 3,
    clip_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Main Step #4 function.

    Flow:
    1. Load selected clips from shorts_analysis.json.
    2. Get first image first.
    3. Start rendering first video.
    4. While first video renders, search/generate images for the rest.
    5. Render remaining clips.
    6. Save manifest, metadata, and artwork credits.
    """

    load_dotenv()

    paths = resolve_output_paths(shorts_analysis_path)
    analysis = load_json(paths["shorts_analysis"])
    selected_clips = get_selected_clips(analysis)

    if clip_ids:
        wanted = {int(clip_id) for clip_id in clip_ids}
        selected_clips = [clip for clip in selected_clips if int(clip.get("id", 0)) in wanted]
        found = {int(clip.get("id", 0)) for clip in selected_clips}
        missing = sorted(wanted - found)

        if missing:
            raise ValueError(f"Requested clip IDs are not selected in shorts_analysis.json: {missing}")

        if not selected_clips:
            raise ValueError("No selected clips match the requested --clip-id values.")

    validate_clips_against_audio_duration(selected_clips, source_audio)

    timed_words = load_timed_words(paths)

    image_meta_by_clip_id: Dict[int, Dict[str, Any]] = {}
    selected_clip_ids = {int(clip.get("id", 0)) for clip in selected_clips}
    used_artwork_keys = load_existing_artwork_keys_for_other_clips(paths, selected_clip_ids)
    artwork_lock = threading.Lock()

    bg_audio_files = []

    if bg_audio_dir:
        bg_audio_files = list_background_audio_files(bg_audio_dir)

    first_clip = selected_clips[0]
    remaining_clips = selected_clips[1:]

    print()
    print("Step #4: Render selected Shorts with images")
    print("-" * 80)
    print(f"Selected clips: {len(selected_clips)}")
    print(f"First clip: {first_clip.get('id')} - {first_clip.get('title')}")
    print(f"Timed transcript words: {len(timed_words)}")
    print("-" * 80)
    print()

    first_meta = ensure_image_for_clip(
        clip=first_clip,
        paths=paths,
        force_image=force_images,
        allow_ai_fallback=allow_ai_fallback,
        used_artwork_keys=used_artwork_keys,
        artwork_lock=artwork_lock,
    )

    image_meta_by_clip_id[int(first_clip["id"])] = first_meta

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_futures = {
            executor.submit(
                ensure_image_for_clip,
                clip,
                paths,
                force_images,
                allow_ai_fallback,
                used_artwork_keys,
                artwork_lock,
            ): clip
            for clip in remaining_clips
        }

        first_clip = render_video_clip(
            clip=first_clip,
            source_audio=source_audio,
            paths=paths,
            timed_words=timed_words,
            bg_audio_files=bg_audio_files,
            force=force_render,
        )

        for future in concurrent.futures.as_completed(image_futures):
            clip = image_futures[future]

            try:
                meta = future.result()
                image_meta_by_clip_id[int(clip["id"])] = meta
                print(f"Image ready for clip {clip.get('id')}: {clip.get('title')}")
            except Exception as exc:
                print(f"Image failed for clip {clip.get('id')}: {exc}")
                meta = {
                    "clip_id": clip.get("id"),
                    "clip_title": clip.get("title"),
                    "image_source_type": "failed",
                    "image_path": "",
                    "artwork": None,
                }
                image_meta_by_clip_id[int(clip["id"])] = meta

    rendered_clips = [first_clip]

    for clip in remaining_clips:
        rendered = render_video_clip(
            clip=clip,
            source_audio=source_audio,
            paths=paths,
            timed_words=timed_words,
            bg_audio_files=bg_audio_files,
            force=force_render,
        )
        rendered_clips.append(rendered)

    write_manifest_and_metadata(
        selected_clips=rendered_clips,
        image_meta_by_clip_id=image_meta_by_clip_id,
        paths=paths,
        source_audio=source_audio,
    )

    print()
    print("Finished Step #4")
    print("-" * 80)

    for clip in rendered_clips:
        print(f"{clip.get('id')}. {clip.get('title')} -> {clip.get('output_paths', {}).get('video', '')}")

    print("-" * 80)
    print()

    return rendered_clips


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step #4: Render selected Shorts with public-domain images first.")
    parser.add_argument(
        "shorts_analysis",
        help="Path to Video Clips/shorts_analysis.json or the Video Clips folder.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the full homily audio file.",
    )
    parser.add_argument(
        "--bg-audio-dir",
        default=None,
        help="Optional folder containing background music files.",
    )
    parser.add_argument(
        "--force-images",
        action="store_true",
        help="Force image search/generation even if image files already exist.",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Force video rendering even if video files already exist.",
    )
    parser.add_argument(
        "--no-ai-fallback",
        action="store_true",
        help="Do not generate AI images if public-domain art is not found.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of parallel image lookup/generation workers.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help="Render only specific selected clip IDs. Accepts repeated values or comma lists, e.g. --clip-id 6 --clip-id 8,9.",
    )

    args = parser.parse_args()

    cli_clip_ids = []
    for value in args.clip_id:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                cli_clip_ids.append(int(piece))

    run_step_04_render_with_images(
        shorts_analysis_path=args.shorts_analysis,
        source_audio=args.audio,
        bg_audio_dir=args.bg_audio_dir,
        force_images=args.force_images,
        force_render=args.force_render,
        allow_ai_fallback=not args.no_ai_fallback,
        max_workers=args.max_workers,
        clip_ids=cli_clip_ids or None,
    )
