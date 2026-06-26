import argparse
import base64
import inspect
import os
from io import BytesIO

import requests
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


MAX_THUMBNAIL_BYTES = 1_900_000
THUMBNAIL_SIZE = (1280, 720)
DEFAULT_IMAGE_MODEL = "gpt-image-1.5"
THUMBNAIL_TEXT_COLOR = (255, 255, 255, 255)
THUMBNAIL_TEXT_SHADOW = (0, 0, 0, 230)
THUMBNAIL_ACCENT_COLOR = (206, 24, 32, 255)


def build_thumbnail_prompt(title, idea=None):
    subject = str(idea or title or "").strip()
    if not subject:
        raise ValueError("A title or thumbnail idea is required to generate a thumbnail.")

    return (
        "Create a YouTube thumbnail for a speech about\n\n"
        f"- {subject}.\n\n"
        "- Make it in the style of light watercolor with ink accents in landscape format.\n\n"
        "Use a reverent Catholic visual tone. Do not include text, captions, logos, "
        "watermarks, or typography."
    )


def _openai_client():
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or OPENAI_KEY must be set before generating a thumbnail.")
    return OpenAI(api_key=api_key)


def _image_bytes_from_response(response):
    if not response.data:
        raise RuntimeError("OpenAI image generation returned no images.")

    image_result = response.data[0]
    image_base64 = getattr(image_result, "b64_json", None)
    if image_base64:
        return base64.b64decode(image_base64)

    image_url = getattr(image_result, "url", None)
    if image_url:
        download = requests.get(image_url, timeout=60)
        download.raise_for_status()
        return download.content

    raise RuntimeError("OpenAI image generation returned no image data.")


def _supported_image_kwargs(generate_method, kwargs):
    parameters = inspect.signature(generate_method).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _generate_image_with_compatible_kwargs(client, image_kwargs):
    retryable_params = {
        "response_format",
        "output_format",
        "output_compression",
        "background",
        "moderation",
        "quality",
        "style",
    }
    cleaned_kwargs = dict(image_kwargs)

    while True:
        try:
            return client.images.generate(**cleaned_kwargs)
        except BadRequestError as exc:
            error = getattr(exc, "body", {}).get("error", {}) if getattr(exc, "body", None) else {}
            param = error.get("param")
            message = str(error.get("message") or exc)
            if param in cleaned_kwargs and (param in retryable_params or "Unknown parameter" in message):
                print(f"OpenAI Images rejected '{param}'; retrying without it.")
                cleaned_kwargs.pop(param)
                continue
            raise


def _center_crop_to_ratio(image, target_ratio):
    width, height = image.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))

    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def find_font(bold=True, serif=True):
    serif_candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
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
        candidates = [path for path in candidates if "Bold" not in path] + candidates

    for path in candidates:
        if os.path.exists(path):
            return path

    raise RuntimeError("Could not find a usable font for thumbnail text.")


def load_font(size, bold=True, serif=True):
    return ImageFont.truetype(find_font(bold=bold, serif=serif), size)


def wrap_text(text, font, max_width):
    words = str(text or "").strip().split()
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


def text_size(font, text):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text_strict(text, font, max_width):
    lines = []

    for word in str(text or "").strip().split():
        if text_size(font, word)[0] <= max_width:
            pending_words = [word]
        else:
            pending_words = []
            chunk = ""
            for char in word:
                trial = chunk + char
                if chunk and text_size(font, trial)[0] > max_width:
                    pending_words.append(chunk)
                    chunk = char
                else:
                    chunk = trial
            if chunk:
                pending_words.append(chunk)

        for item in pending_words:
            trial = f"{lines[-1]} {item}".strip() if lines else item
            if lines and text_size(font, trial)[0] <= max_width:
                lines[-1] = trial
            else:
                lines.append(item)

    return lines


def clean_thumbnail_text(text):
    text = " ".join(str(text or "").split()).strip()
    return text.upper()


def apply_thumbnail_text_overlay(image, text):
    text = clean_thumbnail_text(text)
    if not text:
        return image

    image = image.convert("RGBA")
    width, height = image.size

    gradient = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_gradient = ImageDraw.Draw(gradient)
    gradient_top = int(height * 0.38)
    gradient_height = height - gradient_top

    for offset in range(gradient_height):
        alpha = int(225 * ((offset + 1) / gradient_height))
        y = gradient_top + offset
        draw_gradient.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    image = Image.alpha_composite(image, gradient)
    draw = ImageDraw.Draw(image)

    safe_left = 72
    safe_right = 72
    safe_bottom = 58
    max_width = width - safe_left - safe_right
    max_lines = 2
    font_size = 76
    font = load_font(font_size, bold=True, serif=True)
    lines = wrap_text_strict(text, font, max_width)

    def metrics():
        line_gap = int(font_size * 0.18)
        widths = []
        heights = []
        for line_text in lines[:max_lines]:
            line_width, line_height = text_size(font, line_text)
            widths.append(line_width)
            heights.append(line_height)
        total_height = sum(heights) + line_gap * max(0, len(lines) - 1)
        return total_height, widths, heights, line_gap

    total_height, widths, heights, line_gap = metrics()
    max_height = int(height * 0.31)

    while (len(lines) > max_lines or total_height > max_height or any(w > max_width for w in widths)) and font_size > 34:
        font_size -= 4
        font = load_font(font_size, bold=True, serif=True)
        lines = wrap_text_strict(text, font, max_width)
        total_height, widths, heights, line_gap = metrics()

    lines = lines[:max_lines]
    total_height, widths, heights, line_gap = metrics()
    y = height - safe_bottom - total_height
    accent_y = max(gradient_top + 16, y - 22)
    draw.rounded_rectangle(
        [safe_left, accent_y, min(width - safe_right, safe_left + 168), accent_y + 8],
        radius=4,
        fill=THUMBNAIL_ACCENT_COLOR,
    )

    for line, line_width, line_height in zip(lines, widths, heights):
        x = safe_left
        draw.text((x + 4, y + 4), line, font=font, fill=THUMBNAIL_TEXT_SHADOW)
        draw.text((x, y), line, font=font, fill=THUMBNAIL_TEXT_COLOR)
        y += line_height + line_gap

    return image.convert("RGB")


def save_as_valid_youtube_thumbnail(
    image_bytes,
    output_path,
    max_bytes=MAX_THUMBNAIL_BYTES,
    overlay_text=None,
):
    target_ratio = THUMBNAIL_SIZE[0] / THUMBNAIL_SIZE[1]

    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise RuntimeError("Generated thumbnail was not a readable image.") from exc

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _center_crop_to_ratio(image, target_ratio)
    image = image.resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    image = apply_thumbnail_text_overlay(image, overlay_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for quality in range(92, 54, -4):
        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        if os.path.getsize(output_path) <= max_bytes:
            return output_path

    raise RuntimeError(
        f"Could not compress thumbnail below {max_bytes} bytes: {output_path}"
    )


def is_valid_youtube_thumbnail(path, max_bytes=MAX_THUMBNAIL_BYTES):
    if not path or not os.path.isfile(path):
        return False

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        return False

    if os.path.getsize(path) > max_bytes:
        return False

    try:
        with Image.open(path) as image:
            return image.format in {"JPEG", "PNG"}
    except UnidentifiedImageError:
        return False


def generate_youtube_thumbnail(title, output_dir, idea=None, output_filename="thumbnail.jpg"):
    output_path = os.path.join(output_dir, output_filename)
    prompt = build_thumbnail_prompt(title=title, idea=idea)
    model = os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)

    print("Generating YouTube thumbnail with OpenAI Images...")
    print(f"Thumbnail subject: {str(idea or title).strip()}")

    client = _openai_client()
    image_kwargs = _supported_image_kwargs(
        client.images.generate,
        {
            "model": model,
            "prompt": prompt,
            "size": "1536x1024",
            "quality": os.getenv("OPENAI_IMAGE_QUALITY", "high"),
            "output_format": "jpeg",
            "output_compression": 92,
            "response_format": "b64_json",
            "n": 1,
        },
    )
    if str(model).startswith("gpt-image"):
        image_kwargs.pop("response_format", None)
    response = _generate_image_with_compatible_kwargs(client, image_kwargs)

    image_bytes = _image_bytes_from_response(response)
    return save_as_valid_youtube_thumbnail(image_bytes, output_path, overlay_text=title)


def ensure_youtube_thumbnail(metadata, output_dir):
    output_path = os.path.join(output_dir, "thumbnail.jpg")
    if is_valid_youtube_thumbnail(output_path):
        print(f"Valid thumbnail already exists; skipping generation: {output_path}")
        return output_path

    title = metadata.get("title") or ""
    idea = metadata.get("thumbnail_idea") or title
    thumbnail_path = generate_youtube_thumbnail(title=title, idea=idea, output_dir=output_dir)

    if not is_valid_youtube_thumbnail(thumbnail_path):
        raise RuntimeError(f"Generated thumbnail is not valid for YouTube: {thumbnail_path}")

    print(f"Thumbnail ready: {thumbnail_path} ({os.path.getsize(thumbnail_path)} bytes)")
    return thumbnail_path


def main():
    parser = argparse.ArgumentParser(description="Generate a YouTube thumbnail image.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--idea")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    path = generate_youtube_thumbnail(
        title=args.title,
        idea=args.idea or args.title,
        output_dir=args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
