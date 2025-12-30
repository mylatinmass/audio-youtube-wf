import os
import psycopg2
from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
from get_token import get_and_refresh_google_user_tokens, SCOPES

# Load environment variables (if using a .env file)
load_dotenv()


def find_thumbnail_in_final_folder(final_video_path):
    """
    Find ANY image in the same folder as the final video.
    Accepts jpg, jpeg, png, webp.
    Returns the first one found.
    """
    folder = os.path.dirname(final_video_path)
    if not os.path.isdir(folder):
        return None

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return os.path.join(folder, filename)

    return None


def build_youtube_service(tokens):
    """
    Helper function to build the YouTube service object from tokens.
    """
    creds = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        scopes=SCOPES,
    )
    return build("youtube", "v3", credentials=creds)


def youtube_list_videos(tokens, max_results=25):
    """
    Lists videos uploaded by the authenticated user's channel.
    """
    youtube = build_youtube_service(tokens)
    try:
        channels_response = (
            youtube.channels().list(
                part="contentDetails",
                mine=True,
            ).execute()
        )
        if not channels_response.get("items"):
            print("No channel found for the authenticated user.")
            return []

        uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        playlist_response = (
            youtube.playlistItems()
            .list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=max_results,
            )
            .execute()
        )

        return playlist_response.get("items", [])
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return []


def _append_edu_block_to_description(description, edu_type=None, edu_problems=None):
    """
    Minimal helper: append Education 'type' and 'problems' to the description so they are preserved on upload.
    """
    if not edu_type and not edu_problems:
        return description

    lines = []
    lines.append(description or "")

    lines.append("\n\n---")
    lines.append("EDUCATION METADATA")
    if edu_type:
        lines.append(f"Type: {edu_type}")
    if edu_problems:
        lines.append("Problems:")
        for p in edu_problems:
            lines.append(f"- {p}")

    return "\n".join(lines).strip()


def youtube_upload_video(
    tokens,
    file_path,
    title,
    description,
    tags=None,
    categoryId="27",
    privacyStatus="public",
    edu_type=None,
    edu_problems=None,
):
    """
    Uploads a video to YouTube.

    - Default categoryId set to "27" (Education).
    - Optional 'edu_type' and 'edu_problems' get appended to the description.
    """
    youtube = build_youtube_service(tokens)

    final_description = _append_edu_block_to_description(description, edu_type, edu_problems)

    body = {
        "snippet": {
            "title": title,
            "description": final_description,
            "tags": tags,
            "categoryId": categoryId,
        },
        "status": {"privacyStatus": privacyStatus},
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    try:
        print("Uploading video...")
        print(f"File: {file_path}")
        print(f"Title: {title}")
        print(f"Category ID: {categoryId}")
        print(f"Privacy Status: {privacyStatus}")
        if tags:
            print(f"Tags: {tags}")
        if edu_type or edu_problems:
            print("Including Education metadata in description block.")

        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%...")

        print("Upload complete!")
        return response.get("id") if response else None

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None


def youtube_upload_video_with_optional_thumbnail(
    tokens,
    file_path,
    title,
    description,
    tags=None,
    categoryId="27",
    edu_type=None,
    edu_problems=None,
):
    """
    RULE:
    - If ANY image exists in the final folder -> upload PUBLIC and set thumbnail.
    - If no image exists -> upload PRIVATE (draft).
    """
    thumbnail_path = find_thumbnail_in_final_folder(file_path)
    privacyStatus = "public" if thumbnail_path else "private"

    # ✅ FIX: call youtube_upload_video (not this function)
    video_id = youtube_upload_video(
        tokens=tokens,
        file_path=file_path,
        title=title,
        description=description,
        tags=tags,
        categoryId=categoryId,
        privacyStatus=privacyStatus,
        edu_type=edu_type,
        edu_problems=edu_problems,
    )

    if not video_id:
        return None

    if thumbnail_path:
        youtube_update_video(
            tokens=tokens,
            video_id=video_id,
            new_thumbnail_path=thumbnail_path,
        )
        print(f"✅ Thumbnail set from: {thumbnail_path}")
    else:
        print("⚠️ No image found in final folder → uploaded as PRIVATE draft.")

    return video_id


def youtube_update_video(tokens, video_id, new_description=None, new_title=None, new_thumbnail_path=None):
    """
    Updates an existing video's metadata and/or thumbnail.
    """
    youtube = build_youtube_service(tokens)
    update_response = None
    thumbnail_response = None

    try:
        if new_description or new_title:
            video_response = youtube.videos().list(part="snippet", id=video_id).execute()
            if not video_response.get("items"):
                raise Exception("Video not found.")
            snippet = video_response["items"][0]["snippet"]

            if new_description is not None:
                snippet["description"] = new_description
            if new_title is not None:
                snippet["title"] = new_title

            update_response = (
                youtube.videos()
                .update(
                    part="snippet",
                    body={
                        "id": video_id,
                        "snippet": snippet,
                    },
                )
                .execute()
            )
            print("Video metadata updated.")

        if new_thumbnail_path:
            thumbnail_response = youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(new_thumbnail_path),
            ).execute()
            print("Thumbnail updated.")

        return {"video_metadata": update_response, "thumbnail_response": thumbnail_response}

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None


def youtube_upload_captions(tokens, video_id, caption_file_path, language="en", name="English Captions"):
    """
    Uploads a caption file (.srt) to a YouTube video.
    """
    youtube = build_youtube_service(tokens)
    try:
        response = youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "language": language,
                    "name": name,
                    "videoId": video_id,
                    "isDraft": False,
                }
            },
            media_body=MediaFileUpload(caption_file_path),
        ).execute()
        print("✅ Captions uploaded successfully.")
        return response
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while uploading captions:\n{e.content}")
        return None


# Example usage:
if __name__ == "__main__":
    google_user_id = "XYZ"
    try:
        tokens = get_and_refresh_google_user_tokens(google_user_id)

        videos = youtube_list_videos(tokens)
        print("Uploaded videos:")
        for item in videos:
            title = item["snippet"]["title"]
            video_id = item["contentDetails"]["videoId"]
            print(f"Title: {title} (ID: {video_id})")

    except Exception as e:
        print("Error:", e)
