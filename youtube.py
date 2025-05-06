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


def build_youtube_service(tokens):
    """
    Helper function to build the YouTube service object from tokens.
    """
    creds = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ['GOOGLE_CLIENT_ID'],
        client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
        scopes=SCOPES
    )
    return build('youtube', 'v3', credentials=creds)


def youtube_list_videos(tokens, max_results=25):
    """
    Lists videos uploaded by the authenticated user's channel.
    """
    youtube = build_youtube_service(tokens)
    try:
        channels_response = youtube.channels().list(
            part="contentDetails",
            mine=True
        ).execute()
        if not channels_response["items"]:
            print("No channel found for the authenticated user.")
            return []

        uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        playlist_response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        ).execute()

        return playlist_response.get("items", [])
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return []


def youtube_upload_video(tokens, file_path, title, description, tags=None,
                         categoryId="22", privacyStatus="public"):
    """
    Uploads a video to YouTube.
    """
    youtube = build_youtube_service(tokens)
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": categoryId,
        },
        "status": {
            "privacyStatus": privacyStatus
        }
    }
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    try:
        print("Uploading video...")
        print(f"File: {file_path}")
        print(f"Title: {title}")
        print(f"Description: {description}")
        print(f"Tags: {tags}")
        print(f"Category ID: {categoryId}")
        print(f"Privacy Status: {privacyStatus}")
        print("Please wait...")
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
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


def youtube_update_video(tokens, video_id, new_description=None, new_title=None, new_thumbnail_path=None):
    """
    Updates an existing video's metadata and/or thumbnail.
    """
    youtube = build_youtube_service(tokens)
    update_response = None
    thumbnail_response = None

    try:
        # If updating title or description, first fetch the current snippet.
        if new_description or new_title:
            video_response = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            if not video_response["items"]:
                raise Exception("Video not found.")
            snippet = video_response["items"][0]["snippet"]

            if new_description:
                snippet["description"] = new_description
            if new_title:
                snippet["title"] = new_title

            update_response = youtube.videos().update(
                part="snippet",
                body={
                    "id": video_id,
                    "snippet": snippet
                }
            ).execute()
            print("Video metadata updated.")

        # If updating the thumbnail
        if new_thumbnail_path:
            thumbnail_response = youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(new_thumbnail_path)
            ).execute()
            print("Thumbnail updated.")

        return {
            "video_metadata": update_response,
            "thumbnail_response": thumbnail_response
        }
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
                    "isDraft": False
                }
            },
            media_body=MediaFileUpload(caption_file_path)
        ).execute()
        print("✅ Captions uploaded successfully.")
        return response
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while uploading captions:\n{e.content}")
        return None


# Example usage:
if __name__ == "__main__":
    # Replace with your actual Google user ID stored in your database.
    google_user_id = "XYZ"
    try:
        tokens = get_and_refresh_google_user_tokens(google_user_id)

        # 1. List Videos
        videos = youtube_list_videos(tokens)
        print("Uploaded videos:")
        for item in videos:
            title = item["snippet"]["title"]
            video_id = item["contentDetails"]["videoId"]
            print(f"Title: {title} (ID: {video_id})")

        # 2. Upload Video
        # Uncomment and update the following lines to test video upload.
        # upload_response = youtube_upload_video(
        #     tokens,
        #     file_path="path/to/your/video.mp4",
        #     title="Test Video",
        #     description="This is a test video upload.",
        #     tags=["test", "video"],
        #     categoryId="22",  # People & Blogs category
        #     privacyStatus="private"
        # )
        # print("Upload Response Video ID:", upload_response)

        # 3. Update Video (metadata and/or thumbnail)
        # Uncomment and update the following lines to test updating a video.
        # video_id_to_update = "YOUR_VIDEO_ID"
        # update_result = youtube_update_video(
        #     tokens,
        #     video_id=video_id_to_update,
        #     new_description="Updated video description",
        #     new_title="Updated Video Title",
        #     new_thumbnail_path="path/to/new/thumbnail.jpg"
        # )
        # print("Update Response:", update_result)

        # 4. Upload Captions (Uncomment to test)
        # youtube_upload_captions(
        #     tokens,
        #     video_id="YOUR_VIDEO_ID",
        #     caption_file_path="path/to/homily_captions.srt"
        # )

    except Exception as e:
        print("Error:", e)
