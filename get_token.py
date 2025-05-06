
import os
import psycopg2
from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load environment variables from .env (if you use one)
load_dotenv()

# Google API scopes used in your project
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/webmasters.readonly',
    'https://www.googleapis.com/auth/analytics.readonly',
    'https://www.googleapis.com/auth/gmail.readonly',
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly"


]

def get_and_refresh_google_user_tokens(google_id):
    """
    Retrieves tokens for the given Google user from CockroachDB.
    If the access token is expired, it automatically refreshes it,
    updates the database entry, and returns the fresh tokens.
    
    :param google_id: The Google user ID as stored in your database.
    :return: A dict with access_token, refresh_token, and token_expiry.
    """
    # Connect to CockroachDB using psycopg2
    conn = psycopg2.connect(os.environ['COCKROACHDB_CONNECTION_STRING'])
    cur = conn.cursor()

    # Retrieve the stored tokens for this user
    cur.execute(
        "SELECT access_token, refresh_token, token_expiry FROM google_users WHERE google_id = %s",
        (google_id,)
    )
    row = cur.fetchone()
    if row is None:
        cur.close()
        conn.close()
        raise Exception(f"No user found with google_id: {google_id}")
    
    access_token, refresh_token, token_expiry = row
    # Ensure token_expiry is timezone aware in UTC
    if token_expiry.tzinfo is None:
        token_expiry = token_expiry.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    # Check if the access token has expired
    if now >= token_expiry:
        print(f"Access token expired for user {google_id}. Refreshing token... If a new token is needed, visit this website while logged in as mylatinmass@gmail.com https://mylatinmass.com/.netlify/functions/google-oauth")
        
        # Create credentials object using the stored tokens and your client info
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ['GOOGLE_CLIENT_ID'],
            client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
            scopes=SCOPES
        )
        # Refresh the access token if needed
        request = Request()
        creds.refresh(request)
        
        # Update tokens with new values
        new_access_token = creds.token
        new_expiry = creds.expiry  # This is a datetime object in UTC
        # Sometimes a new refresh token may not be returned
        new_refresh_token = creds.refresh_token if creds.refresh_token else refresh_token

        # Update the database record with the new tokens
        cur.execute(
            """
            UPDATE google_users 
            SET access_token = %s, token_expiry = %s, refresh_token = %s
            WHERE google_id = %s
            """,
            (new_access_token, new_expiry, new_refresh_token, google_id)
        )
        conn.commit()

        # Replace the tokens with the new values
        access_token = new_access_token
        token_expiry = new_expiry
        refresh_token = new_refresh_token
        print(f"Token refreshed and updated in the database for user {google_id}.")
    else:
        print(f"Access token is still valid for user {google_id}.")

    cur.close()
    conn.close()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_expiry": token_expiry
    }

# Example usage:
if __name__ == "__main__":
    google_user_id = "102136376185174842894"  # Replace with the actual Google user ID
    try:
        tokens = get_and_refresh_google_user_tokens(google_user_id)
        print("Access Token:", tokens["access_token"])
        print("Refresh Token:", tokens["refresh_token"])
        print("Token Expiry:", tokens["token_expiry"])
        # Now you can use tokens["access_token"] to make authenticated API calls
    except Exception as e:
        print("Error retrieving tokens:", e)
