import os
import re
import requests
import time

# Replace with your actual Auphonic API key.
api_key = "SFrUYP0rB9A5jKbr7t1N5o7SlfKdhxHZ"  # <-- Your Auphonic API key


def to_kebab_case(filename):
    """
    Convert a filename to kebab-case (lowercase, words separated by dashes).
    The extension is removed.
    """
    name, _ = os.path.splitext(filename)
    # Replace underscores and spaces with dashes.
    name = re.sub(r'[\s_]+', '-', name)
    # Remove any non-alphanumeric characters except dashes.
    name = re.sub(r'[^a-z0-9-]', '', name.lower())
    # Remove extra dashes.
    name = re.sub(r'-+', '-', name)
    return name.strip('-')

def poll_production_status(production_uuid, poll_interval=10):
    """
    Poll the Auphonic production status until the production is complete.

    Args:
      production_uuid (str): The production UUID.
      api_key (str): Your Auphonic API key.
      poll_interval (int): Seconds between polls.

    Returns:
      dict: The final production data when status is "Done".
    """
    status_url = f"https://auphonic.com/api/production/{production_uuid}.json"
    headers = {"Authorization": f"bearer {api_key}"}
    
    while True:
        response = requests.get(status_url, headers=headers)
        print(response)
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Received an invalid JSON response. Waiting before retrying...")
            time.sleep(poll_interval)
            continue
        
        data = json_data.get("data", {})
        status_string = data.get("status_string", "").lower()
        print("Current status:", status_string)
        
        if status_string == "done":
            return data
        elif status_string in ("error", "failed"):
            raise Exception("Production failed:", data.get("error_message"))
        
        time.sleep(poll_interval)

# ------------------ Main Code ------------------

def start_production(audio_file_path):
    # Generate a kebab-case title from the audio file name.
    file_basename = os.path.basename(audio_file_path)
    title = to_kebab_case(file_basename)

    # The production endpoint URL.
    url = "https://auphonic.com/api/simple/productions.json"

    headers = {
        "Authorization": f"bearer {api_key}"
    }

    # Create the data payload with your preset, title, and action.
    data = {
        "preset": "FwYyPMNMQBkYQLFUa3xpjZ",  # <-- Replace with your actual preset ID.
        "title": title,
        "action": "start"
    }

    # Submit the file.
    with open(audio_file_path, "rb") as f:
        files = {
            "input_file": f
        }
        response = requests.post(url, headers=headers, data=data, files=files)

    # Print the initial response.
    resp_json = response.json()
    print("Initial response:", resp_json)

    # Extract the production UUID from the response.
    production_uuid = resp_json.get("data", {}).get("uuid")
    print("Production UUID:", production_uuid)
    return production_uuid

def download_file(production_uuid, directory):
    print("Production UUID:", production_uuid)
    
    # Poll until production is complete.
    final_data = poll_production_status(production_uuid)
    output_files = final_data.get("output_files", [])
    
    if output_files:
        # Get output_basename and file extension.
        output_basename = final_data.get("output_basename", "output")
        file_extension = output_files[0].get("ending", "mp3")
        
        # Construct the download URL.
        constructed_url = f"https://auphonic.com/api/download/audio-result/{production_uuid}/{output_basename}.{file_extension}"
        
        # Use the download_url provided by the API if available.
        if output_files[0].get("download_url"):
            download_url = output_files[0].get("download_url") + "?bearer_token=" + api_key
            print("Download URL provided by API:", download_url)
        else:
            download_url = constructed_url
            print("Download URL constructed manually:", download_url)
        
        # Automatically download the file.
        print("Downloading file...")
        download_response = requests.get(download_url)
        if download_response.status_code == 200:
            local_filename = os.path.join(directory, f"{output_basename}.{file_extension}")
            with open(local_filename, "wb") as f:
                f.write(download_response.content)
            print(f"File downloaded successfully as {local_filename}")
            return local_filename
        else:
            print("Failed to download the file.")
    else:
        print("Production finished but no output files were found.")
    return None

if __name__ == "__main__":
    # Prompt the user for the audio file path and remove any extra quotes.
    audio_file_path = input("Enter the path to your audio file: ").strip().strip('"\'')
    
    uuid = start_production(audio_file_path)
    #uuid = "PCCMcXrs2wwYTTBTcw3u2Z" 

    if uuid:
        download_file(uuid)
    else:
        print("Production not started successfully; no UUID received.")

