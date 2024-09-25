import os
import cv2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set path to the directory containing the train and val folders
from googleapiclient.http import MediaFileUpload

path = "C:/Users/ohad/Pictures/RWF-2000"

# Set path to the directory containing the service account JSON file
json_path =  "C:/Users/ohad/Downloads/glowing-patrol-381809-1ec86098c2e7.json"
# Set name of the Google Drive folder where the images will be uploaded
folder_name = "RWF-2000 images"
print("here")
# Set scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

# Create credentials object for Google Drive API using service account JSON file
creds = None
try:
    creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
except FileNotFoundError:
    print("The service account JSON file was not found.")
except HttpError as error:
    print(f"An error occurred: {error}")

# Create Google Drive API client
service = build('drive', 'v3', credentials=creds)
print("here")

# Loop through the train and val folders
for dataset in ["train", "val"]:
    dataset_path = os.path.join(path, dataset)
    # Loop through the Violence and NonViolence subfolders
    for subdir in ["Fight", "NonFight"]:
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.exists(subdir_path):
            continue

        # Create a new folder in Google Drive for the images
        file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')

        # Loop through each image file in the subfolder
        for filename in os.listdir(subdir_path):
            if not filename.endswith(".jpg"):
                continue

            # Set metadata for the image file
            file_metadata = {'name': filename, 'parents': [folder_id]}
            print("here")

            # Set path to the image file
            image_path = os.path.join(subdir_path, filename)

            # Upload the image file to Google Drive
            media = MediaFileUpload(image_path, resumable=True)
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            print(f"{filename} uploaded to Google Drive.")