import os
import cv2

# Set path to the directory containing the two folders
path = "C:/Users/ohad/Videos/RWF-2000/val"

# Create a new directory for the extracted frames
picture_path = os.path.join(path, "C:/Users/ohad/Pictures/RWF-2000/val")
os.makedirs(picture_path, exist_ok=True)

# Loop through the two subfolders
for subdir in ["Fight", "NonFight"]:
    subdir_path = os.path.join(path, subdir)

    # Create a new subdirectory within the pictures directory
    picture_subdir_path = os.path.join(picture_path, subdir)
    os.makedirs(picture_subdir_path, exist_ok=True)

    # Loop through each AVI video file in the subfolder
    for filename in os.listdir(subdir_path):
        if filename.endswith(".avi"):
            # Open the video file using OpenCV
            video_path = os.path.join(subdir_path, filename)
            cap = cv2.VideoCapture(video_path)

            # Loop through each frame and save as an image file
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == False:
                    break
                frame_num += 1
                frame_filename = f"{filename[:-4]}_{frame_num}.jpg"
                frame_path = os.path.join(picture_subdir_path, frame_filename)
                cv2.imwrite(frame_path, frame)

            # Release the video capture object
            cap.release()