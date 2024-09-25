import cv2

def extract_frames(input_file, output_folder):
    # Open the video file
    video = cv2.VideoCapture(input_file)

    # Check if the video file is successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    # Get some video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Total frames:", total_frames)
    print("FPS:", fps)
    print("Frame size:", frame_width, "x", frame_height)

    # Create the output folder if it doesn't exist
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Read and save each frame
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not read successfully, the video has ended
        if not ret:
            break

        # Save the frame as an image file
        output_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_file, frame)

        # Print the progress
        print("Saved frame", frame_count + 1, "of", total_frames)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

    print("Extraction complete.")

# Example usage
extract_frames("C:/Users/ohad/Videos/RWF-2000/val/Fight/0Ow4cotKOuw_0.avi", "C:/Users/ohad/Documents/Test Frames")