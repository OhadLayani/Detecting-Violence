import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("C:/Users/ohad/Downloads/combined_model (2).h5")

# Load the video
cap = cv2.VideoCapture("C:/Users/ohad/Videos/RWF-2000/val/Fight/0Ow4cotKOuw_0.avi")

# Create empty lists to store the frames and optical flow
frames = []
flows = []

# Create a loop to read each frame of the video
# Create a variable to store the previous gray frame
prev_gray = None

# Create a loop to read each frame of the video
while True:
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Append the frame to the list of frames
    frames.append(frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # Compute the optical flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Normalize the optical flow
        flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
        flow = np.uint8(flow)

        # Append the optical flow to the list of flows
        flows.append(flow)

    # Update the previous gray frame
    prev_gray = gray


# Add an extra dummy optical flow frame at the beginning or end of the flows array
dummy_flow = np.zeros_like(flows[0])
flows.insert(0, dummy_flow)
# Convert the lists of frames and flows to numpy arrays
frames = np.array(frames)
flows = np.array(flows)

# Preprocess the frames (resize and normalize)
frames = np.array([cv2.resize(frame, (70, 70)) for frame in frames])
frames = frames.astype('float32') / 255.0

# Preprocess the optical flow (resize and normalize)
flows = np.array([cv2.resize(flow, (70, 70)) for flow in flows])
flows = flows.astype('float32') / 255.0


# Combine frames and flows into input tensor
inputs = [frames, flows]

# Make prediction on input tensor
prediction = model.predict(inputs)

# Calculate percentage of violent frames
percent_violent = 100 * np.sum(prediction) / len(prediction)

# Print prediction result
if percent_violent > 50:
    print(f"The video is violent with {percent_violent:.2f}% confidence")
else:
    print(f"The video is not violent with {(100 - percent_violent):.2f}% confidence")

# Release video file
cap.release()