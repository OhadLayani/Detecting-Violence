import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model_name="C:/Users/ohad/Downloads/Rgb_model.h5"
#model_name = "C:/Users/ohad/Downloads/opt_model (1).h5"
#model_name = "C:/Users/ohad/Downloads/combined_model (2).h5"
if 'Rgb' in model_name:
    model_type = 'rgb'
elif 'opt' in model_name:
    model_type = 'opt'
else:
    model_type = 'combined'
model = tf.keras.models.load_model(model_name)

# Load the video and extract frames
#violent test
video_folder = "C:/Users/ohad/Videos/RWF-2000/val/Fight"
video_file = "C:/Users/ohad/Videos/RWF-2000/val/Fight/0Ow4cotKOuw_0.avi"

#Non violent test
#video_file = "C:/Users/ohad/Videos/RWF-2000/val/NonFight/1AURh0Wj_0.avi"
cap = cv2.VideoCapture(video_file)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (70, 70))
    frames.append(frame)
cap.release()

# Convert frames to numpy array
frames = np.array(frames)
frames = frames.astype('float32') / 255.0

# Compute optical flow if required
if model_type == 'opt' or model_type == 'combined':
    # Convert frames to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    # Compute optical flow
    flows = []
    prev_gray = gray_frames[0]
    for gray in gray_frames[1:]:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Normalize the optical flow
        flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
        flow = np.uint8(flow)

        flows.append(flow)
        prev_gray = gray
    # Convert flows to numpy array
    # Add an extra dummy optical flow frame at the beginning or end of the flows array
    dummy_flow = np.zeros_like(flows[0])
    flows.insert(0, dummy_flow)

    flows = np.array(flows)
    flows = flows.astype('float32') / 255.0



# Prepare input for the model
if model_type == 'rgb':
    inputs = frames
elif model_type == 'opt':
    inputs = flows
else:
    inputs = [frames, flows]

# Make prediction
predictions = model.predict(inputs)

# Compute percentage of violent frames
if model_type == 'rgb':
    percent_violent = 100 * np.sum(predictions) / len(predictions)
else:
    percent_violent = 100 * np.sum(predictions) / len(predictions)

# Print prediction
if model_type == 'rgb':
    if percent_violent > 50:
        print('The video is violent with %.2f%% confidence.' % percent_violent)
    else:
        print('The video is not violent with %.2f%% confidence.' % (100 - percent_violent))
if model_type == 'opt':
    if percent_violent > 50:
        print('The video is violent according to optical flow with %.2f%% confidence.' % percent_violent)
    else:
        print('The video is not violent according to optical flow with %.2f%% confidence.' % (100 - percent_violent))
if model_type == 'combined':
    if percent_violent > 50:
        print('The video is violent according to rgb and optical with %.2f%% confidence.' % percent_violent)
    else:
        print('The video is not violent according to rgb and optical flow with %.2f%% confidence.' % (100 - percent_violent))