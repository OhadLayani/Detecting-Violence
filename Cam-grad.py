import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_heat_map(model, preprocessed_frame):
    # Create a model that outputs the last convolutional layer and the model's output
    last_conv_layer_model = tf.keras.models.Model(model.inputs, model.get_layer("conv2d").output)
    output_layer = model.output

    with tf.GradientTape() as tape:
        # Watch the last convolutional layer's output
        last_conv_layer_output = last_conv_layer_model(preprocessed_frame[None, ...])
        tape.watch(last_conv_layer_output)

        # Forward pass
        predictions = model(preprocessed_frame[None, ...])

        # Get the top predicted class index
        top_class_index = tf.argmax(predictions[0])

        # Retrieve the top predicted class score
        top_class_score = predictions[0, top_class_index]

    # Calculate the gradients of the top predicted class score with respect to the last conv layer output
    grads = tape.gradient(top_class_score, last_conv_layer_output)

    # Pool the gradients over all the axes except the channel axis
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the corresponding importance score
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Resize the heatmap to match the input frame size
    heatmap = tf.image.resize(heatmap, (preprocessed_frame.shape[0], preprocessed_frame.shape[1]))

    # Convert the heatmap to RGB
    heatmap = tf.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.grayscale_to_rgb(heatmap)
    heatmap = np.array(heatmap)

    return heatmap


# Function to load and analyze frames
def analyze_frames(video_path, model):
    video = cv2.VideoCapture(video_path)
    frames = []
    predictions = []

    # Iterate over frames
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Preprocess frame (if required)
        preprocessed_frame = preprocess_frame(frame)

        # Generate heat map
        heatmap = generate_heat_map(model, preprocessed_frame)

        # Overlay the heat map on the frame
        overlay = cv2.addWeighted(frame, 0.8, heatmap, 0.4, 0)

        # Save the modified frame with heat map overlay to file
        modified_frame_path = f"frame_{frame_count}.jpg"
        cv2.imwrite(modified_frame_path, overlay)

        # Make prediction
        prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))

        # Store frame path and prediction
        frames.append(modified_frame_path)
        predictions.append(prediction)

        frame_count += 1

    video.release()

    return frames, predictions

# Example usage
video_path = "C:/Users/ohad/Videos/RWF-2000/val/Fight/0Ow4cotKOuw_0.avi"
model_name = "C:/Users/ohad/Downloads/Rgb_model.h5"

model = tf.keras.models.load_model(model_name)

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = tf.image.resize(frame, [70, 70])
    return resized_frame

# Analyze frames and predictions
frames, predictions = analyze_frames(video_path, model)

print(frames)
print(predictions)