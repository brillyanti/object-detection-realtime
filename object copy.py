import cv2
import torch
from gtts import gTTS
from pygame import mixer
import time
from PIL import Image
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Set the model to evaluation mode
model.eval()

# Initialize the mixer for audio playback
mixer.init()

# Initialize a variable to keep track of time
last_detection_time = time.time()

# Function to perform object detection and provide voice feedback
def perform_object_detection(image):
    global last_detection_time  # Declare last_detection_time as a global variable

    with torch.no_grad():
        results = model(image)

    # Extract detection results
    pred = results.pred[0]

    for detection in pred:
        label = int(detection[5])
        label_name = model.names[label]
        confidence = float(detection[4])

        # Display the detected object and confidence
        print(f"Detected: {label_name}, Confidence: {confidence:.2f}")

        # Check if a certain amount of time has passed since the last detection
        current_time = time.time()
        if current_time - last_detection_time >= 3:
            # Convert the label name to text
            text = f"I see a {label_name}"
            tts = gTTS(text=text, lang='en')

            # Save the text to an audio file
            tts.save("output.mp3")

            # Play the audio feedback
            mixer.music.load("output.mp3")
            mixer.music.play()

            # Update the time of the last detection
            last_detection_time = current_time

# Open a video capture source (e.g., webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Read a frame from the video source
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to a format compatible with YOLOv5
    img = Image.fromarray(frame[:, :, ::-1])

    # Perform object detection and provide voice feedback
    perform_object_detection(img)

    # Display the frame with detection results
    cv2.imshow("YOLOv5 Object Detection", np.array(img)[:, :, ::-1])

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source and close all windows
cap.release()
cv2.destroyAllWindows()
