import streamlit as st
import cv2
import torch
from gtts import gTTS
from pygame import mixer
from PIL import Image
import numpy as np
import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.eval()

# Initialize the mixer for audio playback
mixer.init()

# Initialize a variable to keep track of time
last_detection_time = 0

# Function for object detection and feedback
def perform_object_detection(image):
    global last_detection_time

    with torch.no_grad():
        results = model(image)

    pred = results.pred[0]

    for detection in pred:
        label = int(detection[5])
        label_name = model.names[label]
        confidence = float(detection[4])

        if confidence > 0.5:
            current_time = time.time()
            if current_time - last_detection_time >= 3:
                text = f"I see a {label_name}"
                tts = gTTS(text=text, lang='en')
                tts.save("output.mp3")
                mixer.music.load("output.mp3")
                mixer.music.play()
                last_detection_time = current_time

# Streamlit app layout
st.title('YOLOv5 Object Detection with Voice Feedback')

# Use checkbox for Start/Stop functionality
is_detecting = st.checkbox("Start Detecting")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while is_detecting:
    ret, frame = cap.read()

    if not ret:
        break

    img = Image.fromarray(frame[:, :, ::-1])

    # Perform object detection and provide voice feedback
    perform_object_detection(img)

    st.image(np.array(img)[:, :, ::-1], channels="BGR")

cap.release()
mixer.quit()
