import cv2
from gtts import gTTS
from pygame import mixer
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = "/Users/briliantypuspita/Documents/Tugas-Akhir/bismillah-2/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/Users/briliantypuspita/Documents/Tugas-Akhir/bismillah-2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/Users/briliantypuspita/Documents/Tugas-Akhir/bismillah-2/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

mixer.init()  # Initialize the mixer for audio playback

# Initialize a variable to keep track of time
last_detection_time = time.time()

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    
    if len(classIds) != 0:
        current_time = time.time()
        # Check if 3 seconds have passed since the last detection
        if current_time - last_detection_time >= 3:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                label = f"{classNames[classId - 1].upper()}: {int(confidence * 100)}%"  # Display class and confidence
                cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                answer = classNames[classId - 1]
                newVoiceRate = 10
                tts = gTTS(text=answer, lang='en')
                tts.save("output.mp3")
                mixer.music.load("output.mp3")
                mixer.music.play()

                # Update the time of the last detection
                last_detection_time = current_time

    cv2.imshow("output", img)
    cv2.waitKey(1)
