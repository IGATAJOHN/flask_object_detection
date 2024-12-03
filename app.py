from flask import Flask, render_template, request, Response, send_file
import cv2
import numpy as np
from io import BytesIO
import threading
import os
import time
from playsound import playsound

app = Flask(__name__)

# Load the pre-trained model and classes
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone"]

# Global variable to control sound playback
sound_playing = False
stop_sound = False

def play_sound():
    global stop_sound
    while not stop_sound:
        playsound("alert.wav")
        time.sleep(0.1)  # Short delay to prevent high CPU usage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sound_playing, stop_sound

    # Get the frame from the request
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Perform mobile phone detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    phone_detected = False

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 8:  # Check if the detected object is a cell phone
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                phone_detected = True

    if phone_detected and not sound_playing:
        stop_sound = False
        sound_playing = True
        threading.Thread(target=play_sound).start()
    elif not phone_detected and sound_playing:
        stop_sound = True
        sound_playing = False

    # Encode frame to JPEG and send it back
    ret, buffer = cv2.imencode('.jpg', frame)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
