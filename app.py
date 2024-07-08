# app.py

from flask import Flask, render_template, Response
import cv2
import numpy as np
app = Flask(__name__)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify video file path
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame to 300x300 for MobileNet SSD input
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # Set input to the network and perform forward pass
            net.setInput(blob)
            detections = net.forward()

            # Loop over the detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:  # Filter weak detections
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 8:  # Check if the detected object is a cat
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                        label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                        cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
