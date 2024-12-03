from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the pre-trained model and classes
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    # Get the frame from the request
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Perform object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id in [14, 15, 72]:  # Bird, Cell phone, Remote
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detected_objects.append(CLASSES[class_id])

    # Encode frame to JPEG and send it back with detection status
    ret, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    response = {
        'image': img_str,
        'detected_objects': detected_objects
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
