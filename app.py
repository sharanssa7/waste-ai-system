from flask import Flask, render_template, request
from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load trained classifier
classifier = tf.keras.models.load_model("bio_nonbio_classifier.keras")

def classify_object(cropped_img):
    img = cv2.resize(cropped_img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img)[0][0]

    if prediction > 0.5:
        return "Non-Bio", (255, 0, 0)
    else:
        return "Bio", (0, 255, 0)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    results = yolo_model(filepath)

    bio_count = 0
    nonbio_count = 0

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        label, color = classify_object(cropped)

        if label == "Bio":
            bio_count += 1
        else:
            nonbio_count += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    output_path = "static/output.jpg"
    cv2.imwrite(output_path, image)

    return render_template(
        "index.html",
        output_image="output.jpg",
        bio=bio_count,
        nonbio=nonbio_count
    )

if __name__ == '__main__':
    app.run(debug=True)