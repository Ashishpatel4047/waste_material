from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import cv2
import numpy as np
import uuid

app = Flask(__name__)

# 🔥 Lazy loading (RAM optimize)
model = None

def load_model():
    global model
    if model is None:
        print("🔄 Loading YOLO model...")
        model = YOLO("yolov8n.pt")  # lightweight model
        print("✅ Model loaded")

# Upload folder
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    load_model()

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"})

    # Save file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (640, 640))

    # Prediction
    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": model.names[cls],
                "confidence": conf
            })

    return jsonify({
        "detections": detections,
        "count": len(detections)
    })

# ✅ Render ke liye
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)