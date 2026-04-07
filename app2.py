from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np

app = Flask(__name__)

# 🔥 Lazy loading (RAM bachane ke liye)
model = None

def load_model():
    global model
    if model is None:
        print("🔄 Loading lightweight model...")
        model = YOLO("yolov8n.pt")  # ✅ nano model (lightweight)
        print("✅ Model loaded")

@app.route('/')
def home():
    return "🚀 YOLO App Running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    load_model()

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"})

    # Read image
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize image (RAM optimize)
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

# ✅ IMPORTANT: Render ke liye correct port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)