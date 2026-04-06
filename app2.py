from flask import Flask, request, render_template
import os
import cv2
from detector import Detector
import uuid

app = Flask(__name__)

model = Detector("weights/best.pt")

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if not file:
        return "No file uploaded"

    # unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    # detect
    detections = model.detect(img)

    # draw boxes
    output_img = model.draw_detections(img, detections)

    # save output image
    output_path = os.path.join(UPLOAD_FOLDER, "output_" + filename)
    cv2.imwrite(output_path, output_img)

    # object names list
    objects = [d["class_name"] for d in detections]

    return render_template(
        "index.html",
        image_path=output_path,
        objects=objects,
        count=len(objects)
    )

if __name__ == "__main__":
    app.run(debug=True)