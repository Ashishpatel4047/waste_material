import cv2
import torch
from ultralytics import YOLO
import os

class Detector:
    def __init__(self, model_path):
        model_path = model_path.strip('"').strip("'")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")

        print(f"🔄 Loading model: {model_path}")
        self.model = YOLO(model_path)

        self.class_names = self.model.names
        print(f"📊 Classes: {self.class_names}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")

    def detect(self, frame, conf_threshold=0.4):
        if frame is None:
            return []

        results = self.model.predict(
            frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id]
                })

        return detections

    # ✅ FIX: function class ke andar
    def draw_detections(self, frame, detections, show_confidence=True):
        if frame is None:
            return None

        frame_copy = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class_name"]
            conf = det["confidence"]
            cls_id = det["class_id"]

            # 🎨 color generate
            color = (
                int((cls_id * 123) % 255),
                int((cls_id * 456) % 255),
                int((cls_id * 789) % 255)
            )

            # 🟩 Bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # 🏷 Label text
            text = f"{label} {conf:.2f}" if show_confidence else label

            font_scale = 0.6
            thickness = 2

            (w, h), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            # label position fix
            y_text = max(y1 - 10, h + 10)

            # 🟥 background box
            cv2.rectangle(
                frame_copy,
                (x1, y_text - h - 5),
                (x1 + w, y_text + baseline),
                color,
                -1
            )

            # 📝 text
            cv2.putText(
                frame_copy,
                text,
                (x1, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

        return frame_copy