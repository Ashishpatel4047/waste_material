from html import parser

import cv2
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Custom Modules Import
from detector import Detector
from tracker import Tracker
import utils  # Jisme draw_boxes, draw_legend, draw_fps hai

class WasteDetectorApp:
    def __init__(self, model_path):
        self.model_path = model_path
        self.detector = None
        # Tracker initialize (15 frames tak missing object yaad rakhega)
        self.tracker = Tracker(max_disappeared=15, max_distance=80)
        self.class_names = [] # Model load hone par fill hoga

    def load_model(self):
        try:
            print(f"🚀 Loading Model: {self.model_path}")
            self.detector = Detector(self.model_path)
            # Model se class names nikalna (agar detector class mein metadata hai)
            # Defaulting to common waste classes agar nahi milta
            self.class_names = getattr(self.detector, 'classes', ["Plastic", "Metal", "Paper", "Glass", "Trash"])
            return True
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False

    def process_frame(self, frame, conf=0.6):
        """Detection -> Tracking -> Drawing ka main logic"""
        if frame is None: return None

        # 1. Detection
        detections = self.detector.detect(frame, conf_threshold=conf)

        # 2. Tracking (IDs assign karna)
        tracked_objects = self.tracker.update(detections)

        # 3. Visualization (Using your updated utils)
        # Bounding boxes aur IDs draw karein
        frame = utils.draw_boxes(frame, tracked_objects)
        
        # Legend dikhayein (Kaunsa color kis waste ka hai)
        frame = utils.draw_legend(frame, self.class_names)
        
        return frame, tracked_objects

    def process_webcam(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Camera access denied")
            return

        print("🎥 Live Tracking Start! 'Q': Quit | 'S': Save | 'R': Reset Tracker")
        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Frame Process karein
            processed_frame, tracked_map = self.process_frame(frame)

            # FPS Calculate aur Draw karein
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            processed_frame = utils.draw_fps(processed_frame, fps)

            # Screen par stats dikhayein (Current count)
            stats = {"Current Objects": len(tracked_map), "Total Detected": self.tracker.next_object_id}
            processed_frame = utils.draw_statistics(processed_frame, stats, position=(20, 80))

            cv2.imshow("AI Waste Monitor", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): self.tracker.reset()
            elif key == ord('s'):
                fn = f"waste_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn, processed_frame)
                print(f"📸 Saved: {fn}")

        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, video_path, output_path="output_tracked.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_val, (width, height))

        print(f"🎬 Processing Video: {video_path}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed_frame, tracked_map = self.process_frame(frame, conf=0.7)
            out.write(processed_frame)
            
            cv2.imshow("Processing...", utils.resize_for_display(processed_frame))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
# Is path ko double check karein. Kya 'best.pt' isi folder mein hai?
    parser.add_argument('--model', default=r"C:\Users\91840\waste_material\weights\best.pt")
    parser.add_argument('--source', help="path to file or 'webcam'")
    
    args = parser.parse_args()

    # Path handling for Windows users (Removes extra quotes)
    model_fixed = args.model.strip('"').strip("'")
    
    app = WasteDetectorApp(model_fixed)
    if not app.load_model(): return

    if args.source:
        src = args.source.strip('"')
        if src.lower() == "webcam":
            app.process_webcam()
        elif src.lower().endswith(('.mp4', '.avi')):
            app.process_video(src)
        else:
            # For images, tracking isn't needed much, but we use draw_detections
            img = cv2.imread(src)
            dets = app.detector.detect(img)
            res = utils.draw_detections(img, dets)
            cv2.imshow("Detection", utils.resize_for_display(res))
            cv2.waitKey(0)
    else:
        # Interactive Menu
        print("\n♻️  WASTE DETECTION SYSTEM v2.0")
        print("1. Webcam (Live Tracking)\n2. Video File\n3. Single Image")
        ch = input("Selection: ")
        
        if ch == '1': app.process_webcam()
        elif ch == '2':
            v_path = input("Video Path: ").strip('"')
            app.process_video(v_path)
        elif ch == '3':
            i_path = input("Image Path: ").strip('"')
            img = cv2.imread(i_path)
            dets = app.detector.detect(img)
            res = utils.draw_detections(img, dets)
            cv2.imshow("Result", utils.resize_for_display(res))
            cv2.waitKey(0)

if __name__ == "__main__":
    main()