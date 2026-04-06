import cv2
import os
import sys
# detector.py se Detector class import kar rahe hain
try:
    from detector import Detector
except ImportError:
    print("❌ detector.py file nahi mili! Make sure dono files ek hi folder mein hain.")
    sys.exit()

def run_inference():
    # --- CONFIGURATION ---
    model_path = r"C:\Users\91840\waste_material\weights\best.pt"
    image_path = r"C:\Users\91840\waste_material\dataset\test\images\62b308b0205ab62b308b0205b3_frame44_jpg.rf.5826b5c938968c8362a8ebf3cfdf79b2.jpg"
    
    # 1. Path Verification
    if not os.path.exists(model_path):
        print(f"❌ Model path galat hai: {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"❌ Image path galat hai: {image_path}")
        return

    # 2. Load Detector
    print("🚀 Loading model...")
    try:
        detector = Detector(model_path)
    except Exception as e:
        print(f"❌ Detector load karne mein error: {e}")
        return

    # 3. Read Image
    print("📸 Reading image...")
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image read nahi ho paayi (Empty image)")
        return

    # 4. Multi-Threshold Testing (Debug Mode)
    # Kabhi-kabhi custom model low confidence par result dete hain
    thresholds = [0.25, 0.15, 0.05]
    detections = []
    final_threshold = 0

    print("🔍 Searching for objects...")
    for t in thresholds:
        detections = detector.detect(img, conf_threshold=t)
        if len(detections) > 0:
            final_threshold = t
            print(f"✅ Success! Threshold {t} par {len(detections)} objects mile.")
            break
        else:
            print(f"ℹ️ Threshold {t}: Koi object nahi mila.")

    # 5. Final Processing
    if len(detections) == 0:
        print("⚠️ Bilkul kuch detect nahi hua. Check karein ki kya model sahi train hua hai?")
        # Khali image hi dikha dete hain bina boxes ke
        result = img.copy()
    else:
        print(f"🎯 Total {len(detections)} objects identified.")
        for i, det in enumerate(detections, 1):
            print(f"   [{i}] {det['class_name']} - Conf: {det['confidence']} - Box: {det['bbox']}")
        
        # Draw bounding boxes
        result = detector.draw_detections(img, detections)

    # 6. Save and Show
    output_path = "output_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"💾 Result save kar diya gaya hai: {output_path}")

    # Window ko resizeable banayein taki screen se bahar na jaye
    try:
        window_name = f"Inference Result (Thresh: {final_threshold})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result)
        print("\n💡 Image window par koi bhi key dabayein band karne ke liye.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"⚠️ Display error (Headless environment): {e}")

if __name__ == "__main__":
    run_inference()