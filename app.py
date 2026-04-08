import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import Counter

from detector import Detector
from tracker import Tracker
import utils

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Waste Monitor", layout="wide", page_icon="♻️")

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "history_data" not in st.session_state:
    st.session_state.history_data = []

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets(model_path):
    try:
        return Detector(model_path), Tracker(max_disappeared=15, max_distance=80)
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None, None

detector, tracker = load_assets("weights/best.pt")

if detector is None:
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Control Panel")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
show_id = st.sidebar.checkbox("Show Tracking ID", True)
show_legend = st.sidebar.checkbox("Show Legend", True)

if st.sidebar.button("🔄 Reset Tracker"):
    tracker.reset()

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history_data = []

# ---------------- HEADER ----------------
st.title("♻️ AI Waste Detection & Tracking")

tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Tracking"])

# =====================================================
# 🖼️ IMAGE ANALYSIS
# =====================================================
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        col1, col2 = st.columns(2)
        col1.image(img, width="stretch")

        if st.button("🚀 Run Detection"):
            detections = detector.detect(img, conf_threshold)
            result_img = utils.draw_detections(img.copy(), detections)

            if show_legend:
                class_names = getattr(detector, 'classes', ["Waste"])
                result_img = utils.draw_legend(result_img, class_names)

            col2.image(result_img, width="stretch")

            if detections:
                counts = Counter([d['class_name'] for d in detections if isinstance(d, dict)])

                st.subheader("📊 Detection Summary")
                cols = st.columns(len(counts))
                for i, (name, count) in enumerate(counts.items()):
                    cols[i].metric(name, count)

                top_class = max(counts, key=counts.get)
                st.success(f"🔥 Most Detected: {top_class}")

                st.session_state.history_data.append({
                    "type": "Image",
                    "time": time.strftime("%H:%M:%S"),
                    "counts": dict(counts),
                    "image": result_img.copy()
                })
            else:
                st.info("No objects detected")

# =====================================================
# 🎥 VIDEO TRACKING (FIXED)
# =====================================================
with tab2:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    colA, colB = st.columns(2)
    if colA.button("▶️ Start"):
        st.session_state.run = True
    if colB.button("⛔ Stop"):
        st.session_state.run = False

    if video_file:
        # ✅ Create temp file safely
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()  # 🔥 IMPORTANT FIX

        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()
        stats = st.empty()

        tracker.reset()
        prev_time = time.time()
        frame_count = 0

        try:
            while cap.isOpened() and st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                detections = detector.detect(frame, conf_threshold)
                tracked = tracker.update(detections)

                annotated = utils.draw_boxes(frame.copy(), tracked, show_id)

                fps = 1 / (time.time() - prev_time)
                prev_time = time.time()

                annotated = utils.draw_fps(annotated, fps)

                if show_legend:
                    class_names = getattr(detector, 'classes', ["Waste"])
                    annotated = utils.draw_legend(annotated, class_names)

                st_frame.image(annotated, channels="BGR", width="stretch")

                counts = Counter([d['class_name'] for d in detections if isinstance(d, dict)])

                stats.write(f"🧠 Objects: {tracker.next_object_id, tracker.class_names} | ⚡ FPS: {int(fps)}")

                # Save history every 60 frames
                if frame_count % 60 == 0:
                    st.session_state.history_data.append({
                        "type": "Video",
                        "time": time.strftime("%H:%M:%S"),
                        "counts": dict(counts),
                        "image": annotated.copy()
                    })

        finally:
            cap.release()

            # ✅ Safe delete (no crash)
            try:
                os.unlink(tfile.name)
            except PermissionError:
                pass

        st.success("✅ Video Processing Completed")

# =====================================================
# 📂 HISTORY
# =====================================================
st.markdown("## 📂 Detection History")

if st.session_state.history_data:
    for item in reversed(st.session_state.history_data):
        with st.expander(f"{item['type']} | {item['time']}"):

            col1, col2 = st.columns([2, 1])

            if "image" in item:
                col1.image(item["image"], channels="BGR", width="stretch")

            col2.write("### 📊 Counts")
            col2.json(item["counts"])

else:
    st.info("No history available")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Made with ❤️ using Streamlit | AI Waste Monitoring System
</p>
""", unsafe_allow_html=True)