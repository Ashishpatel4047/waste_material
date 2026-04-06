import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

# Custom Modules
from detector import Detector
from tracker import Tracker
import utils  # Jisme naye drawing functions hain

# Page setup
st.set_page_config(page_title="AI Waste Monitor", layout="wide", page_icon="♻️")

# Model & Tracker Caching
@st.cache_resource
def load_assets(model_path):
    try:
        det = Detector(model_path)
        # Tracker ko cache mein rakhein taaki state bani rahe
        track = Tracker(max_disappeared=15, max_distance=80)
        return det, track
    except Exception as e:
        st.error(f"❌ Assets load karne mein dikkat: {e}")
        return None, None

# Settings & Assets
model_path = "weights/best.pt"
detector, tracker = load_assets(model_path)

# Sidebar
st.sidebar.header("⚙️ Control Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
show_id = st.sidebar.checkbox("Show Tracking IDs", value=True)
show_legend = st.sidebar.checkbox("Show Color Legend", value=True)

if st.sidebar.button("🔄 Reset Tracker"):
    tracker.reset()
    st.sidebar.success("Tracker Reset Done!")

# UI Header
st.title("♻️ AI Waste Detection & Tracking")
st.markdown("Automated waste classification using YOLO & Centroid Tracking")

if detector is None:
    st.stop()

# 📌 Tabs
tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Live Video Tracking"])

# ================= IMAGE DETECTION =================
with tab1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"], key="img_up")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", channels="BGR", use_container_width=True)

        if st.button("🚀 Run Analysis"):
            with st.spinner('Analyzing...'):
                detections = detector.detect(img, conf_threshold)
                # Image ke liye direct detections use karein (Tracking ki zaroorat nahi)
                res_img = utils.draw_detections(img, detections)
                
                if show_legend:
                    class_names = getattr(detector, 'classes', ["Waste"])
                    res_img = utils.draw_legend(res_img, class_names)

                col2.image(res_img, caption="Analysis Result", channels="BGR", use_container_width=True)

                # Summary Metrics
                if detections:
                    from collections import Counter
                    counts = Counter([d['class_name'] for d in detections])
                    m_cols = st.columns(len(counts))
                    for i, (name, count) in enumerate(counts.items()):
                        m_cols[i].metric(name, count)
                else:
                    st.info("No objects detected.")

# ================= VIDEO TRACKING =================
with tab2:
    st.subheader("Upload Video")
    video_file = st.file_uploader("Choose Video", type=["mp4", "avi", "mov"], key="vid_up")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()

        if st.button("🚀 Start Tracking"):
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            st_stats = st.empty() # For real-time counts
            
            # Reset tracker for new video
            tracker.reset()
            prev_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 1. Detection & Tracking
                detections = detector.detect(frame, conf_threshold)
                tracked_objects = tracker.update(detections)

                # 2. Visualization
                # IDs aur Boxes draw karein
                annotated = utils.draw_boxes(frame, tracked_objects, show_id=show_id)
                
                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                annotated = utils.draw_fps(annotated, fps)

                # Legend
                if show_legend:
                    class_names = getattr(detector, 'classes', ["Waste"])
                    annotated = utils.draw_legend(annotated, class_names)

                # 3. Display
                st_frame.image(annotated, channels="BGR")
                
                # Dynamic Stats Update
                with st_stats.container():
                    st.write(f"**Total Unique Objects Detected:** `{tracker.next_object_id}`")

            cap.release()
            os.unlink(tfile.name)
            st.success("✅ Video Processing Finished")

# Custom Styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 18px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)