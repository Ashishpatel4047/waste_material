import cv2
import numpy as np

# 🎨 Class ID ke hisaab se unique colors
def get_color(cls_id):
    """Har class ke liye alag color generate karein"""
    colors = [
        (255, 100, 0),   # Blue-ish
        (0, 255, 100),   # Green-ish
        (100, 0, 255),   # Red-ish
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (255, 128, 0),   # Orange
        (0, 128, 255),   # Azure
    ]
    return colors[cls_id % len(colors)]

def draw_boxes(frame, objects, show_confidence=True, show_id=True):
    """Tracked objects par boxes aur IDs draw karein (Video/Webcam ke liye)"""
    if frame is None: return None
    frame_copy = frame.copy()
    
    for obj_id, obj_data in objects.items():
        if isinstance(obj_data, dict):
            bbox = obj_data.get('bbox')
            det = obj_data.get('detection', {})
            cx, cy = obj_data.get('centroid', (0, 0))
        else:
            cx, cy, bbox, det = obj_data
            
        x1, y1, x2, y2 = map(int, bbox)
        cls_id = det.get("class_id", 0)
        color = get_color(cls_id)
        
        label_text = []
        if show_id: label_text.append(f"ID:{obj_id}")
        if det.get("class_name"): label_text.append(det["class_name"])
        if show_confidence and det.get("confidence"): label_text.append(f"{det['confidence']:.2f}")
        
        full_label = " ".join(label_text)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame_copy, full_label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame_copy, (cx, cy), 4, (0, 0, 255), -1)
        
    return frame_copy

def draw_detections(frame, detections, show_confidence=True):
    """Bina tracking ke sirf boxes draw karein (Single Image ke liye)"""
    if frame is None: return None
    frame_copy = frame.copy()
    
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        class_name = det.get("class_name", "Unknown")
        confidence = det.get("confidence", 0)
        class_id = det.get("class_id", 0)
        
        color = get_color(class_id)
        label = f"{class_name} {confidence:.2f}" if show_confidence else class_name
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame_copy, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
    return frame_copy

def draw_legend(frame, class_names):
    """Screen par legend box dikhayein"""
    if frame is None or not class_names: return frame
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    x_pos, y_pos = w - 180, 40
    
    overlay = frame_copy.copy()
    legend_h = len(class_names) * 30 + 20
    cv2.rectangle(overlay, (x_pos - 10, y_pos - 30), (w - 10, y_pos + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)

    cv2.putText(frame_copy, "WASTE TYPES", (x_pos, y_pos - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for i, name in enumerate(class_names):
        color = get_color(i)
        y = y_pos + (i * 30) + 15
        cv2.rectangle(frame_copy, (x_pos, y - 10), (x_pos + 15, y), color, -1)
        cv2.putText(frame_copy, name.upper(), (x_pos + 25, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame_copy

def draw_statistics(frame, stats, position=(20, 80)):
    """Live stats (counts etc.) draw karein"""
    if frame is None: return frame
    y_offset = position[1]
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (position[0], y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    return frame

def draw_fps(frame, fps):
    """FPS counter draw karein"""
    if frame is None: return None
    text = f"Speed: {fps:.1f} FPS"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def resize_for_display(frame, width=1000):
    """Display ke liye frame size adjust karein"""
    if frame is None: return None
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_h = int(width * aspect_ratio)
    return cv2.resize(frame, (width, new_h))