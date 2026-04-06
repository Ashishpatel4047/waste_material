import math
import cv2
import numpy as np
from collections import defaultdict

class Tracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_object_id = 0
        self.objects = {}  # id -> {centroid, bbox, detection, disappeared}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def get_centroid(self, bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        return (cx, cy)

    def calculate_distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def reset(self):
        """Tracker state ko poori tarah clear karne ke liye"""
        self.next_object_id = 0
        self.objects = {}
        print("🔄 Tracker has been reset.")

    def update(self, detections):
        # Agar koi detection nahi hai, toh purane objects ko "disappeared" mark karein
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    del self.objects[object_id]
            return self.objects

        # New detections ke centroids nikalein
        input_centroids = [self.get_centroid(d["bbox"]) for d in detections]

        # Agar tracker khali hai, sabko naye IDs dein
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register_object(input_centroids[i], detections[i])
            return self.objects

        # Matching logic
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]

        used_detections = set()
        used_objects = set()

        # Simple Greedy Matching
        for i, obj_centroid in enumerate(object_centroids):
            distances = [self.calculate_distance(obj_centroid, ic) for ic in input_centroids]
            if not distances: break
            
            min_dist_idx = np.argmin(distances)
            
            if distances[min_dist_idx] < self.max_distance and min_dist_idx not in used_detections:
                obj_id = object_ids[i]
                self.objects[obj_id]['centroid'] = input_centroids[min_dist_idx]
                self.objects[obj_id]['bbox'] = detections[min_dist_idx]['bbox']
                self.objects[obj_id]['detection'] = detections[min_dist_idx]
                self.objects[obj_id]['disappeared'] = 0
                used_detections.add(min_dist_idx)
                used_objects.add(obj_id)

        # Jo match nahi huye unhe register karein (New Objects)
        for i in range(len(input_centroids)):
            if i not in used_detections:
                self.register_object(input_centroids[i], detections[i])

        # Jo objects frames mein nahi mile unhe remove karein (Disappeared logic)
        for obj_id in list(self.objects.keys()):
            if obj_id not in used_objects:
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]

        return self.objects

    def register_object(self, centroid, detection):
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': detection['bbox'],
            'detection': detection,
            'disappeared': 0
        }
        self.next_object_id += 1