"""
SafeTurn AI — Detection Module
================================
YOLOv8-based person and vehicle detection.

Optimized for CPU:
  - Frames resized to 640×360 before inference
  - Detections scaled back to original resolution
  - Lightweight YOLOv8n model
"""

import sys

try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.  Fix: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[FATAL] ultralytics not found.  Fix: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.35       # Slightly lower — temporal filter catches false positives
PERSON_CLASS_ID      = 0
VEHICLE_CLASS_IDS    = {1, 2, 3, 5, 7}   # bicycle, car, motorbike, bus, truck
VEHICLE_NAMES        = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# Inference resolution (smaller = faster)
INFER_WIDTH  = 640
INFER_HEIGHT = 360
YOLO_IMGSZ   = 640                # YOLO internal size (640 = best accuracy)

# Stability
STABILITY_FRAMES = 3              # Require N consistent frames before changing count


# ═══════════════════════════════════════════════════════════════════
# DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════

class Detector:
    """
    Wraps YOLOv8 with frame resizing for speed.

    Usage:
        det = Detector("yolov8n.pt")
        persons, vehicles = det.detect(frame)
    """

    def __init__(self, model_path="yolov8n.pt"):
        print(f"[INFO] Loading YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        print("[INFO] YOLOv8 loaded OK.")

    def detect(self, frame):
        """
        Detect persons and vehicles in a frame.

        Args:
            frame: original-resolution BGR frame

        Returns:
            persons:  list of (x1, y1, x2, y2, conf)
            vehicles: list of (x1, y1, x2, y2, conf, class_name)
        """
        # Let YOLO handle resizing internally (imgsz controls this)
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                             imgsz=YOLO_IMGSZ)

        persons  = []
        vehicles = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if cls == PERSON_CLASS_ID:
                    persons.append((x1, y1, x2, y2, conf))
                elif cls in VEHICLE_CLASS_IDS:
                    name = VEHICLE_NAMES.get(cls, "vehicle")
                    vehicles.append((x1, y1, x2, y2, conf, name))

        return persons, vehicles

    def detect_persons_only(self, frame):
        """Detect only persons (for pedestrian camera in dual mode)."""
        persons, _ = self.detect(frame)
        return persons


# ═══════════════════════════════════════════════════════════════════
# STABLE DETECTOR — Temporal filtering to prevent flickering
# ═══════════════════════════════════════════════════════════════════

class StableDetector:
    """
    Wraps Detector with 3-frame temporal filtering.

    The raw YOLO ped count can jump frame-to-frame (e.g. 0→2→0→1).
    This wrapper smooths it:
      - Count only INCREASES if the new count persists for 3 frames
      - Count only DECREASES if the lower count persists for 3 frames

    Result: signal decisions are based on a stable, non-flickering count.
    """

    def __init__(self, model_path="yolov8n.pt"):
        self.detector = Detector(model_path)
        self.ped_history = []         # last N raw ped counts
        self.stable_ped_count = 0     # smoothed output

        # Store last raw detections for drawing bounding boxes
        self.last_persons  = []
        self.last_vehicles = []

    def detect(self, frame):
        """
        Run detection + update temporal filter.
        Returns raw (persons, vehicles) for drawing bounding boxes.
        Use get_stable_ped_count() for signal decisions.
        """
        persons, vehicles = self.detector.detect(frame)
        self.last_persons  = persons
        self.last_vehicles = vehicles

        # ── Temporal filter ───────────────────────────────────────
        raw_count = len(persons)
        self.ped_history.append(raw_count)
        if len(self.ped_history) > STABILITY_FRAMES:
            self.ped_history.pop(0)

        if len(self.ped_history) >= STABILITY_FRAMES:
            recent_min = min(self.ped_history)
            recent_max = max(self.ped_history)

            # Count goes UP only if ALL recent frames show higher
            if recent_min > self.stable_ped_count:
                self.stable_ped_count = recent_min

            # Count goes DOWN only if ALL recent frames show lower
            elif recent_max < self.stable_ped_count:
                self.stable_ped_count = recent_max
        else:
            self.stable_ped_count = raw_count

        return persons, vehicles

    def detect_persons_only(self, frame):
        """Detect only persons (for pedestrian camera in dual mode)."""
        persons, _ = self.detect(frame)
        return persons

    def get_stable_ped_count(self):
        """Get the temporally-filtered pedestrian count (use for signal decisions)."""
        return self.stable_ped_count
