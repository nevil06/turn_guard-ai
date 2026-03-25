"""
SafeTurn AI — Single-Camera Pedestrian Signal Controller
=========================================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

ONE camera, ONE window — detects pedestrians and controls a 3-stage signal.

Signal States:
  🟢 GREEN  → normal flow
  🟡 ORANGE → warning — prepare to stop (2 seconds)
  🔴 RED    → stop vehicles

Pedestrian Logic:
  0 pedestrians         → Stay GREEN
  1 pedestrian          → Wait 5s, then ORANGE → RED
  2+ pedestrians        → Immediately ORANGE → RED

Transition Rules:
  GREEN → ORANGE → RED (never skips a stage)
  ORANGE duration = 2 seconds (fixed)
  RED minimum     = 3 seconds
  RED → GREEN only when 0 pedestrians

Controls:
  Q — Quit
  A — Simulate accident (RED + Ambulance Alert for 5s)
  S — Screenshot

Usage:
  python step2_dual_camera.py --video test4vedio.mp4
  python step2_dual_camera.py --camera 0
"""

import sys
import time
import argparse
import numpy as np

# ─── OpenCV ────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.  Fix: pip install opencv-python")
    sys.exit(1)

# ─── YOLOv8 ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[FATAL] ultralytics not found.  Fix: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

YOLO_MODEL_PATH      = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.40
PERSON_CLASS_ID      = 0

# Signal timing (seconds)
ORANGE_DURATION      = 2.0         # ORANGE lasts exactly 2s
RED_MIN_DURATION     = 3.0         # RED holds at least 3s
SINGLE_PED_WAIT      = 5.0         # Wait 5s before triggering for 1 person
ACCIDENT_DURATION    = 5.0         # Accident override

# Colors (BGR)
COL_RED    = (0, 0, 255)
COL_GREEN  = (0, 200, 0)
COL_ORANGE = (0, 140, 255)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_CYAN   = (255, 255, 0)
COL_GRAY   = (180, 180, 180)

# Map state → color
STATE_COLORS = {
    "GREEN":  COL_GREEN,
    "ORANGE": COL_ORANGE,
    "RED":    COL_RED,
}


# ═══════════════════════════════════════════════════════════════════
# 3-STAGE SIGNAL CONTROLLER
# ═══════════════════════════════════════════════════════════════════
#
#  State machine:
#
#    GREEN ──(pedestrians detected)──→ ORANGE ──(2s)──→ RED
#      ▲                                                 │
#      └─────────(0 pedestrians + timer done)────────────┘
#
# ═══════════════════════════════════════════════════════════════════

class SignalController:
    def __init__(self):
        self.state            = "GREEN"
        self.state_start      = time.time()

        # Tracking when a single pedestrian first appeared
        self.single_ped_since = None    # timestamp when 1 ped was first seen

        # Accident override
        self.accident         = False
        self.accident_start   = 0.0

    def trigger_accident(self):
        """Force immediate RED for ACCIDENT_DURATION seconds."""
        self.accident       = True
        self.accident_start = time.time()
        self.state          = "RED"
        self.state_start    = time.time()

    def update(self, ped_count):
        """
        Update signal based on pedestrian count.
        Returns: (state_str, color_bgr, remaining_seconds, status_text)
        """
        now = time.time()
        elapsed = now - self.state_start

        # ── Accident override ─────────────────────────────────────
        if self.accident:
            t = now - self.accident_start
            if t >= ACCIDENT_DURATION:
                self.accident = False
                # After accident, go back to GREEN if clear
                if ped_count == 0:
                    self._go("GREEN", now)
                else:
                    self._go("RED", now)
            else:
                remaining = ACCIDENT_DURATION - t
                return "RED", COL_RED, remaining, "ACCIDENT — ALL STOP"

        # ══════════════════════════════════════════════════════════
        # STATE: GREEN
        # ══════════════════════════════════════════════════════════
        if self.state == "GREEN":
            if ped_count == 0:
                # All clear — stay green, reset single-ped timer
                self.single_ped_since = None
                return "GREEN", COL_GREEN, 0.0, "No pedestrians — FREE TURN"

            elif ped_count == 1:
                # 1 pedestrian — start the 5s wait timer
                if self.single_ped_since is None:
                    self.single_ped_since = now

                waited = now - self.single_ped_since
                if waited >= SINGLE_PED_WAIT:
                    # 5 seconds passed, still there → trigger ORANGE
                    self.single_ped_since = None
                    self._go("ORANGE", now)
                    return "ORANGE", COL_ORANGE, ORANGE_DURATION, "Warning — prepare to stop"
                else:
                    # Still waiting
                    wait_left = SINGLE_PED_WAIT - waited
                    return "GREEN", COL_GREEN, wait_left, \
                        f"1 pedestrian — watching ({wait_left:.1f}s)"

            else:
                # 2+ pedestrians — immediately trigger ORANGE
                self.single_ped_since = None
                self._go("ORANGE", now)
                return "ORANGE", COL_ORANGE, ORANGE_DURATION, \
                    f"{ped_count} pedestrians — WARNING"

        # ══════════════════════════════════════════════════════════
        # STATE: ORANGE (fixed 2-second warning)
        # ══════════════════════════════════════════════════════════
        elif self.state == "ORANGE":
            if elapsed < ORANGE_DURATION:
                remaining = ORANGE_DURATION - elapsed
                return "ORANGE", COL_ORANGE, remaining, "Warning — prepare to stop"
            else:
                # ORANGE done → switch to RED
                self._go("RED", now)
                return "RED", COL_RED, RED_MIN_DURATION, "STOP — pedestrians crossing"

        # ══════════════════════════════════════════════════════════
        # STATE: RED
        # ══════════════════════════════════════════════════════════
        elif self.state == "RED":
            if elapsed < RED_MIN_DURATION:
                remaining = RED_MIN_DURATION - elapsed
                return "RED", COL_RED, remaining, "STOP — pedestrians crossing"

            # Minimum RED done — can we go GREEN?
            if ped_count == 0:
                self._go("GREEN", now)
                return "GREEN", COL_GREEN, 0.0, "No pedestrians — FREE TURN"
            else:
                # Still pedestrians → stay RED
                return "RED", COL_RED, 0.0, \
                    f"{ped_count} pedestrian(s) — holding RED"

        # Fallback (should never reach here)
        return self.state, STATE_COLORS.get(self.state, COL_WHITE), 0.0, ""

    def _go(self, new_state, now):
        """Transition to a new state."""
        self.state       = new_state
        self.state_start = now


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION — persons only
# ═══════════════════════════════════════════════════════════════════

def detect_persons(model, frame):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    persons = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            persons.append((x1, y1, x2, y2, conf))
    return persons


# ═══════════════════════════════════════════════════════════════════
# DRAWING — Single clean overlay
# ═══════════════════════════════════════════════════════════════════

def draw_overlay(frame, ped_count, detections, sig_state, sig_color,
                 remaining, status_text, accident_active, fps):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Bounding boxes ────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_CYAN, 2)
        cv2.putText(frame, f"Person {conf:.0%}", (x1, y1 - 8),
                    font, 0.45, COL_CYAN, 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, COL_CYAN, -1)

    # ── Top banner ────────────────────────────────────────────────
    banner_h = 60
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, banner_h), sig_color, -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Signal: {sig_state}", (12, 42),
                font, 1.2, COL_WHITE, 3)

    # Timer
    if remaining > 0:
        cv2.putText(frame, f"{remaining:.1f}s", (w - 110, 42),
                    font, 1.0, COL_WHITE, 2)

    # ── Info below banner ─────────────────────────────────────────
    cv2.putText(frame, f"Pedestrians: {ped_count}", (12, banner_h + 25),
                font, 0.65, COL_WHITE, 2)
    cv2.putText(frame, status_text, (12, banner_h + 52),
                font, 0.55, sig_color, 2)

    # ── 3-Stage Signal Light (top-right) ──────────────────────────
    lx = w - 50
    # Housing
    cv2.rectangle(frame, (lx - 32, 5), (lx + 32, 115), (25, 25, 25), -1)
    cv2.rectangle(frame, (lx - 32, 5), (lx + 32, 115), (80, 80, 80), 2)

    # 3 circles: RED, ORANGE, GREEN (top to bottom)
    lights = [
        (lx, 25,  COL_RED,    sig_state == "RED"),
        (lx, 58,  COL_ORANGE, sig_state == "ORANGE"),
        (lx, 91,  COL_GREEN,  sig_state == "GREEN"),
    ]
    for (cx, cy, color, active) in lights:
        if active:
            cv2.circle(frame, (cx, cy), 16, color, -1)
            cv2.circle(frame, (cx, cy), 16, COL_WHITE, 2)
        else:
            # Dim inactive light
            dim = tuple(max(c // 5, 15) for c in color)
            cv2.circle(frame, (cx, cy), 16, dim, -1)
            cv2.circle(frame, (cx, cy), 16, (50, 50, 50), 1)

    # ── Accident overlay ──────────────────────────────────────────
    if accident_active:
        flash = int(time.time() * 3) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COL_RED, 6)

        bx, by = w // 2, h // 2
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx - 220, by - 60), (bx + 220, by + 75),
                      (0, 0, 120), -1)
        cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, "ACCIDENT DETECTED", (bx - 185, by - 18),
                    font, 1.1, COL_WHITE, 3)
        cv2.putText(frame, "ALL SIGNALS RED", (bx - 120, by + 18),
                    font, 0.8, COL_RED, 2)
        cv2.putText(frame, "Ambulance Alert Triggered",
                    (bx - 165, by + 55), font, 0.65, COL_CYAN, 2)

    # ── Bottom bar ────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Q=Quit  A=Accident  S=Screenshot",
                (8, h - 12), font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, h - 12),
                font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, "SafeTurn AI | HackSETU 2025",
                (w - 260, h - 40), font, 0.45, COL_CYAN, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI — Single-Camera Pedestrian Signal Controller"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Video file path")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam index (e.g. 0)")
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH,
                        help="YOLOv8 model path")
    args = parser.parse_args()

    if args.video is None and args.camera is None:
        print("[INFO] No input specified. Defaulting to --camera 0")
        args.camera = 0

    # ── Load YOLO ─────────────────────────────────────────────────
    print(f"[INFO] Loading YOLOv8: {args.model}")
    model = YOLO(args.model)
    print("[INFO] YOLOv8 loaded OK.")

    # ── Open camera or video ──────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Webcam {args.camera}"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source_name}")
        sys.exit(1)
    print(f"[INFO] Source: {source_name}")

    # ── Init ──────────────────────────────────────────────────────
    signal    = SignalController()
    prev_time = time.time()
    fps       = 0.0

    print()
    print("=" * 55)
    print("  SafeTurn AI — 3-Stage Signal Controller")
    print("  🟢 GREEN → 🟡 ORANGE → 🔴 RED")
    print("  Q = Quit | A = Accident | S = Screenshot")
    print("=" * 55)
    print()

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[WARN] Camera read failed.")
                break

        # Detect pedestrians
        detections = detect_persons(model, frame)
        ped_count  = len(detections)

        # Update signal
        sig_state, sig_color, remaining, status_text = signal.update(ped_count)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Draw overlay
        frame = draw_overlay(
            frame, ped_count, detections,
            sig_state, sig_color, remaining, status_text,
            signal.accident, fps
        )

        # Display
        cv2.imshow("SafeTurn AI", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("[INFO] Quit.")
            break
        elif key == ord('a') or key == ord('A'):
            signal.trigger_accident()
            print(f"[ALERT] Accident simulated! RED for {ACCIDENT_DURATION:.0f}s")
        elif key == ord('s') or key == ord('S'):
            fname = f"safeturn_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] SafeTurn AI finished.")


if __name__ == "__main__":
    main()
