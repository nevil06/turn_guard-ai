"""
SafeTurn AI — Main Application
================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

DEMO-READY system with ZONE-BASED pedestrian detection:

  WAITING ZONE  (footpath)      → prepare to stop (3s → ORANGE → RED)
  CROSSING ZONE (zebra crossing) → immediate ORANGE → RED
  No pedestrians                 → GREEN (free turn)

Modes:
  --mock          Synthetic junction (no camera/video needed)
  --mode single   One camera: detects pedestrians + vehicles
  --mode dual     Two cameras: webcam (pedestrians) + video (traffic)

Controls:
  Q — Quit    S — Screenshot    R — Rewind    A — Accident
"""

import sys
import time
import math
import argparse
import random
import numpy as np

try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.  Fix: pip install opencv-python")
    sys.exit(1)

from signal_controller import SignalController, SimpleSignalController, COL_RED, COL_GREEN, COL_ORANGE


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PROCESS_EVERY     = 2             # Run YOLO every 2nd frame
ACCIDENT_DURATION = 5.0           # Accident override (seconds)

# ── ZONE DEFINITIONS (fractions of frame size 0.0–1.0) ────────────
# These scale automatically with any resolution.
# Adjust these values to match your camera view.

WAIT_ZONE_LEFT = {
    "x1": 0.00, "y1": 0.30,
    "x2": 0.20, "y2": 0.85,
}

WAIT_ZONE_RIGHT = {
    "x1": 0.55, "y1": 0.30,
    "x2": 0.75, "y2": 0.85,
}

CROSS_ZONE = {
    "x1": 0.20, "y1": 0.30,
    "x2": 0.55, "y2": 0.55,
}

# Colors
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_CYAN   = (255, 255, 0)
COL_GRAY   = (180, 180, 180)
COL_BLUE   = (255, 150, 50)       # Waiting zone (blue)
COL_ZONE_R = (80, 80, 255)        # Crossing zone (red-ish)

# Class-specific colors
COL_PERSON = (0, 220, 0)
COL_CAR    = (0, 0, 255)
COL_BIKE   = (0, 165, 255)
COL_BUS    = (255, 0, 200)
COL_TRUCK  = (200, 150, 0)

VEHICLE_COLORS = {
    "car": COL_CAR, "bicycle": COL_BIKE, "motorbike": COL_BIKE,
    "bus": COL_BUS, "truck": COL_TRUCK,
}

SIGNAL_LABELS = {
    "GREEN":  "FREE TURN",
    "ORANGE": "PREPARE TO STOP",
    "RED":    "STOP",
}


# ═══════════════════════════════════════════════════════════════════
# ZONE UTILITIES
# ═══════════════════════════════════════════════════════════════════

def zone_to_pixels(zone, frame_w, frame_h):
    """Convert fractional zone → pixel coords (x1, y1, x2, y2)."""
    return (
        int(zone["x1"] * frame_w), int(zone["y1"] * frame_h),
        int(zone["x2"] * frame_w), int(zone["y2"] * frame_h),
    )


def point_in_zone(cx, cy, zx1, zy1, zx2, zy2):
    """Check if point (cx, cy) is inside the zone rectangle."""
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def classify_pedestrians(persons, frame_w, frame_h):
    """
    Classify each detected person into WAITING (left or right), CROSSING, or OUTSIDE.

    Returns: (waiting_count, crossing_count, wait_persons, cross_persons)
    """
    wzl = zone_to_pixels(WAIT_ZONE_LEFT, frame_w, frame_h)
    wzr = zone_to_pixels(WAIT_ZONE_RIGHT, frame_w, frame_h)
    cz = zone_to_pixels(CROSS_ZONE, frame_w, frame_h)

    waiting_count = 0
    crossing_count = 0
    wait_persons = []
    cross_persons = []

    for p in persons:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if point_in_zone(cx, cy, cz[0], cz[1], cz[2], cz[3]):
            crossing_count += 1
            cross_persons.append(p)
        elif point_in_zone(cx, cy, wzl[0], wzl[1], wzl[2], wzl[3]) or \
             point_in_zone(cx, cy, wzr[0], wzr[1], wzr[2], wzr[3]):
            waiting_count += 1
            wait_persons.append(p)

    return waiting_count, crossing_count, wait_persons, cross_persons


# ═══════════════════════════════════════════════════════════════════
# MOCK DETECTION GENERATOR — Synthetic junction for demo
# ═══════════════════════════════════════════════════════════════════

class MockDetectionGenerator:
    """
    Simulates a junction with pedestrians moving through
    WAITING → CROSSING zones, plus vehicles making turns.
    """

    def __init__(self, w=960, h=540):
        self.w, self.h = w, h
        self.frame_num = 0

        # Ped 1: starts in WAITING zone, moves into CROSSING zone
        self.ped1 = {"x": w * 0.10, "y": h * 0.50, "vx": 1.6, "vy": -0.2}
        # Ped 2: appears later, starts in WAITING zone
        self.ped2 = {"x": w * 0.08, "y": h * 0.60, "vx": 1.4, "vy": -0.3}
        # Ped 3: starts in RIGHT WAIT zone, walks left
        self.ped3 = {"x": w * 0.65, "y": h * 0.45, "vx": -1.2, "vy": 0.1}

        # Vehicles
        self.car1 = {"x": w * 0.78, "y": h * 0.85, "vx": -3.0, "vy": -2.5}
        self.car2 = {"x": w * 0.85, "y": h * 0.75, "vx": -1.0, "vy": -0.5}
        self.moto = {"x": w * 0.6, "y": h * 0.9, "vx": -2.5, "vy": -3.0}

    def _move(self, obj, nx=0.3, ny=0.2):
        obj["x"] += obj["vx"] + random.gauss(0, nx)
        obj["y"] += obj["vy"] + random.gauss(0, ny)

    def _reset(self, obj, sx, sy, r=30):
        if (obj["x"] < -80 or obj["x"] > self.w + 80 or
                obj["y"] < -80 or obj["y"] > self.h + 80):
            obj["x"] = sx + random.uniform(-r, r)
            obj["y"] = sy + random.uniform(-r, r)

    def generate_detections(self):
        self.frame_num += 1
        persons = []
        vehicles = []

        # Ped 1 — always, starts in wait zone, crosses
        self._move(self.ped1)
        self._reset(self.ped1, self.w * 0.10, self.h * 0.50)
        px, py = int(self.ped1["x"]), int(self.ped1["y"])
        persons.append((px - 15, py - 40, px + 15, py + 40, 0.92))

        # Ped 2 — after frame 40
        if self.frame_num > 40:
            self._move(self.ped2)
            self._reset(self.ped2, self.w * 0.08, self.h * 0.60)
            px2, py2 = int(self.ped2["x"]), int(self.ped2["y"])
            persons.append((px2 - 15, py2 - 40, px2 + 15, py2 + 40, 0.88))

        # Ped 3 — after frame 25
        if self.frame_num > 25:
            self._move(self.ped3, 0.1, 0.1)
            self._reset(self.ped3, self.w * 0.65, self.h * 0.45)
            px3, py3 = int(self.ped3["x"]), int(self.ped3["y"])
            persons.append((px3 - 15, py3 - 40, px3 + 15, py3 + 40, 0.85))

        # Car 1
        self._move(self.car1, 0.5, 0.3)
        self._reset(self.car1, self.w * 0.78, self.h * 0.85)
        cx, cy = int(self.car1["x"]), int(self.car1["y"])
        vehicles.append((cx - 55, cy - 30, cx + 55, cy + 30, 0.95, "car"))

        # Car 2
        if self.frame_num > 50:
            self._move(self.car2, 0.3, 0.2)
            self._reset(self.car2, self.w * 0.85, self.h * 0.75)
            cx2, cy2 = int(self.car2["x"]), int(self.car2["y"])
            vehicles.append((cx2 - 55, cy2 - 30, cx2 + 55, cy2 + 30, 0.90, "car"))

        # Motorcycle
        if self.frame_num % 200 < 130:
            self._move(self.moto, 0.4, 0.4)
            self._reset(self.moto, self.w * 0.6, self.h * 0.9)
            mx, my = int(self.moto["x"]), int(self.moto["y"])
            vehicles.append((mx - 22, my - 20, mx + 22, my + 20, 0.87, "motorbike"))

        return persons, vehicles

    def generate_frame(self):
        f = np.full((self.h, self.w, 3), (55, 55, 55), dtype=np.uint8)

        # Main road
        cv2.rectangle(f, (0, int(self.h * 0.30)),
                      (self.w, int(self.h * 0.70)), (45, 45, 45), -1)
        # Side road
        cv2.rectangle(f, (int(self.w * 0.55), int(self.h * 0.55)),
                      (int(self.w * 0.80), self.h), (45, 45, 45), -1)
        # Crosswalk stripes
        for x in range(int(self.w * 0.20), int(self.w * 0.55), 45):
            cv2.rectangle(f, (x, int(self.h * 0.36)),
                          (x + 22, int(self.h * 0.50)), (190, 190, 190), -1)
        # Free-left arrow
        ax, ay = int(self.w * 0.65), int(self.h * 0.72)
        cv2.arrowedLine(f, (ax, ay + 30), (ax - 40, ay - 10),
                        (80, 100, 80), 2, tipLength=0.3)
        cv2.putText(f, "FREE LEFT", (ax - 45, ay + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 80), 1)
        # Camera label
        cv2.putText(f, "JUNCTION CAM 01", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(f, time.strftime("%H:%M:%S"), (self.w - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return f


# ═══════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════

def draw_zones(frame):
    """Draw WAITING and CROSSING zones with color-coded overlays."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── WAITING ZONE LEFT (blue) ──────────────────────────────────
    wzl = zone_to_pixels(WAIT_ZONE_LEFT, w, h)
    ov = frame.copy()
    cv2.rectangle(ov, (wzl[0], wzl[1]), (wzl[2], wzl[3]), COL_BLUE, -1)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (wzl[0], wzl[1]), (wzl[2], wzl[3]), COL_BLUE, 2)
    cv2.putText(frame, "WAIT LEFT", (wzl[0] + 5, wzl[1] - 8),
                font, 0.55, COL_BLUE, 2)

    # ── WAITING ZONE RIGHT (blue) ─────────────────────────────────
    wzr = zone_to_pixels(WAIT_ZONE_RIGHT, w, h)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (wzr[0], wzr[1]), (wzr[2], wzr[3]), COL_BLUE, -1)
    cv2.addWeighted(ov2, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (wzr[0], wzr[1]), (wzr[2], wzr[3]), COL_BLUE, 2)
    cv2.putText(frame, "WAIT RIGHT", (wzr[0] + 5, wzr[1] - 8),
                font, 0.55, COL_BLUE, 2)

    # ── CROSSING ZONE (red) ──────────────────────────────────────
    cz = zone_to_pixels(CROSS_ZONE, w, h)
    ov3 = frame.copy()
    cv2.rectangle(ov3, (cz[0], cz[1]), (cz[2], cz[3]), COL_ZONE_R, -1)
    cv2.addWeighted(ov3, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (cz[0], cz[1]), (cz[2], cz[3]), COL_ZONE_R, 2)
    cv2.putText(frame, "CROSS ZONE", (cz[0] + 5, cz[1] - 8),
                font, 0.55, COL_ZONE_R, 2)


def draw_signal_light(frame, sig_state):
    """Draw a 3-light traffic signal in the top-right corner."""
    h, w = frame.shape[:2]
    lx = w - 50

    cv2.rectangle(frame, (lx - 32, 5), (lx + 32, 115), (25, 25, 25), -1)
    cv2.rectangle(frame, (lx - 32, 5), (lx + 32, 115), (80, 80, 80), 2)

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
            dim = tuple(max(c // 5, 15) for c in color)
            cv2.circle(frame, (cx, cy), 16, dim, -1)
            cv2.circle(frame, (cx, cy), 16, (50, 50, 50), 1)


def draw_detections(frame, persons, vehicles=None,
                    wait_persons=None, cross_persons=None):
    """Draw bounding boxes — color-coded by zone."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    wait_set = set(id(p) for p in (wait_persons or []))
    cross_set = set(id(p) for p in (cross_persons or []))

    for det in persons:
        x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]

        # Color by zone
        if id(det) in cross_set:
            color = COL_ZONE_R
            zone_tag = " [CROSSING]"
        elif id(det) in wait_set:
            color = COL_BLUE
            zone_tag = " [WAITING]"
        else:
            color = COL_PERSON
            zone_tag = ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"person {conf:.0%}{zone_tag}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, 0.5, COL_BLACK, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)

    if vehicles:
        for det in vehicles:
            x1, y1, x2, y2, conf, name = det[0], det[1], det[2], det[3], det[4], det[5]
            color = VEHICLE_COLORS.get(name, COL_CAR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), font, 0.55, COL_BLACK, 2)


def draw_center_signal(frame, sig_state, sig_color, remaining):
    """Draw large signal indicator in center-bottom."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = SIGNAL_LABELS.get(sig_state, sig_state)

    text_scale = 1.8
    thickness = 4
    (tw, th), _ = cv2.getTextSize(label, font, text_scale, thickness)
    tx = (w - tw) // 2
    ty = h - 80

    pad_x, pad_y = 30, 18
    ov = frame.copy()
    cv2.rectangle(ov, (tx - pad_x, ty - th - pad_y),
                  (tx + tw + pad_x, ty + pad_y), sig_color, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (tx - pad_x, ty - th - pad_y),
                  (tx + tw + pad_x, ty + pad_y), COL_WHITE, 2)

    cv2.putText(frame, label, (tx + 2, ty + 2), font, text_scale, COL_BLACK, thickness + 2)
    cv2.putText(frame, label, (tx, ty), font, text_scale, COL_WHITE, thickness)

    if remaining > 0:
        timer_text = f"{remaining:.1f}s"
        (ttw, _), _ = cv2.getTextSize(timer_text, font, 0.9, 2)
        cv2.putText(frame, timer_text, ((w - ttw) // 2, ty + 35),
                    font, 0.9, COL_WHITE, 2)


def draw_overlay(frame, waiting_count, crossing_count, sig_state, sig_color,
                 remaining, status_text, fps, mode_label, accident_active=False):
    """Draw the main HUD overlay."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    total_peds = waiting_count + crossing_count

    # ── Top banner ────────────────────────────────────────────────
    banner_h = 60
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, banner_h), sig_color, -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Signal: {sig_state}", (12, 42),
                font, 1.2, COL_WHITE, 3)

    if remaining > 0:
        cv2.putText(frame, f"{remaining:.1f}s", (w - 160, 42),
                    font, 1.0, COL_WHITE, 2)

    # ── Zone info below banner ────────────────────────────────────
    y_info = banner_h + 25
    cv2.putText(frame, f"Pedestrians: {total_peds}", (12, y_info),
                font, 0.6, COL_WHITE, 2)
    cv2.putText(frame, f"Waiting: {waiting_count}  |  Crossing: {crossing_count}",
                (12, y_info + 25), font, 0.5, COL_CYAN, 2)
    cv2.putText(frame, status_text, (12, y_info + 50),
                font, 0.5, sig_color, 2)

    # ── Signal light ──────────────────────────────────────────────
    draw_signal_light(frame, sig_state)

    # ── Center signal ─────────────────────────────────────────────
    if not accident_active:
        draw_center_signal(frame, sig_state, sig_color, remaining)

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
    cv2.putText(frame, "Q=Quit  S=Screenshot  A=Accident",
                (8, h - 12), font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}  |  {mode_label}",
                (w - 260, h - 12), font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, "SafeTurn AI | HackSETU 2025",
                (w - 260, h - 40), font, 0.45, COL_CYAN, 1)

    return frame


def draw_traffic_overlay(frame, sig_state, sig_color, remaining, fps, mode_label, accident_active=False):
    """Draw the traffic camera HUD overlay (No pedestrian zone data)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top banner ────────────────────────────────────────────────
    banner_h = 60
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, banner_h), sig_color, -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Signal: {sig_state}", (12, 42),
                font, 1.2, COL_WHITE, 3)

    if remaining > 0:
        cv2.putText(frame, f"{remaining:.1f}s", (w - 160, 42),
                    font, 1.0, COL_WHITE, 2)

    # ── Signal light ──────────────────────────────────────────────
    draw_signal_light(frame, sig_state)

    # ── Center signal ─────────────────────────────────────────────
    if not accident_active:
        draw_center_signal(frame, sig_state, sig_color, remaining)
        
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
    cv2.putText(frame, "Q=Quit  S=Screenshot  A=Accident",
                (8, h - 12), font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}  |  {mode_label}",
                (w - 260, h - 12), font, 0.42, COL_GRAY, 1)
    cv2.putText(frame, "SafeTurn AI | HackSETU 2025",
                (w - 260, h - 40), font, 0.45, COL_CYAN, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# MOCK MODE
# ═══════════════════════════════════════════════════════════════════

def run_mock(args):
    """Synthetic junction demo — no camera needed."""
    mock = MockDetectionGenerator()
    signal = SignalController()
    fps = 30.0

    # Accident simulation
    accident_active = False
    accident_start  = 0.0

    print()
    print("=" * 55)
    print("  SafeTurn AI — MOCK MODE (Zone-Based)")
    print("  🟢 GREEN → 🟡 ORANGE → 🔴 RED")
    print("  Q = Quit | S = Screenshot | A = Accident")
    print("=" * 55)
    print()

    while True:
        frame_start = time.time()

        frame = mock.generate_frame()
        persons, vehicles = mock.generate_detections()

        # ── Zone classification ───────────────────────────────────
        h, w = frame.shape[:2]
        waiting, crossing, wait_p, cross_p = classify_pedestrians(persons, w, h)

        # ── Accident override ─────────────────────────────────────
        if accident_active:
            elapsed_acc = time.time() - accident_start
            if elapsed_acc >= ACCIDENT_DURATION:
                accident_active = False
            else:
                crossing = max(crossing, 1)

        # ── Update signal (zone-based) ────────────────────────────
        sig_state, sig_color, remaining, status_text = signal.update(waiting, crossing)

        if accident_active:
            sig_state = "RED"
            sig_color = COL_RED
            remaining = ACCIDENT_DURATION - (time.time() - accident_start)
            status_text = "ACCIDENT — ALL STOP"

        # ── Smoothed FPS ──────────────────────────────────────────
        now = time.time()
        dt = max(now - frame_start, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # ── Draw ──────────────────────────────────────────────────
        draw_zones(frame)
        draw_detections(frame, persons, vehicles, wait_p, cross_p)
        frame = draw_overlay(frame, waiting, crossing, sig_state, sig_color,
                             remaining, status_text, fps, "MOCK",
                             accident_active)

        cv2.imshow("SafeTurn AI — Mock Demo", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            f = f"safeturn_mock_{int(time.time())}.png"
            cv2.imwrite(f, frame)
            print(f"[SAVED] {f}")
        elif key == ord('a') or key == ord('A'):
            if not accident_active:
                accident_active = True
                accident_start = time.time()
                print(f"[ALERT] Accident simulated! RED for {ACCIDENT_DURATION:.0f}s")

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# SINGLE CAMERA MODE
# ═══════════════════════════════════════════════════════════════════

def run_single(args, detector):
    """One camera — detects pedestrians and vehicles with zone logic."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source = args.video
    else:
        cap = cv2.VideoCapture(args.camera if args.camera is not None else 0)
        source = f"Webcam {args.camera or 0}"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)
    print(f"[INFO] Source: {source}")

    signal      = SignalController()
    frame_count = 0
    fps         = 30.0

    last_persons  = []
    last_vehicles = []

    accident_active = False
    accident_start  = 0.0

    print()
    print("=" * 55)
    print("  SafeTurn AI — SINGLE CAMERA MODE (Zone-Based)")
    print("  🟢 GREEN → 🟡 ORANGE → 🔴 RED")
    print("  Q = Quit | S = Screenshot | R = Rewind | A = Accident")
    print("=" * 55)
    print()

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Detection (with frame skipping) ───────────────────────
        if frame_count % PROCESS_EVERY == 1:
            last_persons, last_vehicles = detector.detect(frame)

        # ── Zone classification ───────────────────────────────────
        waiting, crossing, wait_p, cross_p = classify_pedestrians(
            last_persons, w, h)

        # Use stable count for signal
        stable_total = detector.get_stable_ped_count()
        # But zone ratios from raw detection
        raw_total = waiting + crossing
        if raw_total > 0 and stable_total > 0:
            # Scale zone counts proportionally to stable total
            ratio_w = waiting / raw_total
            ratio_c = crossing / raw_total
            waiting = max(0, round(stable_total * ratio_w))
            crossing = max(0, round(stable_total * ratio_c))

        # ── Accident override ─────────────────────────────────────
        if accident_active:
            elapsed_acc = time.time() - accident_start
            if elapsed_acc >= ACCIDENT_DURATION:
                accident_active = False
            else:
                crossing = max(crossing, 1)

        # ── Update signal ─────────────────────────────────────────
        sig_state, sig_color, remaining, status_text = signal.update(waiting, crossing)

        if accident_active:
            sig_state = "RED"
            sig_color = COL_RED
            remaining = ACCIDENT_DURATION - (time.time() - accident_start)
            status_text = "ACCIDENT — ALL STOP"

        # ── Smoothed FPS ──────────────────────────────────────────
        now = time.time()
        dt = max(now - frame_start, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # ── Draw ──────────────────────────────────────────────────
        draw_zones(frame)
        draw_detections(frame, last_persons, last_vehicles, wait_p, cross_p)
        frame = draw_overlay(frame, waiting, crossing, sig_state, sig_color,
                             remaining, status_text, fps, "SINGLE CAM",
                             accident_active)

        cv2.imshow("SafeTurn AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            f = f"safeturn_{int(time.time())}.png"
            cv2.imwrite(f, frame)
            print(f"[SAVED] {f}")
        elif key == ord('r') or key == ord('R'):
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("[INFO] Rewound.")
        elif key == ord('a') or key == ord('A'):
            if not accident_active:
                accident_active = True
                accident_start = time.time()
                print(f"[ALERT] Accident simulated! RED for {ACCIDENT_DURATION:.0f}s")

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# DUAL CAMERA MODE
# ═══════════════════════════════════════════════════════════════════

def run_dual(args, detector):
    """Two cameras with simplified count-based detection."""
    cam_idx = args.camera if args.camera is not None else 0
    ped_cap = cv2.VideoCapture(cam_idx)
    if not ped_cap.isOpened():
        print(f"[ERROR] Cannot open webcam {cam_idx}")
        sys.exit(1)
    print(f"[INFO] Pedestrian camera: Webcam {cam_idx}")

    if not args.video:
        print("[ERROR] Dual mode requires --video <path>")
        sys.exit(1)
    traf_cap = cv2.VideoCapture(args.video)
    if not traf_cap.isOpened():
        print(f"[ERROR] Cannot open traffic video: {args.video}")
        sys.exit(1)
    print(f"[INFO] Traffic video: {args.video}")

    signal      = SimpleSignalController()
    frame_count = 0
    fps         = 30.0
    last_persons = []
    last_vehicles = []
    
    accident_active = False
    accident_start  = 0.0

    print()
    print("=" * 55)
    print("  SafeTurn AI — DUAL CAMERA MODE (Count-Based)")
    print("  🟢 GREEN → 🟡 ORANGE → 🔴 RED")
    print("  Q = Quit | S = Screenshot | A = Accident")
    print("=" * 55)
    print()

    while True:
        frame_start = time.time()

        ret1, ped_frame = ped_cap.read()
        if not ret1:
            ped_frame = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)

        ret2, traf_frame = traf_cap.read()
        if not ret2:
            traf_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, traf_frame = traf_cap.read()
            if not ret2:
                break

        frame_count += 1
        ph, pw = ped_frame.shape[:2]

        if frame_count % PROCESS_EVERY == 1:
            if ret1:
                last_persons = detector.detect_persons_only(ped_frame)
            if ret2:
                # Bypass StableDetector's history, use raw Detector for vehicles
                _, last_vehicles = detector.detector.detect(traf_frame)

        # Basic signal logic with stable pedestrian count (no zones)
        ped_count = detector.get_stable_ped_count()

        # ── Accident override ─────────────────────────────────────
        if accident_active:
            elapsed_acc = time.time() - accident_start
            if elapsed_acc >= ACCIDENT_DURATION:
                accident_active = False
            else:
                ped_count = max(ped_count, 2)  # Force RED logic

        sig_state, sig_color, remaining, status_text = signal.update(ped_count)
        
        if accident_active:
            sig_state = "RED"
            sig_color = COL_RED
            remaining = ACCIDENT_DURATION - (time.time() - accident_start)
            status_text = "ACCIDENT — ALL STOP"

        now = time.time()
        dt = max(now - frame_start, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Draw bounding boxes only (zones removed entirely)
        draw_detections(ped_frame, last_persons, None, None, None)

        # Ped cam overlay
        ph2, pw2 = ped_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        banner_h = 55
        ov = ped_frame.copy()
        banner_c = COL_GREEN if ped_count == 0 else COL_ORANGE if ped_count == 1 else COL_RED
        if accident_active: banner_c = COL_RED
        cv2.rectangle(ov, (0, 0), (pw2, banner_h), banner_c, -1)
        cv2.addWeighted(ov, 0.6, ped_frame, 0.4, 0, ped_frame)
        cv2.putText(ped_frame, f"Pedestrian Count: {ped_count}",
                    (12, 38), font, 0.8, COL_WHITE, 2)
        cv2.putText(ped_frame, f"Signal: {sig_state} | {status_text}", (12, banner_h + 25),
                    font, 0.6, sig_color, 2)
        
        if accident_active:
            cv2.putText(ped_frame, "ACCIDENT OVERRIDE", (pw2 // 2 - 120, ph2 - 40), font, 0.8, COL_RED, 2)

        # Traffic Camera Visualization
        draw_detections(traf_frame, [], last_vehicles, [], [])
        traf_frame = draw_traffic_overlay(traf_frame, sig_state,
                                   sig_color, remaining,
                                   fps, "DUAL CAM", accident_active)

        cv2.imshow("Pedestrian Camera", ped_frame)
        cv2.imshow("Traffic View", traf_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            ts = int(time.time())
            cv2.imwrite(f"ped_cam_{ts}.png", ped_frame)
            cv2.imwrite(f"traffic_{ts}.png", traf_frame)
            print(f"[SAVED] ped_cam_{ts}.png, traffic_{ts}.png")
        elif key == ord('a') or key == ord('A'):
            if not accident_active:
                accident_active = True
                accident_start = time.time()
                print(f"[ALERT] Accident simulated! RED for {ACCIDENT_DURATION:.0f}s")

    ped_cap.release()
    traf_cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI — Demo-Ready Traffic Signal Controller"
    )
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "dual"],
                        help="single = one camera | dual = webcam + video")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file path")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model path")
    parser.add_argument("--mock", action="store_true",
                        help="Run with synthetic mock detections")
    args = parser.parse_args()

    if args.mock:
        run_mock(args)
    else:
        from detection import StableDetector
        detector = StableDetector(args.model)
        if args.mode == "single":
            run_single(args, detector)
        else:
            run_dual(args, detector)

    print()
    print("=" * 55)
    print("  SafeTurn AI — Session Complete")
    print("  'Predictive, not reactive — we act before danger arrives.'")
    print("=" * 55)


if __name__ == "__main__":
    main()
