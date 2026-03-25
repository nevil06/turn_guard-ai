# 🛡️ SafeTurn AI

**Intelligent Free-Left Turn Traffic Signal Controller**

> *"Predictive, not reactive — we act before danger arrives."*

Built for **HackSETU 2025** | Theme 3: Intelligent Free-Left Turn Management for Safer Roads

---

## 💡 What It Does

SafeTurn AI uses a camera feed with **YOLOv8 real-time object detection** to count pedestrians at a junction and **dynamically control traffic signals** for free-left turns. The system implements a realistic 3-stage signal (🟢 GREEN → 🟡 ORANGE → 🔴 RED) with stability logic to prevent flickering.

```
    ┌──────────────────────────────────────────────┐
    │           SafeTurn AI Pipeline                │
    │                                               │
    │  Camera Feed (video or webcam)                │
    │     │                                         │
    │     ▼                                         │
    │  YOLOv8n Detection (person, car, bike, bus)   │
    │     │                                         │
    │     ▼                                         │
    │  Pedestrian Count                             │
    │     │                                         │
    │     ▼                                         │
    │  3-Stage Signal Controller                    │
    │     │                                         │
    │     ▼                                         │
    │  Decision: 🟢 GREEN │ 🟡 ORANGE │ 🔴 RED     │
    └──────────────────────────────────────────────┘
```

---

## 🚦 Signal Logic

| Pedestrians | Action |
|-------------|--------|
| **0** | 🟢 Stay GREEN — free turn |
| **1** | Wait 5 seconds, then 🟡 ORANGE → 🔴 RED |
| **2+** | Immediately 🟡 ORANGE → 🔴 RED |

### Transition Rules

- GREEN → ORANGE → RED *(never skips a stage)*
- 🟡 ORANGE duration = **2 seconds** (warning)
- 🔴 RED minimum = **3 seconds**
- RED → GREEN only when **0 pedestrians** detected

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

```bash
# Single camera — one video with pedestrians + vehicles
python app.py --mode single --video test4vedio.mp4

# Single camera — webcam
python app.py --mode single --camera 0

# Dual camera — webcam (pedestrians) + video (traffic)
python app.py --mode dual --video test4vedio.mp4
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `R` | Rewind video (single mode) |
| `A` | Simulate accident |

---

## 🏗️ Architecture (Modular)

```
app.py                    ← Main entry point (--mode single / dual)
├── detection.py          ← YOLOv8 wrapper with frame resize optimization
├── signal_controller.py  ← 3-stage state machine (GREEN → ORANGE → RED)
│
├── Single Mode           ← One camera: detects persons + vehicles
└── Dual Mode             ← Webcam (pedestrians) + Video (traffic view)
```

### Other Files

```
safeturn_final.py         ← Legacy complete system (predictive engine)
safeturn_main.py          ← Legacy single-camera with zone detection
step2_dual_camera.py      ← Standalone single-camera version
dashboard.py              ← Streamlit monitoring dashboard
```

---

## ⚡ Performance Optimizations

| Optimization | Details |
|-------------|---------|
| **Frame skipping** | YOLO runs every 2nd frame, cached results reused |
| **Frame resize** | Input resized to 640×360 before inference |
| **YOLO imgsz** | Internal YOLO size set to 480px (vs default 640) |
| **YOLOv8n** | Nano model — fastest, optimized for CPU |
| **cv2.waitKey(1)** | Zero display delay for smooth playback |

---

## 🎯 Detection & Visualization

**Detected classes** with color-coded bounding boxes:

| Class | Color | COCO ID |
|-------|-------|---------|
| person | 🟢 Green | 0 |
| car | 🔴 Red | 2 |
| bicycle / motorbike | 🟠 Orange | 1, 3 |
| bus | 🩷 Pink | 5 |
| truck | 🫧 Teal | 7 |

Each detection shows: **class name + confidence** (e.g., `person 85%`)

---

## 🚨 Accident Simulation

Press **A** during the demo to trigger:

| Feature | What Happens |
|---------|-------------|
| 🔴 Signal Override | Immediate RED for 5 seconds |
| 🚑 Ambulance Alert | "Ambulance Alert Triggered" displayed |
| ⚠️ Visual | Flashing red border + center banner |

---

## 🛠️ Tech Stack

- **Python** — core language
- **OpenCV** — video processing + UI overlays
- **YOLOv8** (ultralytics) — real-time object detection
- **NumPy** — array operations

---

## 👥 Team

Built at **HackSETU 2025**
