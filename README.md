# Human Activity Detection for Surveillance Video Compression

**Mini Project BCS554 / V Semester**  
**Department of Computer Science & Engineering, ABES Engineering College, Ghaziabad**  
**Supervisor:** Dr. Disha Mohini Pathak (Assistant Professor)  
**Team Members:**
1. Rajshree Bhuyan – 2400320100886
2. Rimjhim Madan – 2400320100898
3. Riya Maurya – 2400320100921
4. Rishabh Kumar – 2400320100902

---

## 📌 Abstract & Overview

Surveillance systems generate large volumes of continuous high-resolution video recordings, requiring significant digital storage and bandwidth. Existing systems compress videos with fixed parameters, either wasting storage on empty scenes or degrading video quality during critical events.

This project implements an intelligent, activity-aware surveillance compression pipeline:
$$\text{Detect Activity} \longrightarrow \text{Assess Importance} \longrightarrow \text{Adaptive Compression}$$

1. **Human Activity & Landmark Detection**: Leverages **YOLOv8 Pose Estimation** to detect human presence and track 17 body skeleton keypoints. Classifies postures and activities into **Sitting**, **Standing**, **Walking**, and **Fall/Distress**.
2. **Critical Event & Object Detection**: Utilizes **YOLOv8 Object Detection** to identify critical items (handheld devices, sharp tools, suspicious items) and abnormal human movements.
3. **Importance Assessment**:
   - **Emergency Mode (Critical Alert)**: Full frame rate (30 FPS) and high visual quality (CRF 24) to preserve fine investigative details.
   - **Normal Human Mode**: Standard frame rate (10-15 FPS) and balanced quality (CRF 32) for routine human actions (Sitting, Standing, Walking).
   - **Idle Mode**: Low frame rate (5 FPS) and high compression (CRF 36) for unoccupied, static scenes.
4. **Adaptive Compression Engine**: Employs FFmpeg with frame-accurate multi-segment encoding, variable bitrates/CRF, and synchronized AAC audio preservation.
5. **Quality & Storage Evaluation**: Measures **Peak Signal-to-Noise Ratio (PSNR)**, **Structural Similarity Index (SSIM)**, and **Storage Reduction %** against uncompressed sources and fixed uniform baselines.
6. **Real-Time Surveillance Monitoring UI**: A web dashboard (FastAPI + React/Vite) providing video upload, real-time annotated playback (yellow bounding boxes, green skeleton landmarks, red hazard alerts, HUD status banners), side-by-side comparison, and direct video export.

---

## 🏗️ System Architecture

```
                                  [Surveillance Video Input]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [YOLOv8 Human Pose Estimation]                     [YOLOv8 Object Detection]
       (17 Skeletal Body Keypoints)                       (Handheld / Critical Items)
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                             [Activity Importance Assessment]
                                              │
            ┌─────────────────────────────────┼────────────────────────────────┐
            ▼                                 ▼                                ▼
     [EMERGENCY MODE]                [NORMAL HUMAN MODE]                  [IDLE MODE]
   - Critical Alert / Hazard       - Sitting / Standing / Walking      - Unoccupied Scene
   - 30 FPS, CRF 24                - 10-15 FPS, CRF 32                 - 5 FPS, CRF 36
            │                                 │                                │
            └─────────────────────────────────┼────────────────────────────────┘
                                              ▼
                               [FFmpeg Adaptive Compressor]
                        (Multi-Tier Segment Encoding & Concat)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
         [Storage & Quality Eval]                            [Surveillance Web UI]
         - PSNR (dB) & SSIM                                  - Live HUD Monitor
         - Storage Reduction %                               - Side-by-Side Player
         - 3-Way Benchmark                                   - MP4 Download
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.12)
- **Node.js 18+** (tested on Node v22)
- **FFmpeg & FFprobe** installed and added to system `PATH`

### 2. Install Python Dependencies
```powershell
pip install -r requirements.txt
```
*(Or manually: `pip install fastapi uvicorn opencv-python ultralytics torch torchvision scikit-image numpy python-multipart`)*

### 3. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### 4. Run the Application

#### Option A: Interactive Web Application (Recommended)
1. **Start the FastAPI Backend** (runs on port 8000):
   ```powershell
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the React Frontend** (in a separate PowerShell terminal):
   ```powershell
   cd frontend
   npm run dev
   ```
   Open your browser at `http://localhost:5173`.

#### Option B: Standalone CLI Pipeline
To run the complete pipeline on any video from the command line:
```powershell
python main.py input/test_video.mp4 output/adaptive_compressed.mp4
```

---

## 📊 Experimental Results & Benchmarks

Benchmarked on authentic CCTV surveillance footage (1920x1080 @ 29.97 FPS, 2,670 frames, 89.09s):

| Metric | Original Surveillance Source | Fixed Uniform Baseline (CRF 32) | Activity-Aware Adaptive (Proposed) |
| :--- | :---: | :---: | :---: |
| **Compression Strategy** | Camera Raw / Uncompressed | Static CRF 32 | Dynamic Multi-Tier (CRF 24 / 32 / 36) |
| **Framerate Allocation** | Fixed 29.97 FPS | Fixed 29.97 FPS | Dynamic: 30 FPS / 10-15 FPS / 5 FPS |
| **File Size** | 60.45 MB | 30.07 MB | **37.21 MB** |
| **Storage Reduction** | 0.0% | 50.25% | **38.45% – 52.8%** |
| **Average PSNR** | $\infty$ | 33.41 dB | **34.82 dB** |
| **Average SSIM** | 1.0000 | 0.9247 | **0.9385** |
| **Resolution Integrity** | 1920x1080 | 1920x1080 | **1920x1080 (Preserved)** |
| **Audio Track** | AAC Stereo | AAC Stereo | **Bit-exact copy (Synchronized)** |

---

## 📁 Repository Directory Structure

```
ActivityAwareCompression/
├── backend/
│   └── main.py                     # FastAPI REST API server
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Surveillance dashboard & monitor UI
│   │   ├── App.css                 # Custom styles & HUD layouts
│   │   └── main.jsx                # React root
│   ├── package.json
│   └── vite.config.js
├── input/
│   ├── test_video.mp4              # Surveillance demo recording (89s)
│   └── test_video_89s.mp4
├── output/
│   ├── adaptive_compressed.mp4     # Generated activity-compressed video
│   ├── annotated_surveillance.mp4  # AI annotated video stream with overlays
│   └── uniform_compressed.mp4      # Fixed baseline comparison
├── src/
│   ├── activity_detector.py        # YOLOv8 Pose & Critical Alert Detection
│   ├── adaptive_compressor.py      # Multi-tier FFmpeg adaptive compressor
│   ├── evaluator.py                # Objective PSNR & SSIM evaluator
│   ├── motion_detector.py          # Scene-level differencing module
│   ├── quality_metrics.py          # Frame-by-frame metric utility
│   └── video_info.py               # Video inspection utility
├── main.py                         # CLI end-to-end execution script
├── requirements.txt                # Python dependencies
└── README.md
```
