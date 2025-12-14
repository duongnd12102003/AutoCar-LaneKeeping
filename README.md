AutoCar-LaneKeeping

A Modular Real-Time Lane Keeping and Benchmarking Framework for Miniature Autonomous Vehicles

Team Members
    Phan Van Chuong
    Nguyen Dinh Duong
    Nguyen Van Truong 
Mentor
    AnhKD3 (Khuất Đức Anh)

1. Project Overview

This project implements a real-time lane keeping system for a miniature autonomous vehicle (AutoCar-Kit).
The system receives camera frames streamed over UDP from an ESP32-S3, performs lane segmentation using deep learning, estimates lane geometry in Bird’s-Eye View (BEV), and generates steering and speed commands for autonomous driving.

The project is developed as a graduation thesis, with emphasis on:

System-level design (perception → geometry → control)

Fair benchmarking of multiple lane segmentation models

Stability and robustness in real-world indoor environments

Reproducibility for future research and teaching

This is a real deployment system, not a simulation.

2. Main Contributions

Modular, model-agnostic lane-keeping pipeline

Robust UDP JPEG frame reassembly for ESP32 streaming

Unified post-processing and BEV projection for fair comparison

Multi-ratio lane geometry estimation with adaptive selection

EMA-based stabilization for control robustness

Real-time visualization and CSV benchmark logging

3. System Architecture

ESP32-S3 Camera
→ (JPEG over UDP)
→ Frame Receiver
→ Lane Segmentation Backend
→ Mask Processing (ROI + Morphology + CC)
→ Bird’s-Eye View (BEV)
→ Multi-Ratio Geometry Estimation
→ EMA Stabilization
→ Lane Controller
→ Steering & Speed Commands → ESP32

All models share the same downstream pipeline, ensuring fair and reproducible benchmarking.

4. Project Structure

AI/
├─ LaneDetection/
│ ├─ backends/
│ ├─ lane_pipeline.py
│ ├─ lane_controller.py
│ ├─ lane_geometry.py
│ └─ lane_overlay.py
│
├─ utils/
│ ├─ frame_receiver.py
│ ├─ udp.py
│ ├─ calib.py
│ └─ logger.py
│
├─ configs/
│ ├─ config.py
│
├─ logs/
└─ main.py

6. Installation

Python ≥ 3.9
Windows
Install dependencies:
pip install -r requirements.txt

Note:
PyTorch version depends on your CUDA configuration.
Install the appropriate build from https://pytorch.org
 if GPU acceleration is required.

7. Configuration

Edit configs/config.py:

LISTEN_IP = "YOUR_PC_IP"
LISTEN_PORT = 3000

ESP_IP = "ESP32_IP"
ESP_PORT = 3001

LANE_MODEL = "pidnet" # yolov8 | pidnet | twinlite | bisenet
SHOW = True

8. Dataset

The dataset used for training and evaluation is not included in this repository.

Dataset download link:
[(https://drive.google.com/drive/u/0/folders/1fL22grqBu_YjszkUBqKqs98S_Nv2VUEi)]


9. Pretrained Model Weights

Pretrained weights for each supported model are provided separately.

Model weights download link:
[(https://drive.google.com/drive/u/0/folders/1Xdl3OQaeNlNbwnEjTJZj4xluvQplL2np)]

Expected directory structure:

AI/LaneDetection/Lane_weight/
├─ Yolo_v8/best.pt
├─ PIDNet/best.pt
├─ TwinLite/best.pth
└─ BiseNet/best.pth

10. Running the System

Power on the AutoCar-Kit (ESP32-S3 camera streaming enabled)

Ensure PC and ESP32 are on the same local network

Start the system:

python main.py

Runtime controls:

q / ESC : Quit

p : Toggle multi-ratio visualization

11. Runtime Outputs

11.1 Visualization Overlay

Lateral offset (meters)

Heading angle (degrees)

Driving direction

FPS

Selected lane reference points

12. Acknowledgment

Developed as part of the Bachelor of Artificial Intelligence program at FPT University,
building upon the AutoCar-Kit hardware platform from previous student cohorts.
