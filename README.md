# 🏀 Football Drill Object Tracker with Pose Estimation

This project uses **YOLOv8** for both sports ball detection and human pose estimation to track football drills from a video. It combines object tracking with movement state detection and pose keypoint visualization.

---

## 📋 Features
- **Sports ball tracking** using YOLOv8 and ByteTrack.
- **Pose estimation** for players using YOLOv8 pose model.
- **Movement state detection** (`ACTION` or `STATIONARY`) based on centroid displacement.
- **Annotated video output** with bounding boxes, pose skeletons, and movement labels.
- Real-time preview window with option to quit (`q` key).

---

## 📦 Requirements

### Python Version
- Python **3.10** (virtual environment recommended)

### Dependencies
    Install the required libraries:
     ```bash
     pip install ultralytics opencv-python numpy

### 2. Create and activate virtual environment (Python 3.10)
     ```bash
     python3.10 -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
### 3. Install dependencies
    ```bash
    pip install ultralytics opencv-python numpy

### 4. Add input video
- Place your video inside a videos/ directory and name it test.mp4
- Create an outputs/ directory for the generated annotated video.
  
### ▶️ Usage
      ```bash
      Run the script:
      python test.py

### 📁 Project Structure
.
├── test.py               # Main tracking + pose estimation script
├── videos/
│   └── test.mp4           # Input video file
├── outputs/
│   └── annotated_test_2.mp4 # Output annotated video
└── README.md

### 🚀 Future Improvements

-Add a real-time camera input option.
-Implement speed and distance calculation for players and ball.
-Integrate LLM or Agentic AI for tactical analysis of player movements.
-Save tracking logs for post-game analytics.
-Support multi-camera stitching for better coverage.
