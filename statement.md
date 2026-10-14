# Statement

**Name:** [YOUR NAME]
**Roll No. / ID:** [YOUR ID]
**Project:** VisionGuard – Object Detection, Tracking and Counting
**Repository:** https://github.com/Ari09-j/VisionGuard

## 1. Originality declaration
I declare that this project is my own work, except for the third-party libraries and pretrained models listed in Section 3.

## 2. What I implemented myself
- **Line-crossing analytics (`src/tracker.py`, `src/line_counter.py`)**
  - Centroid-based virtual line with direction detection (IN/OUT).
  - Hysteresis margin around the line and a minimum-frames rule, so jitter near the line does not cause double counts.
  - Per-track majority-vote class labelling to avoid class flicker.
  - My own ID-stitching step that merges a lost track with a new track of the same class that appears nearby within a short time window.
- **Unique-ID counting** per class (person / vehicle categories).
- **Analytics export:** `frame_analytics.csv`, `track_events.csv`, `summary.csv`, `advanced_report.json`.
- **Evaluation (`src/evaluate.py`)** comparing system counts against manually counted ground truth (absolute error, percentage error, MAE).
- **CLI pipeline (`src/main.py`)** connecting detection, tracking, analytics and video output.

## 3. Third-party components
| Component | Used for |
|---|---|
| Ultralytics YOLO (pretrained weights) | Object detection |
| Ultralytics ByteTrack / BoT-SORT | Frame-to-frame association (track IDs) |
| OpenCV | Video reading/writing and drawing |
| NumPy / pandas | Numerical work and CSV handling |

Detection and base tracking come from Ultralytics. My contribution is the analytics, counting logic, ID stitching, export and evaluation built on top of them.

## 4. Use of AI tools
[WRITE THIS HONESTLY. Example: "I used an AI assistant to help review my code, suggest improvements to the line-crossing logic, and draft documentation. I tested and verified the final code myself."]

## 5. Data
Test video: [NAME / SOURCE / LICENCE OF YOUR VIDEO]. Ground-truth counts were obtained by manually counting crossings in the clip.

## 6. Known limitations
- ID switches can still occur under heavy occlusion.
- Accuracy depends on video quality, camera angle and placement of the counting line.
- Only the classes known to the pretrained model can be detected.
- No real-world speed or distance estimation.

## 7. How to run
```
pip install -r requirements.txt
python src/main.py --source data/traffic.mp4
python src/evaluate.py --events outputs/result/track_events.csv --truth data/ground_truth.csv
```
