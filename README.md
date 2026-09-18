# VisionGuard — Advanced Computer Vision Analytics

# Computer Vision Project

VisionGuard is an end-to-end computer vision analytics system that performs **object detection, multi-object tracking, person/vehicle counting, line-crossing analytics, frame-level statistics, CSV export, and JSON reporting**.

## Core Features

### 1. Object Detection

Uses a pretrained YOLO model to detect objects in video frames.

### 2. Multi-Object Tracking

Every detected object can receive a persistent track ID across frames. The project supports:

- ByteTrack
- BoT-SORT
- OC-SORT

Ultralytics documents these as supported multi-object trackers for YOLO tracking.

### 3. Person Counting

The system identifies the `person` class and calculates:

- unique people tracked,
- people visible per frame,
- people crossing the counting line.

### 4. Vehicle Counting

The system analyzes common vehicle categories:

- bicycle
- motorcycle
- car
- bus
- truck
- train

It reports unique tracked vehicles and line crossings.

### 5. Virtual Counting Line

A horizontal line is automatically placed at a configurable position.

When a tracked object's centroid crosses the line, the system records:

- track ID,
- class,
- direction,
- confidence,
- timestamp,
- centroid coordinates.

### 6. CSV Analytics

The program generates:

```text
frame_analytics.csv
track_events.csv
summary.csv
```

This makes the computer vision output suitable for later statistical analysis.

### 7. JSON Report

```text
advanced_report.json
```

contains the overall experiment configuration and final analytics.

### 8. Annotated Video

```text
tracked_output.mp4
```

contains bounding boxes, track IDs and an analytics dashboard.

---

# Installation

## Requirements

- Python 3.10 or newer
- 4 GB+ RAM recommended
- Windows/Linux/macOS
- Internet connection for the first model download

Create an environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Place a video inside:

```text
data/
```

For example:

```text
data/traffic.mp4
```

Run:

```bash
python src/main.py --source data/traffic.mp4
```

Use a different tracker:

```bash
python src/main.py --source data/traffic.mp4 --tracker botsort.yaml
```

Change confidence:

```bash
python src/main.py --source data/traffic.mp4 --confidence 0.40
```

Move the counting line:

```bash
python src/main.py --source data/traffic.mp4 --line-ratio 0.50
```

Combine options:

```bash
python src/main.py --source data/traffic.mp4 --tracker botsort.yaml --confidence 0.35 --line-ratio 0.55
```

---

# Output

After execution:

```text
outputs/result/
│
├── tracked_output.mp4
├── frame_analytics.csv
├── track_events.csv
├── summary.csv
└── advanced_report.json
```

## Example Summary

```text
total_unique_tracks: 34
unique_person_tracks: 17
unique_vehicle_tracks: 17
total_line_crossings: 29
person_line_crossings: 12
vehicle_line_crossings: 17
```

The actual numbers depend on the input video.

---

# CSV Files

## frame_analytics.csv

Contains one row per processed frame:

```text
frame
timestamp_sec
active_tracks
persons_visible
vehicles_visible
total_objects_visible
average_confidence
line_crossings_this_frame
person_visible
car_visible
bus_visible
truck_visible
...
```

## track_events.csv

Contains line-crossing events:

```text
frame
timestamp_sec
track_id
class
direction
confidence
centroid_x
centroid_y
```

## summary.csv

Contains experiment-level totals.

---

# Technical Pipeline

```text
                 INPUT VIDEO
                      |
                      v
             Frame Extraction
                      |
                      v
              YOLO Detection
                      |
                      v
             Multi-Object Tracker
             /        |        \
            /         |         \
       Track ID    Class      Confidence
            \         |         /
             \        |        /
                      v
              Centroid Tracking
                      |
                      v
              Virtual Line Test
                 /          \
              IN              OUT
               \              /
                \            /
                 v          v
                 CSV + JSON Analytics
                      |
                      v
              Annotated Video
```

## Tracking Concept

Detection answers:

> "What objects are present in this frame?"

Tracking adds:

> "Is this the same object I saw in the previous frame?"

Ultralytics describes tracking as maintaining unique IDs across video frames. citeturn0search5

## Counting Concept

The project does not simply count every detection. That would count the same person or car many times.

Instead, a persistent track ID is used. A line crossing is counted once for each track ID.

Ultralytics' own object-counting solution similarly performs counting based on object tracks and supports class-wise counts. citeturn0search0turn0search1

---

# Limitations

1. Tracking can experience ID switches during heavy occlusion.
2. A low-quality or very crowded video can reduce detection accuracy.
3. Counting accuracy depends on the location of the virtual line.
4. The system does not estimate real-world distance or speed.
5. Pretrained classes are limited to the categories known by the model.
6. CPU processing is slower than GPU processing.

# Future Scope

- Real-time CCTV integration
- Speed estimation
- Traffic-flow analysis
- Automatic congestion detection
- Region-based counting
- Heatmaps
- Custom dataset training
- Web dashboard
- Edge-device deployment
- Automatic alerts

# Academic Relevance

This project demonstrates practical understanding of:

- Computer Vision
- Deep Learning
- Object Detection
- Multi-Object Tracking
- Image/Video Processing
- Data Analytics
- CSV/JSON data generation
- Python programming
- Command-line application design

# References

Ultralytics YOLO Python API:
https://docs.ultralytics.com/usage/python/

Ultralytics Multi-Object Tracking:
https://docs.ultralytics.com/modes/track/

Ultralytics Object Counting:
https://docs.ultralytics.com/guides/object-counting/
