# VISIONGUARD
## Advanced Object Detection, Multi-Object Tracking and Traffic Analytics

**Course Domain:** Computer Vision  
**Project Type:** Academic / Flipped Course Evaluation  
**Technology:** Python, OpenCV, YOLO, ByteTrack/BoT-SORT/OC-SORT, Pandas  
**Interface:** Command Line

---

## Abstract

Computer vision enables machines to extract meaningful information from images and videos. One important application is intelligent video analytics, where a system must not only detect objects but also understand their movement over time.

VisionGuard is an advanced computer vision project developed to demonstrate this workflow. The system processes video using a YOLO object-detection model and a multi-object tracking algorithm. It assigns persistent IDs to detected objects, distinguishes people from common vehicle categories, records object movement, detects virtual line crossings, and exports frame-level and event-level analytics as CSV and JSON files.

The project is designed as a reproducible command-line application. It produces an annotated output video containing bounding boxes, object labels, track IDs, and a live analytics dashboard. This makes the project suitable for applications such as traffic monitoring, campus surveillance, crowd analysis, and intelligent transportation systems.

---

# 1. Introduction

Traditional video observation requires a human operator to continuously watch video and manually count or monitor objects. This approach becomes difficult when the amount of video increases.

Computer vision can automate this process. Object detection identifies what is visible in a frame, while multi-object tracking associates detections across consecutive frames.

VisionGuard combines both capabilities and adds a data-analysis layer.

---

# 2. Problem Statement

Develop a computer vision system capable of:

1. Detecting objects in video.
2. Assigning persistent IDs to detected objects.
3. Tracking objects across frames.
4. Counting unique people and vehicles.
5. Detecting movement across a virtual counting line.
6. Recording timestamps and track IDs.
7. Generating machine-readable CSV/JSON analytics.
8. Producing an annotated video for visual verification.
9. Running completely from a command-line environment.

---

# 3. Objectives

### Primary objectives

- Implement YOLO-based object detection.
- Implement multi-object tracking.
- Maintain persistent object identities.
- Separate people and vehicles.
- Implement virtual line-crossing logic.
- Generate analytical reports.

### Secondary objectives

- Make the application reproducible.
- Keep the code modular.
- Provide configurable confidence and tracker settings.
- Export results for further statistical analysis.

---

# 4. Scope

The current system focuses on video analytics using pretrained object-detection classes.

Potential application areas include:

- Traffic monitoring
- Campus security
- Crowd monitoring
- Smart-city systems
- Retail analytics
- Transportation analytics

The current version is an analytics prototype rather than a production surveillance system.

---

# 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video I/O and visualization |
| Ultralytics YOLO | Object detection and tracking interface |
| ByteTrack | Multi-object tracking |
| BoT-SORT | Alternative tracking algorithm |
| OC-SORT | Alternative tracking algorithm |
| Pandas | CSV analytics |
| NumPy | Numerical operations |
| JSON | Structured reporting |

Ultralytics documents YOLO tracking as a mode that maintains object IDs across video frames and supports multiple trackers including ByteTrack and BoT-SORT. citeturn0search5

---

# 6. System Architecture

```text
Video Input
    |
    v
Frame Extraction
    |
    v
YOLO Object Detection
    |
    v
Multi-Object Tracking
    |
    +--------------------+
    |                    |
    v                    v
Object Class          Track ID
    |                    |
    +---------+----------+
              |
              v
        Centroid Calculation
              |
              v
       Virtual Line Crossing
          /           \
        IN             OUT
          \           /
           v         v
       Event Analytics
              |
       +------+------+
       |             |
       v             v
      CSV           JSON
       |
       v
Annotated Video
```

---

# 7. Methodology

## 7.1 Video Acquisition

The application receives a video path from the command line.

Example:

```bash
python src/main.py --source data/traffic.mp4
```

The video is divided into individual frames using OpenCV.

## 7.2 Object Detection

The YOLO model processes each frame and returns:

- bounding boxes,
- object classes,
- confidence scores.

## 7.3 Multi-Object Tracking

The tracker associates detections between frames and provides persistent IDs.

For example:

```text
Frame 101:
person ID 7

Frame 102:
person ID 7

Frame 103:
person ID 7
```

Although the person's location changes, the system understands that these detections correspond to the same tracked object.

## 7.4 Centroid Calculation

For a bounding box:

```text
(x1, y1, x2, y2)
```

the centroid is:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

The centroid is used to determine movement relative to the virtual counting line.

## 7.5 Line-Crossing Detection

Suppose the counting line is at:

```text
y = 400
```

If:

```text
previous_y < 400
current_y >= 400
```

the object is classified as moving **IN**.

If:

```text
previous_y > 400
current_y <= 400
```

the object is classified as moving **OUT**.

The track ID is stored so the same object is not counted repeatedly.

## 7.6 Person Counting

The COCO class `person` is treated separately.

The system records:

- unique person IDs,
- visible people per frame,
- line crossings by people.

## 7.7 Vehicle Counting

The system treats the following as vehicle-related classes:

- bicycle
- motorcycle
- car
- bus
- truck
- train

The system records unique vehicle tracks and vehicle line crossings.

## 7.8 Analytics

Frame-level information is written to:

```text
frame_analytics.csv
```

Individual crossing events are written to:

```text
track_events.csv
```

Overall experiment statistics are written to:

```text
summary.csv
```

---

# 8. Algorithms

## Algorithm 1: Object Detection and Tracking

```text
Input video
Load YOLO model
Initialize tracker
Read next frame

For every frame:
    Detect objects
    Extract boxes
    Extract class labels
    Extract confidence
    Extract track IDs

    Calculate object centroids
    Compare current and previous centroid

    If virtual line is crossed:
        Record track ID
        Record class
        Record direction
        Record timestamp

    Draw bounding boxes
    Draw track IDs
    Draw counting line
    Write output frame

Repeat until video ends

Save analytics
```

---

# 9. Why Tracking Is Needed

A naive approach could count detections:

```text
Frame 1: 3 persons
Frame 2: 3 persons
Frame 3: 3 persons
```

This would produce:

```text
3 + 3 + 3 = 9
```

even though only three people exist.

Tracking solves this by assigning IDs:

```text
Frame 1: ID 1, ID 2, ID 3
Frame 2: ID 1, ID 2, ID 3
Frame 3: ID 1, ID 2, ID 3
```

Therefore the unique count remains:

```text
3
```

---

# 10. Output Files

## tracked_output.mp4

Annotated video containing:

- bounding boxes,
- class labels,
- track IDs,
- counting line,
- analytics dashboard.

## frame_analytics.csv

Stores statistics for each frame.

## track_events.csv

Stores every detected line-crossing event.

## summary.csv

Stores final experiment-level statistics.

## advanced_report.json

Stores configuration and final results in structured form.

---

# 11. Sample Analytics

Example:

| Metric | Value |
|---|---:|
| Total unique tracks | 34 |
| Unique persons | 17 |
| Unique vehicles | 17 |
| Total line crossings | 29 |
| Person crossings | 12 |
| Vehicle crossings | 17 |

These are example values; actual results depend on the test video.

---

# 12. Advantages

1. Combines detection and tracking.
2. Avoids repeatedly counting the same object.
3. Provides persistent IDs.
4. Separates people and vehicles.
5. Produces structured analytics.
6. Supports multiple trackers.
7. Works from the command line.
8. Results can be analyzed later using spreadsheet or Python tools.
9. Modular design makes future extensions easier.

---

# 13. Limitations

1. Heavy occlusion can cause ID switches.
2. Poor lighting can reduce detection quality.
3. Very small objects can be difficult to detect.
4. Incorrect line placement can affect crossing counts.
5. CPU inference can be slow.
6. The pretrained model is limited to its known classes.
7. A line-crossing event is an application heuristic rather than a complete traffic-understanding system.

---

# 14. Future Scope

### Real-time CCTV

Connect the system directly to an RTSP camera stream.

### Speed Estimation

Track centroid displacement and calibrate pixels to real-world distance.

### Heatmaps

Generate spatial heatmaps showing where objects spend time.

### Traffic Density

Estimate traffic density from active vehicle tracks.

### Custom Dataset

Train the model on a domain-specific dataset.

### Web Dashboard

Display:

- live counts,
- charts,
- vehicle types,
- crossing rate,
- historical analytics.

### Edge Deployment

Deploy on hardware such as Jetson-class edge devices.

---

# 15. Testing Strategy

The project should be tested with:

### Test 1 — Empty/low-density video

Expected:

- few or zero detections,
- no false crossing events.

### Test 2 — Pedestrian video

Expected:

- person detections,
- unique person IDs,
- person crossings.

### Test 3 — Traffic video

Expected:

- cars/buses/trucks/motorcycles,
- vehicle track IDs,
- vehicle crossings.

### Test 4 — Different tracker

Run:

```bash
python src/main.py --source data/traffic.mp4 --tracker botsort.yaml
```

Compare the generated analytics.

---

# 16. Conclusion

VisionGuard demonstrates how modern computer vision can transform raw video into structured information. The project combines object detection, multi-object tracking, geometric line-crossing logic, and data analytics into one reproducible application.

The project therefore moves beyond basic image classification or object detection and demonstrates a practical video-understanding pipeline.

---

# 17. Viva Preparation

## Q1. What is computer vision?

Computer vision is a field of artificial intelligence that enables computers to extract useful information from images and videos.

## Q2. What is object detection?

Object detection identifies both the class of an object and its location in an image.

## Q3. How is detection different from tracking?

Detection identifies objects independently in each frame. Tracking associates those detections over time and assigns persistent IDs.

## Q4. Why do you need tracking for counting?

Without tracking, the same object could be counted once in every frame. Tracking lets us count unique objects.

## Q5. What is a bounding box?

A bounding box is a rectangle surrounding a detected object, usually represented by its top-left and bottom-right coordinates.

## Q6. What is a confidence score?

It represents the model's confidence that a detected region belongs to a particular class.

## Q7. What is YOLO?

YOLO stands for You Only Look Once. It is a family of deep-learning models designed for fast object detection and related computer-vision tasks.

## Q8. What is multi-object tracking?

It is the process of maintaining identities for multiple objects as they move through consecutive video frames.

## Q9. Which trackers does your project support?

The project supports ByteTrack, BoT-SORT and OC-SORT through the Ultralytics tracking interface.

## Q10. Why did you use ByteTrack?

ByteTrack provides a practical baseline for associating detections across frames and is straightforward to configure in the project.

## Q11. What is the purpose of the virtual line?

The virtual line provides a reference for determining whether a tracked object moved from one side of the scene to the other.

## Q12. How do you detect crossing?

The system compares the object's previous centroid with its current centroid. If the centroid changes sides relative to the counting line, a crossing is recorded.

## Q13. How do you prevent double counting?

A track ID is stored after it crosses the line. The same ID is not counted again for that line-crossing event.

## Q14. What is an ID switch?

An ID switch occurs when the tracker assigns a different identity to an object that was previously being tracked under another ID.

## Q15. What causes ID switches?

They can occur because of occlusion, similar-looking objects, rapid motion, crowded scenes, or weak detections.

## Q16. What is IoU?

Intersection over Union measures the overlap between two bounding boxes:

```text
IoU = Area of Intersection / Area of Union
```

It is commonly used when comparing predicted and reference bounding boxes and in detection/tracking association.

## Q17. What is occlusion?

Occlusion occurs when an object is partially or completely hidden by another object.

## Q18. Why can a person disappear from tracking?

The detector may fail to detect the person temporarily, for example because of occlusion, blur, poor lighting, or a low confidence score.

## Q19. Why did you use OpenCV?

OpenCV provides convenient tools for reading video frames, writing video, drawing boxes/lines/text, and basic image processing.

## Q20. Why did you use Pandas?

Pandas makes it easy to organize frame-level and event-level information and export it to CSV.

## Q21. Why generate CSV files?

CSV files make the analytics reusable. The results can later be analyzed using Excel, Pandas, Power BI, or other data-analysis tools.

## Q22. What happens if the confidence threshold is increased?

Fewer low-confidence detections are accepted. This can reduce false positives but may also remove genuine difficult detections.

## Q23. What happens if the confidence threshold is decreased?

More detections can be accepted, but the number of false positives may increase.

## Q24. Can your project detect every type of vehicle?

No. It can detect the classes supported by the pretrained model. A custom dataset and model would be required for domain-specific categories.

## Q25. How would you improve the project?

I would add custom model training, object tracking improvements, speed estimation, heatmaps, real-time camera input, a web dashboard, and deployment on edge hardware.

## Q26. Is your project AI or traditional computer vision?

It combines both. The YOLO component uses deep learning for object detection, while OpenCV and the line-crossing logic perform conventional image/video processing and geometric analysis.

## Q27. What is the main contribution of your project?

The main contribution is integrating object detection, persistent multi-object tracking, person/vehicle counting, line-crossing analysis, and structured data reporting into one reproducible command-line application.

## Q28. What would happen if there are two cars overlapping?

The detector may still identify them separately, but severe overlap can cause missed detections or ID switches. Tracker performance depends on detection quality and scene conditions.

## Q29. How could you measure project performance?

I could measure detection precision/recall or mAP for detection and tracking metrics such as IDF1, HOTA, MOTA, ID switches, and counting accuracy against manually annotated ground truth.

## Q30. What is the difference between detection accuracy and counting accuracy?

Detection accuracy measures how correctly objects are detected. Counting accuracy measures whether the final number of unique objects or crossing events matches the reference count.

---

# 18. Demonstration Script

During the evaluation, demonstrate:

```bash
python src/main.py --source data/traffic.mp4
```

Then explain:

> "The system first extracts frames from the video. YOLO detects objects and the tracker assigns persistent IDs. The centroid of each tracked object is monitored relative to the virtual counting line. When an object crosses the line, the event is stored with its ID, class, direction, timestamp and confidence. At the end, the system generates an annotated video along with CSV and JSON analytics."

Then show:

```text
outputs/result/
```

and open:

```text
summary.csv
track_events.csv
frame_analytics.csv
```

This demonstrates both the computer-vision component and the data-analysis component of the project.
