from collections import Counter, defaultdict
from pathlib import Path
import csv
import math

import cv2
import pandas as pd


PERSON_CLASS = "person"
VEHICLE_CLASSES = {"bicycle", "motorcycle", "car", "bus", "truck", "train"}

# COCO-style class names are supplied by the model.
def object_group(class_name: str) -> str:
    if class_name == PERSON_CLASS:
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


class TrackAnalytics:
    """Collect frame-level and track-level statistics for a video."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.previous_centroids = {}
        self.counted_line_ids = set()
        self.unique_track_ids = set()
        self.unique_by_class = defaultdict(set)
        self.line_crossings = Counter()
        self.frame_rows = []
        self.track_events = []

    def update(self, frame_no, timestamp, boxes, track_ids, class_names, confidences):
        current_counts = Counter()
        current_persons = 0
        current_vehicles = 0
        active_ids = set()
        frame_crossings = Counter()

        for box, track_id, cls_name, conf in zip(
            boxes, track_ids, class_names, confidences
        ):
            if track_id is None:
                continue

            track_id = int(track_id)
            active_ids.add(track_id)
            self.unique_track_ids.add(track_id)
            self.unique_by_class[cls_name].add(track_id)
            current_counts[cls_name] += 1

            if object_group(cls_name) == "person":
                current_persons += 1
            elif object_group(cls_name) == "vehicle":
                current_vehicles += 1

            x1, y1, x2, y2 = [int(v) for v in box]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            previous = self.previous_centroids.get(track_id)
            if previous is not None and track_id not in self.counted_line_ids:
                prev_y = previous[1]
                crossed = (prev_y < self.line_y <= cy) or (
                    prev_y > self.line_y >= cy
                )
                if crossed:
                    direction = "IN" if cy > prev_y else "OUT"
                    self.counted_line_ids.add(track_id)
                    self.line_crossings[cls_name] += 1
                    frame_crossings[cls_name] += 1
                    self.track_events.append(
                        {
                            "frame": frame_no,
                            "timestamp_sec": round(timestamp, 3),
                            "track_id": track_id,
                            "class": cls_name,
                            "direction": direction,
                            "confidence": round(float(conf), 4),
                            "centroid_x": cx,
                            "centroid_y": cy,
                        }
                    )

            self.previous_centroids[track_id] = (cx, cy)

        row = {
            "frame": frame_no,
            "timestamp_sec": round(timestamp, 3),
            "active_tracks": len(active_ids),
            "persons_visible": current_persons,
            "vehicles_visible": current_vehicles,
            "total_objects_visible": sum(current_counts.values()),
            "average_confidence": round(
                sum(float(c) for c in confidences) / len(confidences), 4
                if confidences else 0.0
            ),
            "line_crossings_this_frame": sum(frame_crossings.values()),
        }

        for cls in sorted(VEHICLE_CLASSES | {PERSON_CLASS}):
            row[f"{cls}_visible"] = current_counts.get(cls, 0)
            row[f"{cls}_crossings"] = frame_crossings.get(cls, 0)

        self.frame_rows.append(row)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_csv = output_dir / "frame_analytics.csv"
        event_csv = output_dir / "track_events.csv"
        summary_csv = output_dir / "summary.csv"

        pd.DataFrame(self.frame_rows).to_csv(frame_csv, index=False)
        pd.DataFrame(
            self.track_events,
            columns=[
                "frame", "timestamp_sec", "track_id", "class",
                "direction", "confidence", "centroid_x", "centroid_y"
            ],
        ).to_csv(event_csv, index=False)

        total_person_tracks = len(self.unique_by_class.get("person", set()))
        total_vehicle_tracks = sum(
            len(self.unique_by_class.get(cls, set()))
            for cls in VEHICLE_CLASSES
        )

        summary = {
            "total_unique_tracks": len(self.unique_track_ids),
            "unique_person_tracks": total_person_tracks,
            "unique_vehicle_tracks": total_vehicle_tracks,
            "total_line_crossings": sum(self.line_crossings.values()),
            "person_line_crossings": self.line_crossings.get("person", 0),
            "vehicle_line_crossings": sum(
                self.line_crossings.get(cls, 0) for cls in VEHICLE_CLASSES
            ),
        }

        for cls in sorted(VEHICLE_CLASSES | {PERSON_CLASS}):
            summary[f"unique_{cls}_tracks"] = len(
                self.unique_by_class.get(cls, set())
            )
            summary[f"{cls}_line_crossings"] = self.line_crossings.get(cls, 0)

        pd.DataFrame([summary]).to_csv(summary_csv, index=False)

        return frame_csv, event_csv, summary_csv, summary


def draw_dashboard(frame, analytics: TrackAnalytics, frame_no: int):
    h, w = frame.shape[:2]
    cv2.line(
        frame,
        (0, analytics.line_y),
        (w, analytics.line_y),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, max(30, analytics.line_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (330, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.65, frame, 0.35, 0)

    text = [
        f"Frame: {frame_no}",
        f"Unique tracks: {len(analytics.unique_track_ids)}",
        f"Persons crossed: {analytics.line_crossings.get('person', 0)}",
        f"Vehicles crossed: {sum(analytics.line_crossings.get(c, 0) for c in VEHICLE_CLASSES)}",
    ]
    for i, line in enumerate(text):
        cv2.putText(
            frame,
            line,
            (20, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
    return frame
