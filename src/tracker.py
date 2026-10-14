from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from line_counter import IdStitcher, LineCounter


PERSON_CLASS = "person"
VEHICLE_CLASSES = {"bicycle", "motorcycle", "car", "bus", "truck", "train"}

EVENT_COLUMNS = [
    "frame", "timestamp_sec", "track_id", "class",
    "direction", "confidence", "centroid_x", "centroid_y",
]


# COCO-style class names are supplied by the model.
def object_group(class_name: str) -> str:
    if class_name == PERSON_CLASS:
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


class TrackAnalytics:
    """
    Collect frame-level and track-level statistics for a video.

    Own analytics layer on top of Ultralytics tracking:
      * IdStitcher   - merges fragmented track IDs (same class, nearby, short gap)
      * LineCounter  - hysteresis + minimum-frames line crossing, direction-aware,
                       majority-vote class per track
    """

    def __init__(self, line_y: int, margin: int = 8, min_frames: int = 3,
                 stitch: bool = True):
        self.line_y = line_y
        self.stitcher = IdStitcher() if stitch else None
        self.counter = LineCounter(line_y, margin=margin, min_frames=min_frames)
        self.unique_track_ids = set()
        self.line_crossings = Counter()
        self.frame_rows = []

    @property
    def track_events(self):
        return self.counter.events

    def _unique_by_voted_class(self):
        """Unique canonical IDs grouped by each track's majority-vote class."""
        grouped = defaultdict(set)
        for tid in self.unique_track_ids:
            votes = self.counter.class_votes.get(tid)
            if votes:
                grouped[votes.most_common(1)[0][0]].add(tid)
        return grouped

    def update(self, frame_no, timestamp, boxes, track_ids, class_names, confidences):
        current_counts = Counter()
        current_persons = 0
        current_vehicles = 0
        active_ids = set()
        frame_crossings = Counter()

        for box, raw_id, cls_name, conf in zip(
            boxes, track_ids, class_names, confidences
        ):
            if raw_id is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            raw_id = int(raw_id)
            if self.stitcher:
                track_id = self.stitcher.resolve(frame_no, raw_id, cls_name, cx, cy)
            else:
                track_id = raw_id

            active_ids.add(track_id)
            self.unique_track_ids.add(track_id)
            current_counts[cls_name] += 1

            group = object_group(cls_name)
            if group == "person":
                current_persons += 1
            elif group == "vehicle":
                current_vehicles += 1

            event = self.counter.update(
                frame_no, timestamp, track_id, cls_name, conf, cx, cy
            )
            if event is not None:
                self.line_crossings[event["class"]] += 1
                frame_crossings[event["class"]] += 1

        # FIX: the original expression put the `if confidences` check inside
        # round()'s second argument, so frames with no detections raised
        # ZeroDivisionError.
        avg_conf = (
            round(sum(float(c) for c in confidences) / len(confidences), 4)
            if confidences else 0.0
        )

        row = {
            "frame": frame_no,
            "timestamp_sec": round(timestamp, 3),
            "active_tracks": len(active_ids),
            "persons_visible": current_persons,
            "vehicles_visible": current_vehicles,
            "total_objects_visible": sum(current_counts.values()),
            "average_confidence": avg_conf,
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
        pd.DataFrame(self.track_events, columns=EVENT_COLUMNS).to_csv(
            event_csv, index=False
        )

        unique_by_class = self._unique_by_voted_class()
        total_person_tracks = len(unique_by_class.get("person", set()))
        total_vehicle_tracks = sum(
            len(unique_by_class.get(cls, set())) for cls in VEHICLE_CLASSES
        )
        merged_ids = (
            sum(1 for r, c in self.stitcher.raw_to_canon.items() if r != c)
            if self.stitcher else 0
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
            "ids_merged_by_stitcher": merged_ids,
        }

        for cls in sorted(VEHICLE_CLASSES | {PERSON_CLASS}):
            summary[f"unique_{cls}_tracks"] = len(unique_by_class.get(cls, set()))
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
