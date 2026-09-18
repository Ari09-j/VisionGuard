import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Dict, Any

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_image_source(source: str) -> bool:
    suffix = Path(source.split("?")[0]).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def is_video_source(source: str) -> bool:
    suffix = Path(source.split("?")[0]).suffix.lower()
    return suffix in VIDEO_EXTENSIONS


def density_label(max_objects: int) -> str:
    if max_objects <= 3:
        return "LOW"
    if max_objects <= 8:
        return "MEDIUM"
    return "HIGH"


def build_report(
    source: str,
    frame_count: int,
    total_detections: int,
    class_counter: Counter,
    confidence_values: Iterable[float],
    max_objects_per_frame: int,
) -> Dict[str, Any]:
    confidence_values = list(confidence_values)
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "project": "VisionGuard",
        "source": source,
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "average_confidence": round(average_confidence, 4),
        "maximum_objects_in_one_frame": max_objects_per_frame,
        "scene_density": density_label(max_objects_per_frame),
        "class_counts": dict(class_counter.most_common()),
    }


def save_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def print_report(report: Dict[str, Any]) -> None:
    print("\nDetection Summary")
    print("-----------------")
    print(f"Objects detected: {report['total_detections']}")
    print(f"Average confidence: {report['average_confidence']}")
    print(
        "Maximum objects in one frame: "
        f"{report['maximum_objects_in_one_frame']}"
    )
    print(f"Scene density: {report['scene_density']}")

    print("\nClass counts:")
    if report["class_counts"]:
        for name, count in report["class_counts"].items():
            print(f"  {name}: {count}")
    else:
        print("  No objects detected.")
