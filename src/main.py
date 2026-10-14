import argparse
from pathlib import Path
import sys
import time

import cv2

from analyzer import (
    build_report,
    is_image_source,
    is_video_source,
    print_report,
    save_report,
)
from line_counter import load_model
from tracker import TrackAnalytics, draw_dashboard


MODEL_NAME = "yolo26n.pt"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VisionGuard - Advanced Computer Vision Analytics"
    )
    parser.add_argument("--source", required=True, help="Image or video path/URL.")
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Minimum detection confidence."
    )
    parser.add_argument(
        "--output", default="outputs/result",
        help="Output directory."
    )
    parser.add_argument(
        "--tracker", default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Multi-object tracker."
    )
    parser.add_argument(
        "--line-ratio", type=float, default=0.60,
        help="Counting-line vertical position as a fraction of frame height."
    )
    parser.add_argument(
        "--margin", type=int, default=8,
        help="Dead-zone (pixels) on each side of the line to suppress jitter."
    )
    parser.add_argument(
        "--min-frames", type=int, default=3,
        help="Consecutive frames an object must stay past the line to count."
    )
    parser.add_argument(
        "--no-stitch", action="store_true",
        help="Disable ID stitching (use raw tracker IDs)."
    )
    return parser.parse_args()


def validate(args):
    if not 0 <= args.confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not 0.1 <= args.line_ratio <= 0.9:
        raise ValueError("Line ratio must be between 0.1 and 0.9.")


def process_image(model, source, output_dir, confidence):
    results = model.predict(source=source, conf=confidence, verbose=False)
    if not results:
        raise RuntimeError("No prediction result returned.")

    result = results[0]
    annotated = result.plot()
    output_image = output_dir / "annotated_image.jpg"
    cv2.imwrite(str(output_image), annotated)

    from collections import Counter
    class_counter = Counter()
    confidence_values = []

    if result.boxes is not None:
        for cls_id, conf in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            class_counter[result.names[int(cls_id)]] += 1
            confidence_values.append(float(conf))

    object_count = sum(class_counter.values())
    report = build_report(
        source, 1, object_count, class_counter, confidence_values, object_count
    )
    save_report(report, output_dir / "report.json")
    print_report(report)
    print(f"\nAnnotated image: {output_image}")


def process_video(model, model_name, source, output_dir, confidence, tracker_name,
                  line_ratio, margin=8, min_frames=3, stitch=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    line_y = int(height * line_ratio)
    analytics = TrackAnalytics(line_y, margin=margin, min_frames=min_frames,
                              stitch=stitch)

    output_video = output_dir / "tracked_output.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_no = 0
    total_detections = 0
    confidence_values = []
    class_counter = {}

    print(f"Tracker: {tracker_name}")
    print("Processing video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        timestamp = (frame_no - 1) / fps

        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_name,
            conf=confidence,
            verbose=False,
        )
        result = results[0]

        boxes = []
        track_ids = []
        class_names = []
        confidences = []

        if result.boxes is not None:
            for box, cls_id, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            ):
                boxes.append(box)
                class_names.append(result.names[int(cls_id)])
                confidences.append(float(conf))
                total_detections += 1
                confidence_values.append(float(conf))

            ids = result.boxes.id
            if ids is not None:
                track_ids = ids.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)
        else:
            track_ids = []

        analytics.update(
            frame_no, timestamp, boxes, track_ids, class_names, confidences
        )

        annotated = result.plot()
        annotated = draw_dashboard(annotated, analytics, frame_no)
        writer.write(annotated)

        if frame_no % 30 == 0:
            print(
                f"Processed {frame_no} frames | "
                f"Unique tracks: {len(analytics.unique_track_ids)}",
                end="\r",
            )

    cap.release()
    writer.release()

    frame_csv, event_csv, summary_csv, summary = analytics.save(output_dir)

    report = {
        "project": "VisionGuard Advanced",
        "source": source,
        "model": model_name,
        "line_margin_px": margin,
        "min_frames_to_confirm": min_frames,
        "id_stitching": stitch,
        "tracker": tracker_name,
        "frames_processed": frame_no,
        "total_detections": total_detections,
        "average_detection_confidence": round(
            sum(confidence_values) / len(confidence_values), 4
        ) if confidence_values else 0.0,
        **summary,
        "outputs": {
            "tracked_video": str(output_video),
            "frame_analytics": str(frame_csv),
            "track_events": str(event_csv),
            "summary": str(summary_csv),
        },
    }
    save_report(report, output_dir / "advanced_report.json")

    print("\n\n================ FINAL ANALYTICS ================")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFiles generated:")
    print(output_video)
    print(frame_csv)
    print(event_csv)
    print(summary_csv)
    print(output_dir / "advanced_report.json")


def main():
    args = parse_arguments()
    validate(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("VISIONGUARD - ADVANCED COMPUTER VISION ANALYTICS")
    print("=" * 62)

    try:
        model, model_name = load_model(MODEL_NAME)

        if is_image_source(args.source):
            process_image(model, args.source, output_dir, args.confidence)
        elif is_video_source(args.source):
            process_video(
                model, model_name, args.source, output_dir, args.confidence,
                args.tracker, args.line_ratio,
                args.margin, args.min_frames, not args.no_stitch
            )
        else:
            raise ValueError("Advanced analytics currently requires a video file.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
