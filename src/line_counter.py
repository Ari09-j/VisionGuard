"""
line_counter.py - VisionGuard's own analytics logic (not provided by Ultralytics).

Contains:
  * load_model()   - loads a YOLO model with a safe fallback
  * IdStitcher     - merges fragmented track IDs (own association logic)
  * LineCounter    - robust horizontal-line crossing with hysteresis,
                     minimum-frames confirmation and majority-vote class
"""

from collections import Counter, defaultdict
from math import hypot


# ---------------------------------------------------------------- model --
def load_model(preferred="yolo26n.pt", fallback="yolo11n.pt"):
    """Load a YOLO model; returns (model, weight_name). Falls back if needed."""
    from ultralytics import YOLO
    try:
        return YOLO(preferred), preferred
    except Exception as exc:  # download failed / name not known to this version
        print(f"[warn] could not load '{preferred}' ({exc}); using '{fallback}'")
        return YOLO(fallback), fallback


# ------------------------------------------------------------ stitching --
class IdStitcher:
    """
    Trackers sometimes drop an object and give it a new ID a few frames later.
    If a *new* raw ID appears close to where a same-class track was last seen
    (lost for between min_gap and max_gap frames), reuse the old canonical ID.
    """

    def __init__(self, min_gap=3, max_gap=30, max_dist=80):
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.max_dist = max_dist
        self.raw_to_canon = {}
        self.last_seen = {}  # canon -> (frame, x, y, cls)

    def resolve(self, frame, raw_id, cls, cx, cy):
        if raw_id in self.raw_to_canon:
            canon = self.raw_to_canon[raw_id]
        else:
            canon, best = raw_id, self.max_dist
            for c, (f, x, y, k) in self.last_seen.items():
                gap = frame - f
                if k != cls or not (self.min_gap <= gap <= self.max_gap):
                    continue
                d = hypot(cx - x, cy - y)
                if d <= best:
                    best, canon = d, c
            self.raw_to_canon[raw_id] = canon
        self.last_seen[canon] = (frame, cx, cy, cls)
        return canon


# ------------------------------------------------------- line crossing --
class LineCounter:
    """
    Horizontal counting line at y = line_y.

    A crossing is confirmed only when the centroid has moved beyond a margin
    on the opposite side AND stayed there for `min_frames` consecutive frames.
    This removes double counts caused by jitter around the line.
    """

    def __init__(self, line_y, margin=8, min_frames=3, stitcher=None):
        self.line_y = line_y
        self.margin = margin
        self.min_frames = min_frames
        self.stitcher = stitcher
        self.stable_side = {}                      # id -> -1 (above) / +1 (below)
        self.pending = {}                          # id -> (side, n_frames)
        self.class_votes = defaultdict(Counter)    # id -> Counter(cls)
        self.events = []
        self.counts = Counter()                    # (cls, direction) -> n
        self.crossed_ids = defaultdict(set)        # cls -> {ids}

    def _side(self, cy):
        if cy < self.line_y - self.margin:
            return -1
        if cy > self.line_y + self.margin:
            return 1
        return 0  # dead zone: ignore

    def update(self, frame, timestamp, raw_id, cls, conf, cx, cy):
        """Feed one tracked detection. Returns an event dict if a crossing is confirmed."""
        tid = (self.stitcher.resolve(frame, raw_id, cls, cx, cy)
               if self.stitcher else raw_id)

        self.class_votes[tid][cls] += 1
        voted_cls = self.class_votes[tid].most_common(1)[0][0]

        side = self._side(cy)
        if side == 0:
            return None

        if tid not in self.stable_side:
            self.stable_side[tid] = side
            return None

        if side == self.stable_side[tid]:
            self.pending.pop(tid, None)
            return None

        # on the opposite side: require min_frames consecutive confirmations
        p_side, n = self.pending.get(tid, (side, 0))
        n = n + 1 if p_side == side else 1
        self.pending[tid] = (side, n)
        if n < self.min_frames:
            return None

        direction = "IN" if side == 1 else "OUT"  # IN = moving down the frame
        self.stable_side[tid] = side
        self.pending.pop(tid, None)
        self.counts[(voted_cls, direction)] += 1
        self.crossed_ids[voted_cls].add(tid)
        event = {
            "frame": frame, "timestamp_sec": round(timestamp, 3),
            "track_id": tid, "class": voted_cls, "direction": direction,
            "confidence": round(float(conf), 3),
            "centroid_x": int(cx), "centroid_y": int(cy),
        }
        self.events.append(event)
        return event

    def summary(self):
        return {
            "crossings_by_class_direction":
                {f"{c}:{d}": n for (c, d), n in self.counts.items()},
            "unique_ids_crossed_by_class":
                {c: len(s) for c, s in self.crossed_ids.items()},
            "total_crossings": sum(self.counts.values()),
        }
