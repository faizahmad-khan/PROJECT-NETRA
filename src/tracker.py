"""
🚦 PROJECT NETRA - Vehicle Tracking Module
ByteTrack multi-object tracking for persistent vehicle identification across frames.

Features:
  - Unique vehicle counting per lane (eliminates double-counting)
  - Per-vehicle speed estimation (pixels/second)
  - Movement trail history for visualization
  - Lane-aware assignment with cumulative session stats
"""

from __future__ import annotations

import supervision as sv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import time


class VehicleTracker:
    """
    Wraps supervision's ByteTrack for multi-object tracking with
    lane assignment, unique counting, and speed estimation.

    Usage:
        tracker = VehicleTracker(frame_rate=30)
        tracked = tracker.update(detections)
        info = tracker.process(tracked, lane1_box, lane2_box)
    """

    def __init__(self, frame_rate: int = 30, trail_length: int = 30) -> None:
        """
        Args:
            frame_rate: Video FPS (used by ByteTrack for motion prediction)
            trail_length: Max number of positions to store per track
        """
        self.frame_rate: int = frame_rate
        self.trail_length: int = trail_length

        self.byte_tracker: sv.ByteTrack = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate,
        )

        # Per-track state
        self.positions: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)  # track_id -> [(cx, cy, t), ...]
        self.speeds: Dict[int, float] = {}                    # track_id -> px/s

        # Cumulative unique IDs per lane (session-level)
        self.unique_ids: Dict[str, Set[int]] = {"lane1": set(), "lane2": set()}

    # ------------------------------------------------------------------
    # Core tracking
    # ------------------------------------------------------------------

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Run ByteTrack on current-frame detections.
        Returns a new Detections object with tracker_id populated.
        """
        return self.byte_tracker.update_with_detections(detections)

    def process(
        self,
        tracked: sv.Detections,
        lane1_box: List[float],
        lane2_box: List[float],
    ) -> Dict[str, Any]:
        """
        Assign tracked detections to lanes, compute speeds, store trails.

        Args:
            tracked:   sv.Detections with tracker_id populated
            lane1_box: [x_min, y_min, x_max, y_max]
            lane2_box: [x_min, y_min, x_max, y_max]

        Returns:
            dict with keys:
                lane1_count, lane2_count   – current-frame counts
                lane1_speeds, lane2_speeds – list of speeds (px/s)
                tracks                     – list of per-vehicle dicts
        """
        now: float = time.time()
        out: Dict[str, Any] = {
            "lane1_count": 0,
            "lane2_count": 0,
            "lane1_speeds": [],
            "lane2_speeds": [],
            "tracks": [],
        }

        if tracked.tracker_id is None:
            return out

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Store position history
            self.positions[tid].append((cx, cy, now))
            if len(self.positions[tid]) > self.trail_length:
                self.positions[tid] = self.positions[tid][-self.trail_length:]

            # Speed estimation
            speed: float = self._estimate_speed(tid)
            self.speeds[tid] = speed

            # Lane assignment
            lane: Optional[str] = None
            if (lane1_box[0] < cx < lane1_box[2]
                    and lane1_box[1] < cy < lane1_box[3]):
                lane = "lane1"
                out["lane1_count"] += 1
                out["lane1_speeds"].append(speed)
                self.unique_ids["lane1"].add(tid)

            elif (lane2_box[0] < cx < lane2_box[2]
                  and lane2_box[1] < cy < lane2_box[3]):
                lane = "lane2"
                out["lane2_count"] += 1
                out["lane2_speeds"].append(speed)
                self.unique_ids["lane2"].add(tid)

            out["tracks"].append({
                "id": int(tid),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (int(cx), int(cy)),
                "speed": speed,
                "lane": lane,
                "trail": [(int(px), int(py))
                          for px, py, _ in self.positions[tid]],
            })

        return out

    # ------------------------------------------------------------------
    # Speed estimation
    # ------------------------------------------------------------------

    def _estimate_speed(self, tid: int) -> float:
        """Smoothed speed in pixels/second over the last 5 positions."""
        pts: List[Tuple[float, float, float]] = self.positions[tid]
        if len(pts) < 2:
            return 0.0
        recent: List[Tuple[float, float, float]] = pts[-min(5, len(pts)):]
        dist: float = sum(
            np.sqrt(
                (recent[j][0] - recent[j - 1][0]) ** 2
                + (recent[j][1] - recent[j - 1][1]) ** 2
            )
            for j in range(1, len(recent))
        )
        dt: float = recent[-1][2] - recent[0][2]
        return dist / dt if dt > 0 else 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_unique_counts(self) -> Tuple[int, int]:
        """Total unique vehicles ever seen per lane in this session."""
        return len(self.unique_ids["lane1"]), len(self.unique_ids["lane2"])

    def active_tracks(self) -> int:
        """Number of track IDs with stored positions."""
        return len(self.positions)

    def reset(self) -> None:
        """
        Reset tracker state (call on video loop / new session).
        NOTE: unique_ids are intentionally NOT cleared — they track
        session totals. Call reset_unique_counts() separately if needed.
        """
        self.byte_tracker.reset()
        self.positions.clear()
        self.speeds.clear()

    def reset_unique_counts(self) -> None:
        """Clear cumulative unique vehicle IDs."""
        self.unique_ids["lane1"].clear()
        self.unique_ids["lane2"].clear()
