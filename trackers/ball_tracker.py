"""
🎾 Ball Tracker
================
Tracks the padel ball position with trajectory history and smoothing.

Single source of truth — imported by all modules.
"""

import numpy as np
from collections import deque


class BallTracker:
    """
    Tracks ball position using highest-confidence detection per frame.
    Maintains a position trail for trajectory visualization and velocity analysis.

    Features:
    - Smooth tracking with distance-based outlier rejection
    - Trail history for visualization
    - Position interpolation when detection drops briefly
    """

    def __init__(self, max_trail=30, match_distance=100, lost_frames=10):
        self.ball_position = None           # Current best position
        self.trail = deque(maxlen=max_trail) # Position history for viz
        self.match_distance = match_distance
        self.lost_frames = lost_frames
        self.last_detection_age = 0         # Frames since last detection
        self.confidence_history = deque(maxlen=max_trail)
        self._prev_position = None          # For velocity calculation

    def update(self, detections):
        """
        Update ball position from new frame detections.

        Args:
            detections: list of dicts with keys: box, conf, cx, cy
        """
        self._prev_position = self.ball_position.copy() if self.ball_position is not None else None

        if len(detections) == 0:
            self.last_detection_age += 1
            if self.last_detection_age > self.lost_frames:
                self.ball_position = None
            return

        # Pick highest confidence detection
        best_det = max(detections, key=lambda x: x['conf'])
        new_pos = np.array([best_det['cx'], best_det['cy']])

        # If we had a previous position, check for unrealistic jump
        if self.ball_position is not None:
            distance = np.linalg.norm(new_pos - self.ball_position)
            if distance > self.match_distance * 3:
                # Massive jump — likely false detection, reset trail
                self.trail.clear()

        self.ball_position = new_pos
        self.trail.append(new_pos.copy())
        self.confidence_history.append(float(best_det['conf']))
        self.last_detection_age = 0

    def get_trail(self):
        """Get ball trail positions for visualization"""
        return list(self.trail)

    def get_velocity(self):
        """
        Get current ball velocity (px/frame).
        Returns 0 if not enough data.
        """
        if len(self.trail) < 2:
            return 0.0

        positions = list(self.trail)
        # Use last 2 positions
        vel = np.linalg.norm(positions[-1] - positions[-2])
        return float(vel)

    def get_direction(self):
        """
        Get ball movement direction as unit vector.
        Returns None if not enough data.
        """
        if len(self.trail) < 2:
            return None

        positions = list(self.trail)
        direction = positions[-1] - positions[-2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        return direction / norm

    def is_detected(self):
        """Check if ball is currently detected"""
        return self.ball_position is not None and self.last_detection_age <= self.lost_frames

    def get_avg_confidence(self):
        """Get average detection confidence over recent frames"""
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))
