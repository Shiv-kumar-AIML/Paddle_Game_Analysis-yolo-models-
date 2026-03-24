"""
🎾 Stable Player Tracker — Restored Original Logic
=====================================================
Tracks up to 4 padel players with stable IDs using Hungarian algorithm.
Restored from the original ball_hit_stats.py / court_coverage_analysis.py
that achieved perfect ID stability.

Key features:
- Position-based initial assignment (sorted by y then x)
- Hungarian algorithm for frame-to-frame matching
- Player slots kept open for max_missing frames when out of view
- Speed/distance tracking built in
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class StablePlayerTracker:
    """
    Tracks players with stable IDs (1-4) across a padel match.

    Original proven logic:
    - First frame: sort detections by position (top-left to bottom-right)
      and assign IDs 1-4
    - Subsequent frames: use Hungarian algorithm for optimal matching
    - If player goes off-screen, their slot is kept for max_missing frames
    - When they return, they get the same ID back
    """

    def __init__(self, max_distance=180, min_age=3, max_missing=20, max_players=4):
        self.players = {}
        self.positions = defaultdict(list)
        self.speeds = defaultdict(list)
        self.max_distance = max_distance
        self.min_age = min_age
        self.max_missing = max_missing
        self.max_players = max_players
        self._initialized = False

    def update(self, detections):
        """
        Update tracking with new frame detections.

        Args:
            detections: list of dicts with keys: box, conf, cx, cy

        Returns:
            list of tracked detections with 'pid' field added
        """

        if len(detections) == 0:
            # No detections: increment missing counter
            for pid in list(self.players.keys()):
                self.players[pid]['missing'] += 1
                if self.players[pid]['missing'] > self.max_missing:
                    del self.players[pid]
            return []

        current_centers = np.array([[d['cx'], d['cy']] for d in detections])

        # ============================================================
        # FIRST FRAME: Position-based initial assignment
        # Sort by y-coordinate first, then x — gives consistent IDs
        # Top-left player = 1, top-right = 2, etc.
        # ============================================================
        if not self._initialized:
            # Sort detections by y first, then x (top to bottom, left to right)
            sorted_indices = sorted(
                range(len(detections)),
                key=lambda i: (detections[i]['cy'], detections[i]['cx'])
            )

            for rank, idx in enumerate(sorted_indices[:self.max_players]):
                pid = rank + 1
                center = np.array([detections[idx]['cx'], detections[idx]['cy']])
                self.players[pid] = {
                    'center': center,
                    'box': detections[idx]['box'],
                    'conf': detections[idx]['conf'],
                    'age': 0,
                    'missing': 0,
                }
                detections[idx]['pid'] = pid
                self.positions[pid].append(center.copy())

            self._initialized = True
            return []  # Don't show players until min_age

        # ============================================================
        # SUBSEQUENT FRAMES: Hungarian algorithm matching
        # ============================================================
        tracked_ids = sorted(list(self.players.keys()))
        if len(tracked_ids) == 0:
            # All players were lost, re-initialize
            self._initialized = False
            return self.update(detections)

        tracked_centers = np.array([self.players[pid]['center'] for pid in tracked_ids])

        # Build cost matrix: distance between each detection and each tracked player
        dist_matrix = np.zeros((len(detections), len(tracked_ids)))
        for i in range(len(detections)):
            for j in range(len(tracked_ids)):
                dist_matrix[i, j] = np.linalg.norm(
                    current_centers[i] - tracked_centers[j]
                )

        # Solve assignment problem
        det_indices, tracked_indices = linear_sum_assignment(dist_matrix)

        # Filter by max distance
        matched_det = set()
        matched_tracked = set()

        for d_idx, t_idx in zip(det_indices, tracked_indices):
            if dist_matrix[d_idx, t_idx] < self.max_distance:
                pid = tracked_ids[t_idx]
                old_center = self.players[pid]['center']
                new_center = current_centers[d_idx]

                # Record speed
                speed = np.linalg.norm(new_center - old_center)
                self.speeds[pid].append(speed)

                # Update player
                self.players[pid]['center'] = new_center
                self.players[pid]['box'] = detections[d_idx]['box']
                self.players[pid]['conf'] = detections[d_idx]['conf']
                self.players[pid]['age'] += 1
                self.players[pid]['missing'] = 0
                detections[d_idx]['pid'] = pid

                # Record position
                self.positions[pid].append(new_center.copy())

                matched_det.add(d_idx)
                matched_tracked.add(t_idx)

        # Handle unmatched detections — assign new IDs from available slots
        for d_idx in range(len(detections)):
            if d_idx not in matched_det:
                # Find available ID (1-4 that is not currently active)
                available = [
                    i for i in range(1, self.max_players + 1)
                    if i not in self.players
                ]
                if available:
                    pid = available[0]
                    center = current_centers[d_idx]
                    self.players[pid] = {
                        'center': center,
                        'box': detections[d_idx]['box'],
                        'conf': detections[d_idx]['conf'],
                        'age': 0,
                        'missing': 0,
                    }
                    detections[d_idx]['pid'] = pid
                    self.positions[pid].append(center.copy())

        # Handle unmatched tracked players — increment missing
        for t_idx in range(len(tracked_ids)):
            if t_idx not in matched_tracked:
                pid = tracked_ids[t_idx]
                self.players[pid]['missing'] += 1
                if self.players[pid]['missing'] > self.max_missing:
                    del self.players[pid]

        # Build result: only players with age >= min_age, IDs 1-4
        result = []
        for det in detections:
            if 'pid' not in det:
                continue
            pid = det['pid']
            if pid < 1 or pid > self.max_players:
                continue
            if pid in self.players and self.players[pid]['age'] >= self.min_age:
                result.append(det)

        # Cap at max players, keep highest confidence
        if len(result) > self.max_players:
            result.sort(key=lambda x: x['conf'], reverse=True)
            result = result[:self.max_players]

        return result

    def get_active_players(self):
        """Count of established players"""
        return min(
            len([p for pid, p in self.players.items()
                 if p['age'] >= self.min_age and pid <= self.max_players]),
            self.max_players
        )

    def calculate_distance(self, player_id, pixels_per_meter=None):
        """Total distance covered by player (in pixels or meters)"""
        positions = self.positions.get(player_id, [])
        if len(positions) < 2:
            return 0.0

        total = sum(
            np.linalg.norm(positions[i] - positions[i - 1])
            for i in range(1, len(positions))
        )
        if pixels_per_meter and pixels_per_meter > 0:
            return total / pixels_per_meter
        return total

    def get_max_speed(self, player_id):
        """Max speed with outlier rejection (top 1%)"""
        speeds = self.speeds.get(player_id, [])
        if not speeds:
            return 0.0
        if len(speeds) > 20:
            sorted_s = sorted(speeds)
            cutoff = int(len(sorted_s) * 0.99)
            return max(sorted_s[:cutoff]) if cutoff > 0 else 0.0
        return max(speeds)

    def get_avg_speed(self, player_id):
        """Average speed with outlier rejection (top 5%)"""
        speeds = self.speeds.get(player_id, [])
        if not speeds:
            return 0.0
        if len(speeds) > 20:
            sorted_s = sorted(speeds)
            cutoff = int(len(sorted_s) * 0.95)
            filtered = sorted_s[:cutoff]
            return float(np.mean(filtered)) if filtered else 0.0
        return float(np.mean(speeds))

    def get_presence_frames(self, player_id):
        """Number of frames this player was tracked"""
        return len(self.positions.get(player_id, []))
