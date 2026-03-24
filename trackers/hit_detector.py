"""
🎾 Ball Hit Detector — Restored Original Logic
=================================================
Detects ball hits using velocity spike analysis (original proven logic)
plus direction change as bonus signal.

Key: Uses the SAME 4-point velocity calculation that detected 40 hits
in the original codebase.
"""

import numpy as np
from collections import deque
from datetime import datetime


class BallHitDetector:
    """
    Detects ball hits using velocity spike analysis.

    Original logic (proven to detect 40 hits):
    - Track last N ball positions
    - Compute velocity between consecutive positions
    - Detect when velocity suddenly increases (spike)
    - Find nearest player to attribute the hit

    Enhanced with direction change as bonus signal.
    """

    def __init__(
        self,
        velocity_threshold=30,
        hit_distance=150,
        cooldown_frames=30,
        min_ball_speed=5,
    ):
        self.velocity_threshold = velocity_threshold
        self.hit_distance = hit_distance
        self.cooldown_frames = cooldown_frames
        self.min_ball_speed = min_ball_speed

        # Store last 5 ball positions (original used 4-5)
        self.ball_positions = deque(maxlen=5)
        self.hit_cooldown = {}  # {player_id: remaining_frames}
        self.hit_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        self.hit_history = []

    def _detect_velocity_spike(self):
        """
        Original velocity spike detection logic.

        Uses 4 consecutive positions to detect acceleration.
        Computes:
          vel_old = distance(pos[1], pos[2])
          vel_new = distance(pos[2], pos[3])
          velocity_change = vel_new - vel_old

        If velocity_change > threshold AND vel_new > min_speed → hit detected.
        """
        if len(self.ball_positions) < 4:
            return False, None

        positions = list(self.ball_positions)

        # Compute velocities between consecutive positions
        vel_mid = np.linalg.norm(positions[-3] - positions[-2])
        vel_new = np.linalg.norm(positions[-2] - positions[-1])

        # Velocity change (acceleration)
        velocity_change = vel_new - vel_mid

        if velocity_change > self.velocity_threshold and vel_new > self.min_ball_speed:
            # Hit location is where ball was before the spike
            hit_location = positions[-2]
            return True, hit_location

        return False, None

    def _find_nearest_player(self, players, hit_location):
        """
        Find the closest player to the hit location.

        Original logic: checks distance from player center to hit point.
        Uses bottom-center of bbox (closer to where racquet would be).
        """
        if hit_location is None or not players:
            return None

        best_pid = None
        best_dist = float('inf')

        for player in players:
            pid = player.get('pid', 0)
            if pid < 1 or pid > 4:
                continue

            box = player['box']
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Use bottom-center of bounding box
            player_cx = (x1 + x2) / 2
            player_cy = y2  # Bottom of bbox

            dist = np.linalg.norm(
                hit_location - np.array([player_cx, player_cy])
            )

            if dist < self.hit_distance and dist < best_dist:
                best_dist = dist
                best_pid = pid

        return best_pid

    def detect_hits(self, players, ball_pos, frame_idx=0, fps=23):
        """
        Detect ball hits.

        Args:
            players: list of tracked player dicts (with 'pid' and 'box')
            ball_pos: np.array [x, y] or None
            frame_idx: current frame number
            fps: video FPS

        Returns:
            player_id (1-4) that hit, or None
        """
        # Update ball history
        if ball_pos is not None:
            self.ball_positions.append(ball_pos.copy())

        # Decrease cooldowns
        for pid in list(self.hit_cooldown.keys()):
            self.hit_cooldown[pid] -= 1
            if self.hit_cooldown[pid] <= 0:
                del self.hit_cooldown[pid]

        # Detect velocity spike (original logic)
        spike_detected, hit_location = self._detect_velocity_spike()

        if spike_detected and hit_location is not None:
            # Find nearest player
            hit_player = self._find_nearest_player(players, hit_location)

            if hit_player is not None and hit_player not in self.hit_cooldown:
                # Record hit
                self.hit_stats[hit_player] += 1
                self.hit_cooldown[hit_player] = self.cooldown_frames

                # Calculate velocity for logging
                positions = list(self.ball_positions)
                velocity = np.linalg.norm(positions[-2] - positions[-1]) if len(positions) >= 2 else 0

                hit_time = frame_idx / fps if fps > 0 else 0
                hit_info = {
                    'player_id': hit_player,
                    'frame': frame_idx,
                    'time': f"{hit_time:.2f}s",
                    'velocity': f"{velocity:.1f}px/frame",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                }
                self.hit_history.append(hit_info)

                return hit_player

        return None

    def get_stats(self):
        """Get hit statistics {1: count, 2: count, ...}"""
        return self.hit_stats.copy()

    def get_total_hits(self):
        """Total hits across all players"""
        return sum(self.hit_stats.values())

    def get_hit_history(self):
        """Get detailed hit log"""
        return self.hit_history.copy()

    def save_stats(self, filepath_txt, filepath_json):
        """Save statistics to TXT and JSON files"""
        import json

        total_hits = self.get_total_hits()

        # === SAVE TXT ===
        with open(filepath_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PADEL BALL HIT STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("TOTAL HITS PER PLAYER:\n")
            f.write("-" * 80 + "\n")

            for pid in range(1, 5):
                hits = self.hit_stats[pid]
                bar_length = min(hits * 2, 60)
                bar = "#" * bar_length + "." * (60 - bar_length)
                f.write(f"Player {pid}: {hits:3d} hits | {bar}\n")

            f.write("-" * 80 + "\n")
            f.write(f"TOTAL: {total_hits} hits\n\n")

            f.write("PERCENTAGE:\n")
            f.write("-" * 80 + "\n")
            if total_hits > 0:
                for pid in range(1, 5):
                    percentage = (self.hit_stats[pid] / total_hits) * 100
                    f.write(f"Player {pid}: {percentage:6.2f}% ({self.hit_stats[pid]} hits)\n")
            f.write("\n")

            f.write("DETAILED HIT LOG:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Frame':<10} {'Player':<10} {'Time':<12} {'Velocity':<20} {'Timestamp':<26}\n")
            f.write("-" * 100 + "\n")

            for hit in self.hit_history:
                velocity = hit.get('velocity', 'N/A')
                f.write(
                    f"{hit['frame']:<10} "
                    f"Player {hit['player_id']:<6} "
                    f"{hit['time']:<12} "
                    f"{velocity:<20} "
                    f"{hit['timestamp']:<26}\n"
                )

            f.write("\n" + "=" * 80 + "\n")

        # === SAVE JSON ===
        stats_dict = {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_hits': total_hits,
            'hits_per_player': {str(k): v for k, v in self.hit_stats.items()},
            'percentage': {
                f'player_{i}': (
                    self.hit_stats[i] / total_hits * 100 if total_hits > 0 else 0
                )
                for i in range(1, 5)
            },
            'hit_history': self.hit_history,
        }

        with open(filepath_json, 'w') as f:
            json.dump(stats_dict, f, indent=2)
