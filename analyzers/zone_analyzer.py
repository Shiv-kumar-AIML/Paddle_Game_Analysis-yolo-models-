"""
🎾 Court Zone Analyzer
=======================
Analyzes which court zones each player occupies during the match.
Divides the padel court into 6 zones: Left/Right × Front/Mid/Back
"""

import json
import numpy as np
from collections import defaultdict


class ZoneAnalyzer:
    """
    Analyze court zones for player positions.

    Zones:
    ┌──────────┬──────────┐
    │ Left-Front│ Right-Front│  (near net)
    ├──────────┼──────────┤
    │ Left-Mid │ Right-Mid │  (service area)
    ├──────────┼──────────┤
    │ Left-Back│ Right-Back│  (back wall)
    └──────────┴──────────┘
    """

    def __init__(self, court_data):
        """
        Initialize with court boundary data.

        Args:
            court_data: dict loaded from court_full_boundaries.json
        """
        self.net_x = (court_data['net_left']['x'] + court_data['net_right']['x']) / 2
        self.service_top = court_data['service_top_left']['y']
        self.service_bottom = court_data['service_bottom_left']['y']
        self.floor_min_x = court_data['floor_outer_top_left']['x']
        self.floor_max_x = court_data['floor_outer_bottom_right']['x']
        self.floor_min_y = court_data['floor_outer_top_left']['y']
        self.floor_max_y = court_data['floor_outer_bottom_right']['y']

        # Court pixel dimensions
        self.court_width_px = self.floor_max_x - self.floor_min_x
        self.court_height_px = self.floor_max_y - self.floor_min_y

        # Tracking data
        self.zones = defaultdict(lambda: defaultdict(int))
        self.total_frames = {1: 0, 2: 0, 3: 0, 4: 0}

    def get_zone(self, x, y):
        """
        Determine which zone a position falls in.

        Args:
            x, y: pixel coordinates

        Returns:
            Zone name (e.g. 'Left-Front') or None if outside court
        """
        if x < self.floor_min_x or x > self.floor_max_x:
            return None
        if y < self.floor_min_y or y > self.floor_max_y:
            return None

        side = "Left" if x < self.net_x else "Right"
        if y < self.service_top:
            depth = "Front"
        elif y > self.service_bottom:
            depth = "Back"
        else:
            depth = "Mid"

        return f"{side}-{depth}"

    def update(self, player_id, x, y):
        """
        Record a player's position for zone analysis.

        Args:
            player_id: Player ID (1-4)
            x, y: pixel coordinates
        """
        if player_id < 1 or player_id > 4:
            return

        zone = self.get_zone(x, y)
        if zone is not None:
            self.zones[player_id][zone] += 1
            self.total_frames[player_id] += 1

    def get_coverage(self, player_id):
        """
        Get zone coverage percentages for a player.

        Args:
            player_id: Player ID (1-4)

        Returns:
            dict of {zone_name: percentage}
        """
        total = self.total_frames.get(player_id, 0)
        if total == 0:
            return {}
        return {
            zone: (frames / total * 100)
            for zone, frames in self.zones[player_id].items()
        }

    def get_primary_zone(self, player_id):
        """Get the zone where player spent the most time"""
        coverage = self.get_coverage(player_id)
        if not coverage:
            return "N/A"
        return max(coverage, key=coverage.get)

    def get_pixels_per_meter(self):
        """
        Estimate pixels-per-meter ratio from court dimensions.
        A padel court is 10m × 20m (full court).
        """
        # Use court width: 10 meters = court_width_px pixels
        from config import COURT_WIDTH_M
        if self.court_width_px > 0:
            return self.court_width_px / COURT_WIDTH_M
        return 100.0  # fallback


def load_court_data(filepath):
    """Load court boundary data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
