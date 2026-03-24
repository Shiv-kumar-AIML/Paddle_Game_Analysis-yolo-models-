"""
🎾 Player Performance Calculator — Fixed
==========================================
Calculates comprehensive player performance metrics.

FIXES applied:
- JSON key mismatch (str vs int) — hit_stats.get(str(pid))
- Coverage data key mismatch — coverage_data['players'].get(str(pid))
- Missing 'total_frames' field → use 'presence_frames'
- Better distance estimation using actual pixel-to-meter conversion
"""

import numpy as np


class PlayerPerformanceCalculator:
    """
    Calculate comprehensive player performance metrics from
    coverage data and hit data JSON files.
    """

    def __init__(self, coverage_data, hit_data, video_duration, fps, pixels_per_meter=100.0):
        self.coverage_data = coverage_data
        self.hit_data = hit_data
        self.video_duration = video_duration
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.player_names = ['Player 1', 'Player 2', 'Player 3', 'Player 4']

    def calculate_metrics(self):
        """Calculate all player metrics. Returns dict {pid: metrics_dict}"""

        metrics = {}
        total_frames = self.coverage_data.get('total_frames', 1)

        # FIX: hit_stats keys are STRINGS in JSON ("1", "2", ...)
        hit_stats = self.hit_data.get('hits_per_player', {})
        total_hits = self.hit_data.get('total_hits', 0)

        for pid in range(1, 5):
            # FIX: coverage data uses string keys "1", "2", ... not "player_1"
            player_key = str(pid)
            coverage_info = self.coverage_data.get('players', {}).get(player_key, {})

            # FIX: use 'presence_frames' field (not 'total_frames' which doesn't exist per-player)
            frames_on_court = coverage_info.get('presence_frames', 0)
            presence_percentage = (
                frames_on_court / total_frames * 100
                if total_frames > 0 else 0
            )

            # FIX: hit_stats keys are strings in JSON
            hits = hit_stats.get(str(pid), hit_stats.get(pid, 0))
            if isinstance(hits, str):
                hits = int(hits)
            hit_percentage = (hits / total_hits * 100) if total_hits > 0 else 0

            # Speed from coverage data (already computed during analysis)
            avg_speed_px = coverage_info.get('avg_speed', 0)
            max_speed_px = coverage_info.get('max_speed', 0)

            # If coverage data doesn't have speed, try from hit velocity data
            if avg_speed_px == 0 and max_speed_px == 0:
                speed_data = self._calculate_speed_from_hits(pid)
                avg_speed_px = speed_data['avg_speed']
                max_speed_px = speed_data['max_speed']

            # Convert speed to m/s
            avg_speed_ms = avg_speed_px / self.pixels_per_meter * self.fps
            max_speed_ms = max_speed_px / self.pixels_per_meter * self.fps

            # Distance from coverage data
            distance_m = coverage_info.get('distance_m', 0)
            if distance_m == 0:
                distance_m = self._estimate_distance(presence_percentage, avg_speed_px)

            distance_km = distance_m / 1000.0

            # Zone coverage
            zone_coverage = coverage_info.get('coverage', {})
            primary_zone = self._get_primary_zone(zone_coverage)

            # Time on court
            time_on_court = frames_on_court / self.fps if self.fps > 0 else 0

            # Performance score (0-100) — weighted composite
            performance_score = self._calculate_performance_score(
                presence_percentage, hits, total_hits,
                avg_speed_px, distance_m
            )

            metrics[pid] = {
                'name': self.player_names[pid - 1],
                'presence_percentage': round(presence_percentage, 2),
                'time_on_court': round(time_on_court, 2),
                'frames_on_court': frames_on_court,
                'hits': hits,
                'hit_percentage': round(hit_percentage, 2),
                'avg_speed': round(avg_speed_px, 2),
                'max_speed': round(max_speed_px, 2),
                'avg_speed_ms': round(avg_speed_ms, 2),
                'max_speed_ms': round(max_speed_ms, 2),
                'distance_covered': round(distance_m, 2),
                'distance_km': round(distance_km, 4),
                'primary_zone': primary_zone,
                'zone_coverage': zone_coverage,
                'performance_score': round(performance_score, 2),
            }

        return metrics

    def _calculate_speed_from_hits(self, player_id):
        """Fallback: calculate speed from hit velocity data"""
        hit_history = self.hit_data.get('hit_history', [])
        speeds = []

        for hit in hit_history:
            if hit.get('player_id') == player_id:
                velocity_str = hit.get('velocity', '0px/frame')
                try:
                    velocity = float(velocity_str.split('px/frame')[0])
                    speeds.append(velocity)
                except (ValueError, IndexError):
                    pass

        if not speeds:
            return {'avg_speed': 0, 'max_speed': 0}

        # Reject outliers (top 5%)
        if len(speeds) > 5:
            sorted_speeds = sorted(speeds)
            cutoff = max(1, int(len(sorted_speeds) * 0.95))
            speeds = sorted_speeds[:cutoff]

        return {
            'avg_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
        }

    def _estimate_distance(self, presence_pct, avg_speed_px):
        """Estimate distance when not available from tracker data"""
        # presence_pct is 0-100, avg_speed is px/frame
        # Rough: time_on_court * avg_speed * pixels_to_meters
        time_on_court = (presence_pct / 100.0) * self.video_duration
        if avg_speed_px > 0 and self.pixels_per_meter > 0:
            speed_m_per_s = avg_speed_px / self.pixels_per_meter * self.fps
            return time_on_court * speed_m_per_s
        return 0.0

    def _get_primary_zone(self, zone_coverage):
        """Get zone with highest percentage"""
        if not zone_coverage:
            return "N/A"
        return max(zone_coverage, key=zone_coverage.get)

    def _calculate_performance_score(self, presence, hits, total_hits, avg_speed, distance):
        """
        Calculate overall performance score (0-100).
        Weighted scoring:
          30% Court Presence
          30% Ball Hits (relative to total)
          20% Speed/Agility
          20% Distance Coverage
        """
        # Presence score (0-100)
        presence_score = min(100, presence * 1.2)

        # Hit score (0-100) — based on share of total hits
        hit_share = (hits / total_hits * 100) if total_hits > 0 else 0
        hits_score = min(100, hit_share * 4)  # 25% share = 100 score

        # Speed score (0-100) — normalized
        speed_score = min(100, avg_speed * 3)

        # Distance score (0-100) — assume 50m is excellent
        distance_score = min(100, (distance / 50.0) * 100)

        performance = (
            presence_score * 0.30
            + hits_score * 0.30
            + speed_score * 0.20
            + distance_score * 0.20
        )

        return min(100, max(0, performance))
