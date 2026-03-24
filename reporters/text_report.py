"""
🎾 Text Report Generator
==========================
Generates formatted TXT reports for court coverage and player performance.
"""

from datetime import datetime
import numpy as np


def generate_coverage_report(filepath, metrics, duration, total_frames):
    """
    Generate court coverage text report.

    Args:
        filepath: output TXT file path
        metrics: dict {pid: {activity, distance_m, distance_km, max_speed,
                            avg_speed, performance, coverage, presence_frames, num_hits}}
        duration: video duration in seconds
        total_frames: total frame count
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n")
        f.write("╔" + "═" * 98 + "╗\n")
        f.write("║" + " " * 25 + "🎾 PADEL COURT COVERAGE ANALYSIS REPORT 🎾" + " " * 30 + "║\n")
        f.write("╚" + "═" * 98 + "╝\n\n")

        f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"⏱️  Duration: {duration:.1f} seconds ({int(duration//60)}m {int(duration%60)}s)\n\n")

        f.write("┌" + "─" * 98 + "┐\n")
        f.write("│ 📋 PLAYER SUMMARY" + " " * 81 + "│\n")
        f.write("└" + "─" * 98 + "┘\n\n")

        for pid in range(1, 5):
            data = metrics[pid]
            f.write(f"{'─' * 100}\n")
            f.write(f"👤 PLAYER {pid}\n")
            f.write(f"{'─' * 100}\n\n")

            # Performance score
            perf = data.get('performance', 0)
            bar_len = max(0, min(50, int(perf / 2)))
            perf_bar = "█" * bar_len + "░" * (50 - bar_len)
            f.write(f"🎯 PERFORMANCE SCORE: {perf:.1f}/100\n")
            f.write(f"   [{perf_bar}]\n\n")

            # Activity
            act = data.get('activity', 0)
            act_bar_len = max(0, min(50, int(act / 2)))
            activity_bar = "█" * act_bar_len + "░" * (50 - act_bar_len)
            f.write(f"🏃 ACTIVITY: {act:.1f}%\n")
            f.write(f"   [{activity_bar}] ({data.get('presence_frames', 0)} frames on court)\n\n")

            # Speed
            f.write(f"⚡ SPEED ANALYSIS:\n")
            f.write(f"   ├─ Max Speed: {data.get('max_speed', 0):.2f} px/frame\n")
            f.write(f"   └─ Avg Speed: {data.get('avg_speed', 0):.2f} px/frame\n\n")

            # Hits
            f.write(f"🎾 BALL HITS:\n")
            f.write(f"   └─ Total Hits: {data.get('num_hits', 0)}\n\n")

            # Distance
            f.write(f"📏 DISTANCE COVERED:\n")
            f.write(f"   ├─ {data.get('distance_m', 0):.0f} meters\n")
            f.write(f"   └─ {data.get('distance_km', 0):.3f} km\n\n")

            # Zone coverage
            f.write(f"🗺️  COURT ZONE COVERAGE:\n")
            coverage = data.get('coverage', {})
            if coverage:
                sorted_zones = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
                for i, (zone, pct) in enumerate(sorted_zones):
                    z_bar_len = max(0, min(25, int(pct / 4)))
                    bar = "█" * z_bar_len + "░" * (25 - z_bar_len)
                    prefix = "   ├─" if i < len(sorted_zones) - 1 else "   └─"
                    f.write(f"{prefix} {zone:<12}: {pct:>6.1f}% [{bar}]\n")
            else:
                f.write("   └─ No data\n")
            f.write("\n\n")

        # Summary table
        f.write("┌" + "─" * 98 + "┐\n")
        f.write("│ 📊 QUICK COMPARISON" + " " * 78 + "│\n")
        f.write("├" + "─" * 98 + "┤\n")
        f.write(f"│ {'Player':<10} {'Performance':<14} {'Activity':<12} {'Hits':<8} {'Distance':<14} {'Max Speed':<12} │\n")
        f.write("├" + "─" * 98 + "┤\n")

        for pid in range(1, 5):
            d = metrics[pid]
            f.write(
                f"│ P{pid:<9} "
                f"{d.get('performance', 0):<13.1f} "
                f"{d.get('activity', 0):<11.1f}% "
                f"{d.get('num_hits', 0):<7} "
                f"{d.get('distance_km', 0):<13.3f}km "
                f"{d.get('max_speed', 0):<11.2f} │\n"
            )

        f.write("└" + "─" * 98 + "┘\n")


def generate_performance_report(filepath, metrics, video_duration, total_frames, fps):
    """
    Generate comprehensive player performance text report.

    Args:
        filepath: output TXT file path
        metrics: dict from PlayerPerformanceCalculator.calculate_metrics()
        video_duration: video duration in seconds
        total_frames: total frame count
        fps: video FPS
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n")
        f.write("╔" + "═" * 118 + "╗\n")
        f.write("║" + " " * 40 + "🎾 PADEL MATCH PERFORMANCE REPORT 🎾" + " " * 42 + "║\n")
        f.write("╚" + "═" * 118 + "╝\n\n")

        f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"⏱️  Match Duration: {video_duration:.1f} seconds ({int(video_duration // 60)}m {int(video_duration % 60)}s)\n")
        f.write(f"📊 Total Frames: {total_frames}\n")
        f.write(f"🎥 Video FPS: {fps}\n\n")

        # Rankings
        f.write("┌" + "─" * 118 + "┐\n")
        f.write("│ 🏆 PERFORMANCE RANKINGS (Score: 0-100)" + " " * 78 + "│\n")
        f.write("└" + "─" * 118 + "┘\n\n")

        sorted_by_score = sorted(metrics.items(), key=lambda x: x[1]['performance_score'], reverse=True)
        for rank, (pid, data) in enumerate(sorted_by_score, 1):
            score = data['performance_score']
            bar_length = max(0, min(50, int(score / 2)))
            bar = "█" * bar_length + "░" * (50 - bar_length)
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            f.write(f"{medal} Rank {rank}: {data['name']:<15} | Score: {score:>6.2f}/100 | {bar}\n")

        f.write("\n\n")

        # Detailed stats per player
        for pid in range(1, 5):
            data = metrics[pid]

            f.write("┌" + "─" * 118 + "┐\n")
            f.write(f"│ 👤 {data['name'].upper():<114}│\n")
            f.write("├" + "─" * 118 + "┤\n")

            # Performance Score
            s_len = max(0, min(50, int(data['performance_score'] / 2)))
            score_bar = "█" * s_len + "░" * (50 - s_len)
            f.write(f"│ 📊 Performance Score: {data['performance_score']:>6.2f}/100  [{score_bar}]\n│\n")

            # Presence
            f.write(f"│ 🏟️  Court Presence: {data['time_on_court']:>8.2f}s ({data['presence_percentage']:>5.2f}%) — {data['frames_on_court']} frames\n│\n")

            # Hits
            freq = data['hits'] / video_duration if video_duration > 0 else 0
            f.write(f"│ 🎾 Ball Hits: {data['hits']:>3} ({data['hit_percentage']:>5.2f}% of all) — {freq:.2f} hits/sec\n│\n")

            # Speed
            f.write(f"│ ⚡ Avg Speed: {data['avg_speed']:>8.2f} px/frame ({data.get('avg_speed_ms', 0):.2f} m/s)\n")
            f.write(f"│    Max Speed: {data['max_speed']:>8.2f} px/frame ({data.get('max_speed_ms', 0):.2f} m/s)\n│\n")

            # Distance
            rate = data['distance_covered'] / video_duration if video_duration > 0 else 0
            f.write(f"│ 📏 Distance: {data['distance_covered']:>8.2f}m ({data['distance_km']:.4f}km) — {rate:.2f} m/s avg\n")
            f.write(f"│    Primary Zone: {data['primary_zone']}\n│\n")

            # Zone Distribution
            f.write(f"│ 🗺️  Zone Distribution:\n")
            zc = data.get('zone_coverage', {})
            if zc:
                sorted_zones = sorted(zc.items(), key=lambda x: x[1], reverse=True)
                for i, (zone, pct) in enumerate(sorted_zones):
                    b_len = max(0, min(30, int(pct / 3.3)))
                    bar = "█" * b_len + "░" * (30 - b_len)
                    prefix = "│    ├─" if i < len(sorted_zones) - 1 else "│    └─"
                    f.write(f"{prefix} {zone:<12}: {pct:>6.2f}% {bar}\n")
            else:
                f.write("│    └─ No data\n")

            f.write("│\n")

        # Match summary
        f.write("┌" + "─" * 118 + "┐\n")
        f.write("│ 📈 MATCH STATISTICS SUMMARY" + " " * 91 + "│\n")
        f.write("├" + "─" * 118 + "┤\n")

        total_hits = sum(m['hits'] for m in metrics.values())
        avg_presence = np.mean([m['presence_percentage'] for m in metrics.values()])
        avg_speed = np.mean([m['avg_speed'] for m in metrics.values()])
        total_distance = sum(m['distance_covered'] for m in metrics.values())

        f.write(f"│ Total Ball Hits: {total_hits}\n")
        f.write(f"│ Average Court Presence: {avg_presence:.2f}%\n")
        f.write(f"│ Average Player Speed: {avg_speed:.2f} px/frame\n")
        f.write(f"│ Total Distance Covered: {total_distance:.2f} meters\n")
        f.write("╚" + "═" * 118 + "╝\n")
