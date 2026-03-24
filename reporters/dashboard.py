"""
🎾 Dashboard Generator
=======================
Generates matplotlib charts: court coverage heatmap and performance dashboard.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt


def generate_heatmap(filepath, metrics, dpi=150):
    """
    Generate court coverage heatmap (2x2 grid, one per player).

    Args:
        filepath: output PNG path
        metrics: dict {pid: {coverage: {zone: pct}, performance, activity, distance_km}}
        dpi: image quality
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('PADEL COURT COVERAGE HEATMAP', fontsize=16, fontweight='bold')

    colors_players = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00']
    zone_positions = {
        'Left-Front': (0, 0), 'Right-Front': (0, 1),
        'Left-Mid': (1, 0), 'Right-Mid': (1, 1),
        'Left-Back': (2, 0), 'Right-Back': (2, 1),
    }

    for pid in range(1, 5):
        ax = axes[(pid - 1) // 2, (pid - 1) % 2]

        coverage = metrics[pid].get('coverage', {})

        if not coverage:
            ax.text(0.5, 0.5, f'P{pid}\nNo Data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor('#f0f0f0')
            continue

        # Create 3x2 grid for zones
        grid = np.zeros((3, 2))
        labels = [['', ''], ['', ''], ['', '']]
        for zone, pct in coverage.items():
            if zone in zone_positions:
                row, col = zone_positions[zone]
                grid[row, col] = pct
                labels[row][col] = f"{pct:.0f}%"

        max_val = max(grid.max(), 50)
        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)

        # Add text labels on each cell
        for i in range(3):
            for j in range(2):
                if labels[i][j]:
                    ax.text(j, i, labels[i][j], ha='center', va='center',
                            color='black', fontsize=12, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Left', 'Right'])
        ax.set_yticklabels(['Front', 'Mid', 'Back'])

        perf = metrics[pid].get('performance', 0)
        act = metrics[pid].get('activity', 0)
        dist = metrics[pid].get('distance_km', 0)
        hits = metrics[pid].get('num_hits', 0)

        title = f"Player {pid}\nPerf: {perf:.0f} | Act: {act:.0f}% | Hits: {hits} | Dist: {dist:.2f}km"
        ax.set_title(title, fontweight='bold', color=colors_players[pid - 1])
        plt.colorbar(im, ax=ax, label='Time (%)')

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def generate_dashboard(filepath, metrics, dpi=150):
    """
    Generate performance dashboard with 6 charts.

    Args:
        filepath: output PNG path
        metrics: dict from PlayerPerformanceCalculator.calculate_metrics()
        dpi: image quality
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🎾 PADEL PLAYER PERFORMANCE DASHBOARD', fontsize=22, fontweight='bold', y=0.98)

    colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00']
    player_labels = [metrics[i]['name'] for i in range(1, 5)]
    player_ids = list(range(1, 5))

    performance_scores = [metrics[i]['performance_score'] for i in player_ids]
    hit_counts = [metrics[i]['hits'] for i in player_ids]
    presence = [metrics[i]['presence_percentage'] for i in player_ids]
    distances = [metrics[i]['distance_covered'] for i in player_ids]
    avg_speeds = [metrics[i]['avg_speed'] for i in player_ids]

    # --- 1. Performance Scores ---
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(player_labels, performance_scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    for bar, score in zip(bars, performance_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                 f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    # --- 2. Ball Hits ---
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(player_labels, hit_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Hits', fontsize=11, fontweight='bold')
    ax2.set_title('Total Ball Hits', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars, hit_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # --- 3. Court Presence ---
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(player_labels, presence, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Court Presence', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars, presence):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # --- 4. Distance ---
    ax4 = plt.subplot(2, 3, 4)
    bars = ax4.bar(player_labels, distances, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax4.set_ylabel('Distance (meters)', fontsize=11, fontweight='bold')
    ax4.set_title('Distance Coverage', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, dist in zip(bars, distances):
        ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                 f'{dist:.1f}m', ha='center', va='bottom', fontweight='bold')

    # --- 5. Average Speed ---
    ax5 = plt.subplot(2, 3, 5)
    bars = ax5.bar(player_labels, avg_speeds, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax5.set_ylabel('Speed (px/frame)', fontsize=11, fontweight='bold')
    ax5.set_title('Average Movement Speed', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, speed in zip(bars, avg_speeds):
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{speed:.1f}', ha='center', va='bottom', fontweight='bold')

    # --- 6. Rankings Table ---
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    sorted_players = sorted(player_ids, key=lambda x: metrics[x]['performance_score'], reverse=True)
    table_data = [['RANK', 'PLAYER', 'SCORE', 'HITS', 'DISTANCE', 'SPEED']]
    medals = ['🥇', '🥈', '🥉', '  ']

    for rank, pid in enumerate(sorted_players):
        d = metrics[pid]
        table_data.append([
            f"{medals[rank]} {rank + 1}",
            d['name'],
            f"{d['performance_score']:.1f}",
            str(d['hits']),
            f"{d['distance_covered']:.1f}m",
            f"{d['avg_speed']:.1f}",
        ])

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.12, 0.2, 0.12, 0.12, 0.2, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#404040')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for rank, pid in enumerate(sorted_players):
        for col in range(6):
            table[(rank + 1, col)].set_facecolor(colors[pid - 1])
            table[(rank + 1, col)].set_alpha(0.3)

    ax6.set_title('FINAL RANKINGS', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
