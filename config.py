"""
🎾 PADEL GAME ANALYZER - Configuration
========================================
Central configuration for all paths and parameters.
Auto-detects project root and uses relative paths.
"""

import os
import argparse

# ============================================================================
# AUTO-DETECT PROJECT ROOT
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DEFAULT PATHS (relative to project root)
# ============================================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# Model paths
PLAYER_MODEL_PATH = os.path.join(MODELS_DIR, "player_detection_model.pt")
BALL_MODEL_PATH = os.path.join(MODELS_DIR, "ball_detetcion.pt")

# Data files
COURT_BOUNDS_FILE = os.path.join(DATA_DIR, "court_full_boundaries.json")

# Output files
COVERAGE_REPORT = os.path.join(OUTPUT_DIR, "court_coverage_report.txt")
COVERAGE_JSON = os.path.join(OUTPUT_DIR, "court_coverage_stats.json")
HIT_STATS_TXT = os.path.join(OUTPUT_DIR, "ball_hit_statistics.txt")
HIT_STATS_JSON = os.path.join(OUTPUT_DIR, "ball_hit_statistics.json")
PERFORMANCE_REPORT = os.path.join(OUTPUT_DIR, "PLAYER_PERFORMANCE_REPORT.txt")
PERFORMANCE_JSON = os.path.join(OUTPUT_DIR, "player_performance_metrics.json")
HEATMAP_PNG = os.path.join(OUTPUT_DIR, "COURT_COVERAGE_HEATMAP.png")
DASHBOARD_PNG = os.path.join(OUTPUT_DIR, "PLAYER_PERFORMANCE_DASHBOARD.png")

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Player detection
PLAYER_CONF_THRESHOLD = 0.15
PLAYER_IOU_THRESHOLD = 0.5
PLAYER_MAX_DET = 20
PLAYER_IMGSZ = 640

# Ball detection
BALL_CONF_THRESHOLD = 0.15       # Low threshold — ball is small, hard to detect
BALL_IOU_THRESHOLD = 0.5
BALL_MAX_DET = 20
BALL_IMGSZ = 640

# ============================================================================
# TRACKING PARAMETERS
# ============================================================================

# Player tracker
TRACKER_MAX_DISTANCE = 180      # Max px to match detection to track
TRACKER_MIN_AGE = 3             # Min frames before showing player
TRACKER_MAX_MISSING = 20        # Frames before forgetting player
MAX_PLAYERS = 4                 # Padel has exactly 4 players

# Ball tracker
BALL_MAX_TRAIL = 30             # Trail length for visualization
BALL_MATCH_DISTANCE = 150       # Max px to match ball detection
BALL_LOST_FRAMES = 15           # Frames before ball considered lost

# Hit detection — restored original proven values
HIT_HISTORY_FRAMES = 8          # Ball position history for velocity calc
HIT_VELOCITY_THRESHOLD = 30     # Original: velocity_change > 30 detected 40 hits
HIT_PLAYER_DISTANCE = 150       # Original: 120-150px distance for hit attribution
HIT_COOLDOWN_FRAMES = 30        # Original: 30 frame cooldown per player
HIT_DIRECTION_WEIGHT = 0.6      # Weight for direction change in hit scoring
HIT_VELOCITY_WEIGHT = 0.4       # Weight for velocity spike in hit scoring
HIT_MIN_BALL_SPEED = 5          # Original: vel_new > 5

# ============================================================================
# VISUALIZATION COLORS (BGR for OpenCV)
# ============================================================================

PLAYER_COLORS = {
    1: (0, 255, 0),      # Green  - Player 1
    2: (0, 0, 255),      # Red    - Player 2
    3: (255, 0, 0),      # Blue   - Player 3
    4: (0, 255, 255),    # Yellow - Player 4
}

PLAYER_COLORS_HEX = {
    1: '#00FF00',
    2: '#FF0000',
    3: '#0000FF',
    4: '#FFFF00',
}

BALL_COLOR = (0, 255, 255)       # Cyan
BALL_OUTLINE_COLOR = (255, 0, 0) # Blue
TRAIL_COLOR = (255, 100, 0)      # Orange

# ============================================================================
# COURT CONVERSION
# ============================================================================

# Padel court real dimensions
COURT_LENGTH_M = 10.0    # 10 meters per half
COURT_WIDTH_M = 10.0     # 10 meters wide
COURT_DIAGONAL_M = 14.14 # sqrt(10^2 + 10^2)

# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="🎾 PADEL GAME ANALYZER - AI-powered padel match analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --video videos/match.mp4
  python run_analysis.py --video videos/match.mp4 --live
  python run_analysis.py --video videos/match.mp4 --output output/results
        """
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to padel match video file"
    )

    parser.add_argument(
        "--live", "-l",
        action="store_true",
        default=False,
        help="Show live OpenCV window during processing"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for results (default: output/)"
    )

    parser.add_argument(
        "--player-model",
        type=str,
        default=PLAYER_MODEL_PATH,
        help="Path to player detection YOLO model"
    )

    parser.add_argument(
        "--ball-model",
        type=str,
        default=BALL_MODEL_PATH,
        help="Path to ball detection YOLO model"
    )

    parser.add_argument(
        "--court-bounds",
        type=str,
        default=COURT_BOUNDS_FILE,
        help="Path to court boundaries JSON"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for YOLO inference: 'cpu' or '0' for GPU (default: cpu)"
    )

    return parser.parse_args()
