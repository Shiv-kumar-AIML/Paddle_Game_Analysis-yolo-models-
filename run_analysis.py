"""
🎾 PADEL GAME ANALYZER - Unified Entry Point
==============================================
Single script that runs the complete analysis pipeline:
  1. Load YOLO models (player + ball detection)
  2. Process video frame-by-frame
  3. Track players (stable IDs) + ball + detect hits
  4. Analyze court zone coverage
  5. Generate reports + dashboards

Usage:
  python run_analysis.py --video videos/match.mp4
  python run_analysis.py --video videos/match.mp4 --live
  python run_analysis.py --video videos/match.mp4 --output output/results
"""

import os
import sys

# CRITICAL: Force matplotlib to use Agg backend BEFORE any imports
# This prevents conflict with OpenCV's Qt backend
os.environ['MPLBACKEND'] = 'Agg'

import cv2
import numpy as np
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trackers import StablePlayerTracker, BallTracker, BallHitDetector
from analyzers.zone_analyzer import ZoneAnalyzer, load_court_data
from analyzers.performance import PlayerPerformanceCalculator
from reporters.text_report import generate_coverage_report, generate_performance_report
from reporters.dashboard import generate_heatmap, generate_dashboard


def print_banner():
    print("\n" + "=" * 100)
    print("🎾 PADEL GAME ANALYZER — Complete Match Analysis Pipeline")
    print("=" * 100)


def validate_files(args):
    """Check that all required files exist"""
    errors = []

    if not os.path.exists(args.video):
        errors.append(f"❌ Video not found: {args.video}")
    if not os.path.exists(args.player_model):
        errors.append(f"❌ Player model not found: {args.player_model}")
    if not os.path.exists(args.ball_model):
        errors.append(f"❌ Ball model not found: {args.ball_model}")
    if not os.path.exists(args.court_bounds):
        errors.append(f"❌ Court boundaries not found: {args.court_bounds}")

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"  ✓ Video: {args.video}")
    print(f"  ✓ Player Model: {args.player_model}")
    print(f"  ✓ Ball Model: {args.ball_model}")
    print(f"  ✓ Court Bounds: {args.court_bounds}")


def load_models(args):
    """Load YOLO models"""
    from ultralytics import YOLO

    print("\n🔄 Loading YOLO models...")
    player_model = YOLO(args.player_model)
    ball_model = YOLO(args.ball_model)
    print("  ✓ Models loaded successfully")
    return player_model, ball_model


def open_video(video_path):
    """Open video and get properties"""
    print(f"\n▶️  Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  ✓ Resolution: {width}×{height}")
    print(f"  ✓ FPS: {fps}")
    print(f"  ✓ Duration: {duration:.1f}s ({total_frames} frames)")

    return cap, fps, width, height, total_frames, duration


def detect_players(player_model, frame, device='cpu'):
    """Run YOLO player detection on a frame"""
    results = player_model(
        frame,
        conf=config.PLAYER_CONF_THRESHOLD,
        iou=config.PLAYER_IOU_THRESHOLD,
        max_det=config.PLAYER_MAX_DET,
        imgsz=config.PLAYER_IMGSZ,
        verbose=False,
        device=device,
    )

    boxes = results[0].boxes
    detections = []

    for box, conf in zip(boxes.xyxy, boxes.conf):
        x1, y1, x2, y2 = box.cpu().numpy()
        conf_val = float(conf)
        detections.append({
            'box': np.array([x1, y1, x2, y2]),
            'conf': conf_val,
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
        })

    # Sort by confidence, keep top candidates
    detections.sort(key=lambda x: x['conf'], reverse=True)
    return detections[:10]


def detect_ball(ball_model, frame, device='cpu'):
    """Run YOLO ball detection on a frame"""
    results = ball_model(
        frame,
        conf=config.BALL_CONF_THRESHOLD,
        iou=config.BALL_IOU_THRESHOLD,
        max_det=config.BALL_MAX_DET,
        imgsz=config.BALL_IMGSZ,
        verbose=False,
        device=device,
    )

    boxes = results[0].boxes
    detections = []

    for box, conf in zip(boxes.xyxy, boxes.conf):
        x1, y1, x2, y2 = box.cpu().numpy()
        conf_val = float(conf)
        detections.append({
            'box': np.array([x1, y1, x2, y2]),
            'conf': conf_val,
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
        })

    return detections


def draw_overlay(frame, tracked_players, ball_tracker, hit_detector, hit_player,
                 frame_idx, total_frames, num_players, width, height, fps):
    """Draw visualization overlay on frame"""
    output = frame.copy()

    # Sort by player ID for consistent display
    sorted_players = sorted(tracked_players, key=lambda x: x.get('pid', 999))

    for det in sorted_players:
        pid = det.get('pid', 0)
        if pid < 1 or pid > 4:
            continue

        x1, y1, x2, y2 = det['box'].astype(int)
        conf = det['conf']
        color = config.PLAYER_COLORS.get(pid, (255, 255, 255))

        # Thicker box if this player just hit
        thickness = 5 if hit_player == pid else 3
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"P{pid} ({conf:.2f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(output, (x1, y1 - 35), (x1 + text_size[0] + 15, y1), color, -1)
        cv2.putText(output, label, (x1 + 8, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(output, (cx, cy), 8, color, -1)
        cv2.circle(output, (cx, cy), 18, color, 2)

    # Draw Ball
    if ball_tracker.is_detected():
        ball_x, ball_y = ball_tracker.ball_position.astype(int)
        cv2.circle(output, (ball_x, ball_y), 12, config.BALL_COLOR, -1)
        cv2.circle(output, (ball_x, ball_y), 12, config.BALL_OUTLINE_COLOR, 3)
        cv2.putText(output, "BALL", (ball_x - 25, ball_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.BALL_OUTLINE_COLOR, 2)

        # Draw trail
        trail = ball_tracker.get_trail()
        if len(trail) > 1:
            for i in range(1, len(trail)):
                pt1 = trail[i - 1].astype(int)
                pt2 = trail[i].astype(int)
                cv2.line(output, tuple(pt1), tuple(pt2), config.TRAIL_COLOR, 2)

    # Compact header overlay (semi-transparent)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, 50), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

    # Line 1: Frame | Players | Ball
    p_color = (0, 255, 0) if num_players == 4 else (0, 165, 255)
    b_icon = "Y" if ball_tracker.is_detected() else "N"
    b_color = (0, 255, 255) if ball_tracker.is_detected() else (100, 100, 100)
    line1 = f"F:{frame_idx}/{total_frames}"
    cv2.putText(output, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, f"PLR:{num_players}/4", (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 1)
    cv2.putText(output, f"BALL:{b_icon}", (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1)

    # Line 2: Hits
    stats = hit_detector.get_stats()
    total_hits = sum(stats.values())
    hits_text = f"HITS:{total_hits} [P1:{stats[1]} P2:{stats[2]} P3:{stats[3]} P4:{stats[4]}]"
    h_color = (0, 255, 100) if hit_player else (200, 200, 200)
    cv2.putText(output, hits_text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1)

    # Thin progress bar at bottom
    progress = frame_idx / total_frames if total_frames > 0 else 0
    bar_x = int(width * progress)
    cv2.rectangle(output, (0, height - 4), (bar_x, height), (0, 200, 0), -1)

    return output


def process_video(player_model, ball_model, cap, fps, width, height,
                  total_frames, duration, court_data, args):
    """
    Main processing loop: detect → track → analyze → report.
    """
    print("\n" + "=" * 100)
    print("🎬 PROCESSING VIDEO")
    print("=" * 100)

    # Initialize trackers
    player_tracker = StablePlayerTracker(
        max_distance=config.TRACKER_MAX_DISTANCE,
        min_age=config.TRACKER_MIN_AGE,
        max_missing=config.TRACKER_MAX_MISSING,
        max_players=config.MAX_PLAYERS,
    )
    ball_tracker = BallTracker(
        max_trail=config.BALL_MAX_TRAIL,
        match_distance=config.BALL_MATCH_DISTANCE,
        lost_frames=config.BALL_LOST_FRAMES,
    )
    hit_detector = BallHitDetector(
        velocity_threshold=config.HIT_VELOCITY_THRESHOLD,
        hit_distance=config.HIT_PLAYER_DISTANCE,
        cooldown_frames=config.HIT_COOLDOWN_FRAMES,
        min_ball_speed=config.HIT_MIN_BALL_SPEED,
    )
    zone_analyzer = ZoneAnalyzer(court_data)

    # Tracking accumulators
    player_presence = {1: 0, 2: 0, 3: 0, 4: 0}
    frame_idx = 0
    start_time = time.time()

    WINDOW_NAME = "PADEL GAME ANALYZER"

    if args.live:
        print("Controls: Q = Quit | P = Pause/Resume | S = Save Frame\n")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    is_paused = False

    try:
        while True:
            if is_paused and args.live:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord('P'):
                    is_paused = False
                    print("▶️  RESUMED")
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # === DETECT ===
            player_dets = detect_players(player_model, frame, device=args.device)
            ball_dets = detect_ball(ball_model, frame, device=args.device)

            # === TRACK ===
            tracked_players = player_tracker.update(player_dets)
            ball_tracker.update(ball_dets)
            num_players = player_tracker.get_active_players()

            # === HIT DETECTION ===
            hit_player = hit_detector.detect_hits(
                tracked_players,
                ball_tracker.ball_position,
                frame_idx,
                fps,
            )

            # === ZONE ANALYSIS ===
            for det in tracked_players:
                pid = det.get('pid', 0)
                if 1 <= pid <= 4:
                    player_presence[pid] += 1
                    cx = (det['box'][0] + det['box'][2]) / 2
                    cy = (det['box'][1] + det['box'][3]) / 2
                    zone_analyzer.update(pid, cx, cy)

            # === LIVE DISPLAY ===
            if args.live:
                output_frame = draw_overlay(
                    frame, tracked_players, ball_tracker, hit_detector,
                    hit_player, frame_idx, total_frames, num_players,
                    width, height, fps,
                )
                cv2.imshow(WINDOW_NAME, output_frame)

                delay = max(1, int(1000 / fps))
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n\n⏹️  Stopped by user")
                    break
                elif key == ord('p') or key == ord('P'):
                    is_paused = True
                    print("\n⏸️  PAUSED")
                elif key == ord('s') or key == ord('S'):
                    save_path = os.path.join(args.output, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(save_path, output_frame)
                    print(f"\n💾 Saved: {save_path}")

            # === CONSOLE PROGRESS ===
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                fps_val = frame_idx / elapsed if elapsed > 0 else 0
                progress = frame_idx / total_frames if total_frames > 0 else 0
                filled = int(50 * progress)
                bar = "█" * filled + "░" * (50 - filled)

                stats = hit_detector.get_stats()
                total_hits = sum(stats.values())

                print(
                    f"\r  [{bar}] {frame_idx}/{total_frames} ({progress * 100:.1f}%) | "
                    f"Players: {num_players}/4 | "
                    f"Ball: {'✓' if ball_tracker.is_detected() else '✗'} | "
                    f"Hits: {total_hits} | "
                    f"FPS: {fps_val:.1f}",
                    end='', flush=True,
                )

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")

    finally:
        cap.release()
        if args.live:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n\n  ✓ Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS)")

    return player_tracker, ball_tracker, hit_detector, zone_analyzer, player_presence, frame_idx


def save_results(player_tracker, ball_tracker, hit_detector, zone_analyzer,
                 player_presence, court_data, frame_idx, fps, duration, args):
    """Save all results: JSONs, TXT reports, dashboards"""

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print("💾 SAVING RESULTS")
    print("=" * 100)

    # Compute pixels-per-meter from court data
    pixels_per_meter = zone_analyzer.get_pixels_per_meter()

    # === 1. Build coverage metrics ===
    coverage_metrics = {}
    for pid in range(1, 5):
        distance_m = player_tracker.calculate_distance(pid, pixels_per_meter)
        distance_km = distance_m / 1000.0
        max_speed = player_tracker.get_max_speed(pid)
        avg_speed = player_tracker.get_avg_speed(pid)
        activity = (player_presence[pid] / frame_idx * 100) if frame_idx > 0 else 0
        num_hits = hit_detector.hit_stats[pid]
        coverage = zone_analyzer.get_coverage(pid)

        # Performance score
        perf = min(100, activity * 1.2 + max_speed * 2 + (distance_m / 100) + (num_hits * 5))

        coverage_metrics[pid] = {
            'activity': activity,
            'distance_m': distance_m,
            'distance_km': distance_km,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'performance': perf,
            'coverage': coverage,
            'presence_frames': player_presence[pid],
            'num_hits': num_hits,
        }

    # === 2. Save hit statistics ===
    hit_txt = os.path.join(output_dir, "ball_hit_statistics.txt")
    hit_json = os.path.join(output_dir, "ball_hit_statistics.json")
    hit_detector.save_stats(hit_txt, hit_json)
    print(f"  ✓ Hit Stats TXT: {hit_txt}")
    print(f"  ✓ Hit Stats JSON: {hit_json}")

    # === 3. Save coverage JSON ===
    coverage_json_path = os.path.join(output_dir, "court_coverage_stats.json")
    coverage_json_data = {
        'generated': datetime.now().isoformat(),
        'duration': float(duration),
        'total_frames': int(frame_idx),
        'players': {}
    }

    for pid, data in coverage_metrics.items():
        coverage_json_data['players'][str(pid)] = {
            'activity': float(data['activity']),
            'distance_m': float(data['distance_m']),
            'distance_km': float(data['distance_km']),
            'max_speed': float(data['max_speed']),
            'avg_speed': float(data['avg_speed']),
            'performance': float(data['performance']),
            'num_hits': int(data['num_hits']),
            'coverage': {k: float(v) for k, v in data['coverage'].items()},
            'presence_frames': int(data['presence_frames']),
        }

    with open(coverage_json_path, 'w') as f:
        json.dump(coverage_json_data, f, indent=2)
    print(f"  ✓ Coverage JSON: {coverage_json_path}")

    # === 4. Save coverage report ===
    coverage_report_path = os.path.join(output_dir, "court_coverage_report.txt")
    generate_coverage_report(coverage_report_path, coverage_metrics, duration, frame_idx)
    print(f"  ✓ Coverage Report: {coverage_report_path}")

    # === 5. Generate heatmap ===
    heatmap_path = os.path.join(output_dir, "COURT_COVERAGE_HEATMAP.png")
    generate_heatmap(heatmap_path, coverage_metrics)
    print(f"  ✓ Heatmap: {heatmap_path}")

    # === 6. Generate performance report + dashboard from saved data ===
    # Load the JSONs we just saved (to test the fixed loader)
    with open(coverage_json_path, 'r') as f:
        cov_data = json.load(f)
    with open(hit_json, 'r') as f:
        hit_data = json.load(f)

    calc = PlayerPerformanceCalculator(
        coverage_data=cov_data,
        hit_data=hit_data,
        video_duration=duration,
        fps=fps,
        pixels_per_meter=pixels_per_meter,
    )
    perf_metrics = calc.calculate_metrics()

    # Save performance JSON
    perf_json_path = os.path.join(output_dir, "player_performance_metrics.json")
    perf_json_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_duration': duration,
        'total_frames': frame_idx,
        'fps': fps,
        'players': perf_metrics,
    }
    with open(perf_json_path, 'w') as f:
        json.dump(perf_json_data, f, indent=2, default=str)
    print(f"  ✓ Performance JSON: {perf_json_path}")

    # Save performance report
    perf_report_path = os.path.join(output_dir, "PLAYER_PERFORMANCE_REPORT.txt")
    generate_performance_report(perf_report_path, perf_metrics, duration, frame_idx, fps)
    print(f"  ✓ Performance Report: {perf_report_path}")

    # Save dashboard
    dashboard_path = os.path.join(output_dir, "PLAYER_PERFORMANCE_DASHBOARD.png")
    generate_dashboard(dashboard_path, perf_metrics)
    print(f"  ✓ Dashboard: {dashboard_path}")

    return coverage_metrics, perf_metrics


def print_summary(coverage_metrics, perf_metrics, hit_detector, duration):
    """Print final summary to console"""
    print("\n" + "=" * 100)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 100)

    stats = hit_detector.get_stats()
    total_hits = sum(stats.values())

    print(f"\n🎾 Total Ball Hits: {total_hits}")
    for pid in range(1, 5):
        print(f"   Player {pid}: {stats[pid]} hits")

    print(f"\n🏆 Performance Rankings:")
    sorted_players = sorted(perf_metrics.items(), key=lambda x: x[1]['performance_score'], reverse=True)
    medals = ['🥇', '🥈', '🥉', '  ']
    for rank, (pid, data) in enumerate(sorted_players):
        medal = medals[rank] if rank < 4 else '  '
        print(f"   {medal} {data['name']}: {data['performance_score']:.1f}/100 "
              f"({data['hits']} hits, {data['presence_percentage']:.1f}% presence)")

    print("\n" + "=" * 100)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 100)


def main():
    print_banner()

    # Parse arguments
    args = config.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Validate files
    print("\n📋 Checking files...")
    validate_files(args)

    # Load court boundaries
    court_data = load_court_data(args.court_bounds)

    # Load models
    player_model, ball_model = load_models(args)

    # Open video
    cap, fps, width, height, total_frames, duration = open_video(args.video)

    # Process video
    (player_tracker, ball_tracker, hit_detector,
     zone_analyzer, player_presence, processed_frames) = process_video(
        player_model, ball_model, cap, fps, width, height,
        total_frames, duration, court_data, args,
    )

    # Save results
    coverage_metrics, perf_metrics = save_results(
        player_tracker, ball_tracker, hit_detector, zone_analyzer,
        player_presence, court_data, processed_frames, fps, duration, args,
    )

    # Print summary
    print_summary(coverage_metrics, perf_metrics, hit_detector, duration)


if __name__ == "__main__":
    main()
