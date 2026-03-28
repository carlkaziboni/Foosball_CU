import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
import time
from typing import Callable, Optional
from pathlib import Path

# Allow importing servo_control when running from pythonserver/
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = str(Path(__file__).parent / "best.pt")

# Set RL_MODEL_PATH via env var, e.g.:
#   export RL_MODEL_PATH=/path/to/model.zip
RL_MODEL_PATH: Optional[str] = os.environ.get("RL_MODEL_PATH")

from ball_tracker      import KalmanBallTracker, draw_ball_overlay
from figure_angle      import estimate_angles_for_result, draw_figure_angles
from formation_tracker import (
    build_figure_list,
    assign_figures_to_rods,
    unify_rod_angles,
    draw_formation_overlay,
)
from obs_builder       import ObsBuilder
from rod_detector      import detect_rod_ys, draw_rod_lines
from table_geometry    import (build_homography, project_all_positions,
                               refine_formation_by_geometry, draw_ghost_overlay,
                               get_formation_data)

# Calibration values
TABLE_X_MIN = 10.1
TABLE_X_MAX = 627.0
TABLE_Y_MIN = 13.7
TABLE_Y_MAX = 401.6


def run(on_tracker_message: Callable[[dict], None], bridge=None) -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Device
    # ─────────────────────────────────────────────────────────────────────────

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # Models
    # ─────────────────────────────────────────────────────────────────────────

    # Main detector: ball (class 0) + figures (class 1)
    model = YOLO(MODEL_PATH)

    # RL policy — set RL_MODEL_PATH env var to enable
    rl_model = None
    if RL_MODEL_PATH is not None:
        try:
            from stable_baselines3 import SAC
            rl_model = SAC.load(RL_MODEL_PATH)
            print(f"RL model loaded: {RL_MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Could not load RL model: {e}")

    if bridge is not None:
        bridge.open()
        print("Serial bridge opened ✓")

    # Dedicated ball tracker with Kalman filter (ball-class-only inference)
    ball_tracker = KalmanBallTracker(
        model_path=MODEL_PATH,
        ball_class_id=0,
        conf=0.25,
        max_coasted_frames=12,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Camera
    # ─────────────────────────────────────────────────────────────────────────

    cap = cv2.VideoCapture(0)
    # 4:3 capture gives more height → more width after 90° rotation → full table in view.
    # Try in order: 1600×1200, 1280×960, 1024×768. Camera uses nearest supported mode.
    for _w, _h in [(1600, 1200), (1280, 960), (1024, 768)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _h)
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (_w, _h):
            break
    cap.set(cv2.CAP_PROP_FPS,   30)
    cap.set(cv2.CAP_PROP_ZOOM, 100)   # minimum zoom (ignored if unsupported)

    # Rotation to correct for camera orientation.
    FRAME_ROTATE = cv2.ROTATE_90_CLOCKWISE

    # After 90° rotation: new_w = captured_h, new_h = captured_w.
    # Scale so width = PROC_W; height preserves aspect ratio.
    PROC_W = 960
    _cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PROC_H = int(PROC_W * _cap_w / _cap_h)   # e.g. 960 × 1280/960 = 1280
    print(f"Capture: {_cap_w}×{_cap_h}  →  rotated & resized to: {PROC_W}×{PROC_H}")

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Starting real-time detection. Press 'q' to quit.")
    print("=" * 60)

    obs_builder = ObsBuilder()
    prev_rot = np.zeros(4, dtype=np.float32)

    sequence_num = 0
    frame_count  = 0
    fps_time     = time.time()
    fps          = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.rotate(frame, FRAME_ROTATE)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (PROC_W, PROC_H))
            h, w  = frame.shape[:2]

            # ── 1. Main YOLO detection (ball + figures) ───────────────────────
            results       = model(frame, conf=0.3, verbose=False, device=device)
            annotated     = results[0].plot()

            # ── 2. Rod detection (chrome/silver colour mask) ──────────────────
            rod_ys        = detect_rod_ys(frame)
            annotated     = draw_rod_lines(annotated, rod_ys)

            # ── 3. Figure kick-angle estimation ──────────────────────────────
            angle_data    = estimate_angles_for_result(frame, results[0])
            annotated     = draw_figure_angles(annotated, angle_data)

            # ── 4. Team classification + formation assignment ─────────────────
            figures       = build_figure_list(frame, results[0], angle_data)
            rods          = assign_figures_to_rods(figures, frame_height=h, rod_ys=rod_ys)
            unify_rod_angles(rods)   # all figures on the same rod share one consensus angle

            # ── 4b. Refine assignment + ghost overlay ─────────────────────────
            H = build_homography(rods)
            if H is not None:
                rods     = refine_formation_by_geometry(figures, H)
                H        = build_homography(rods)          # rebuild from refined rods
            if H is not None:
                expected_img = project_all_positions(H)
                annotated    = draw_ghost_overlay(annotated, expected_img, rods)

            annotated     = draw_formation_overlay(annotated, rods)

            # ── 5. Dedicated ball tracker with Kalman filter ──────────────────
            predicted_pos, detected_box, coasting, _ = ball_tracker.update(frame)
            annotated = draw_ball_overlay(
                annotated,
                predicted_pos,
                detected_box,
                coasting,
                trail=ball_tracker.trail,
                speed=ball_tracker.speed,
            )

            ball_status = "coasting" if coasting else ("tracked" if predicted_pos is not None else "lost")
            on_tracker_message({
                "type": "ball_status",
                "status": ball_status,
                "coasting": coasting,
                "detected": detected_box is not None,
            })

            # Send ball position via callback
            if predicted_pos is not None:
                x, y = predicted_pos
                # Map camera coordinates to table coordinates and clamp to avoid
                # dropping updates when tracking goes just outside calibration bounds.
                mapped_x = 640 - ((x - TABLE_X_MIN) / (TABLE_X_MAX - TABLE_X_MIN)) * 640
                mapped_y = ((y - TABLE_Y_MIN) / (TABLE_Y_MAX - TABLE_Y_MIN)) * 480
                mapped_x = max(0.0, min(640.0, mapped_x))
                mapped_y = max(0.0, min(480.0, mapped_y))
                on_tracker_message({
                    "type": "ball_position",
                    "x": mapped_x,
                    "y": mapped_y,
                    "sequenceNum": sequence_num,
                    "status": ball_status,
                    "coasting": coasting,
                    "detected": detected_box is not None,
                })
                sequence_num += 1

            # ── 6. Build RL observation + run policy ──────────────────────────
            obs = obs_builder.update(ball_tracker, rods, H)
            action = None
            if obs is not None and rl_model is not None:
                action, _ = rl_model.predict(obs, deterministic=True)
                if bridge is not None:
                    bridge.send_action(action, prev_rot)
                prev_rot = action[1::2].astype(np.float32)
                on_tracker_message({
                    "type": "action",
                    "action": action.tolist(),
                })

            # ── 7. FPS ────────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_time    = time.time()

            # ── 8. HUD ────────────────────────────────────────────────────────
            detection_count = len(results[0].boxes)
            ball_status     = ("Coasting" if coasting
                               else ("Tracked" if predicted_pos else "Lost"))
            obs_status      = "ready" if obs is not None else "waiting"

            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detections: {detection_count}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Ball: {ball_status}",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(annotated, f"Obs: {obs_status}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            if action is not None:
                act_str = " ".join(f"{v:+.2f}" for v in action)
                cv2.putText(annotated, f"Act: {act_str}",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            cv2.imshow("Foosball Detection - Real-time", annotated)

            # Console log every 30 ticks
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {detection_count} det | "
                      f"FPS {fps:.1f} | Ball: {ball_status}")
                formation_data = get_formation_data(rods, H)
                for s in formation_data:
                    pos_str = "  ".join(f"({x:.0f},{y:.0f})" for x, y in s.expected_positions)
                    ang_str = f"{s.angle_deg:+.1f}°" if s.angle_deg is not None else "n/a"
                    print(f"  {s.label:<8} | {pos_str or 'no homography':<55} | {ang_str}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting…")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if bridge is not None:
            bridge.close()
        print("Detection stopped.")
