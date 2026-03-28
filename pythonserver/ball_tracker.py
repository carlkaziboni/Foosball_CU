"""
ball_tracker.py
---------------
Dedicated foosball tracker:
  - YOLO inference restricted to the ball class only (separate from the main
    foosball+figure detector in prediction.py)
  - Kalman filter (constant-velocity model) to maintain a smooth position
    estimate and coast through frames where the ball is too fast or briefly
    occluded.

State vector  : [cx, cy, vx, vy]
Measurement   : [cx, cy]
"""

import cv2
import numpy as np
from ultralytics import YOLO


class KalmanBallTracker:
    """
    Detects the ball with a dedicated YOLO model and wraps the raw detections
    with a Kalman filter so that a position estimate is always available even
    when the ball moves too fast for a reliable detection.

    Parameters
    ----------
    model_path : str
        Path to the YOLO weights file used *only* for ball detection.
    ball_class_id : int
        Class index for "ball" in the model (default 0 for the trained foosball model).
    conf : float
        YOLO confidence threshold (0–1).
    max_coasted_frames : int
        Maximum number of consecutive frames to keep predicting without a
        detection before declaring the ball lost.
    process_noise : float
        Kalman process-noise magnitude – increase if the ball accelerates sharply.
    measurement_noise : float
        Kalman measurement-noise magnitude – increase to trust the filter more
        than raw detections (useful for very noisy / bouncing detections).
    """

    def __init__(
        self,
        model_path: str,
        ball_class_id: int = 0,
        conf: float = 0.25,
        max_coasted_frames: int = 12,
        process_noise: float = 1e-2,
        measurement_noise: float = 5e-2,
        device: str = "cpu",
    ):
        # --- Dedicated ball-only YOLO model ---
        self.model = YOLO(model_path)
        self.device = device
        self.ball_class_id = ball_class_id
        self.conf = conf
        self.max_coasted_frames = max_coasted_frames

        # --- Kalman filter: 4 state variables, 2 measurements ---
        self.kf = cv2.KalmanFilter(4, 2)

        # Measurement matrix H  (maps state → measurement)
        #   measurement = H * state  →  [cx, cy] = H * [cx, cy, vx, vy]
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        # Transition matrix F  (constant-velocity model, dt = 1 frame)
        #   x_{k+1} = F * x_k
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )

        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initial error covariance P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self._initialized = False
        self.frames_since_detection = 0

        # Trail of recent predicted positions for visualisation
        self.trail: list[tuple[int, int]] = []
        self.trail_length = 20

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_ball(self, frame: np.ndarray):
        """
        Run YOLO restricted to the ball class and return the
        highest-confidence hit as (cx, cy, w, h), or None.
        Also returns the raw results object for optional downstream use.
        """
        results = self.model(
            frame,
            conf=self.conf,
            classes=[self.ball_class_id],
            verbose=False,
            device=self.device,
        )
        boxes = results[0].boxes

        if len(boxes) == 0:
            return None, results

        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h), results

    def _init_state(self, cx: float, cy: float) -> None:
        self.kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Force-reset the tracker (e.g. after a scene cut)."""
        self._initialized = False
        self.frames_since_detection = 0
        self.trail.clear()

    def update(self, frame: np.ndarray):
        """
        Run one tracker step on *frame*.

        Returns
        -------
        predicted_pos : (int, int) | None
            Kalman-smoothed (cx, cy) in pixel coordinates.
            None if the ball has never been seen or has been lost.
        detected_box  : (float, float, float, float) | None
            Raw YOLO detection (cx, cy, w, h) this frame, or None.
        coasting      : bool
            True when we are predicting without a fresh YOLO detection.
        yolo_results  :
            Raw Ultralytics results object (useful for drawing other classes).
        """
        detection, yolo_results = self._detect_ball(frame)

        # --- Kalman predict ---
        predicted = self.kf.predict()

        if detection is not None:
            cx, cy, *_ = detection
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])

            if not self._initialized:
                self._init_state(cx, cy)

            self.kf.correct(meas)
            self.frames_since_detection = 0
            coasting = False
        else:
            self.frames_since_detection += 1
            coasting = True

        # --- Lost-ball guard ---
        if not self._initialized:
            return None, None, False, yolo_results

        if self.frames_since_detection > self.max_coasted_frames:
            self.reset()
            return None, None, False, yolo_results

        px = int(round(float(predicted[0])))
        py = int(round(float(predicted[1])))

        # Update trail
        self.trail.append((px, py))
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)

        # WebSocket sending commented out - handled in prediction.py instead
        # if detection is not None:
        #     cx, cy, w, h = detection
        #     # Map camera coordinates to table coordinates (X is flipped)
        #     mapped_x = 640 - ((cx - self.TABLE_X_MIN) / (self.TABLE_X_MAX - self.TABLE_X_MIN)) * 640
        #     mapped_y = ((cy - self.TABLE_Y_MIN) / (self.TABLE_Y_MAX - self.TABLE_Y_MIN)) * 480
        #     asyncio.run(self.send_ball_position(mapped_x, mapped_y, KalmanBallTracker.sequence_num))
        #     KalmanBallTracker.sequence_num += 1

        return (px, py), detection, coasting, yolo_results

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Estimated (vx, vy) in pixels/frame, or None if not initialised."""
        if not self._initialized:
            return None
        return float(self.kf.statePost[2]), float(self.kf.statePost[3])

    @property
    def speed(self) -> float | None:
        """Scalar speed in pixels/frame, or None if not initialised."""
        v = self.velocity
        return None if v is None else float(np.hypot(v[0], v[1]))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_ball_overlay(
    frame: np.ndarray,
    predicted_pos: tuple[int, int] | None,
    detected_box: tuple | None,
    coasting: bool,
    trail: list[tuple[int, int]] | None = None,
    speed: float | None = None,
) -> np.ndarray:
    """
    Render the ball tracker state onto *frame* (in-place).

    Visual conventions
    ------------------
    - Green rectangle  : raw YOLO detection this frame
    - Blue filled dot  : Kalman-tracked position (detection available)
    - Orange filled dot: Kalman-coasted position (no detection this frame)
    - Fading cyan trail: recent position history
    """
    # Draw motion trail
    if trail:
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = (int(255 * alpha), int(200 * alpha), 0)  # fading cyan
            cv2.line(frame, trail[i - 1], trail[i], color, 2)

    # Draw raw YOLO bounding box
    if detected_box is not None:
        cx, cy, w, h = detected_box
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, "Ball", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    # Draw Kalman estimate
    if predicted_pos is not None:
        px, py = predicted_pos
        dot_color = (0, 140, 255) if coasting else (255, 80, 0)  # orange : blue
        label = f"Coasting ({speed:.0f}px/f)" if (coasting and speed) else "Tracked"
        cv2.circle(frame, (px, py), 9, dot_color, -1)
        cv2.circle(frame, (px, py), 9, (255, 255, 255), 1)  # white border
        cv2.putText(
            frame, label, (px + 12, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2,
        )

    return frame
