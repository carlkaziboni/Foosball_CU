"""
obs_builder.py
--------------
Converts per-frame CV detections into the 38-dimensional observation vector
expected by FoosballEnv (full_information_protagonist_antagonist_gym.py).

Coordinate conventions (gym / MuJoCo space)
--------------------------------------------
  ball_x    world X, centred at table mid-width  (±24.25 cm; walls at ±WALL_X)
  ball_y    qpos Y  =  world_y + 4
              (ball body sits at y=−4 in world, so qpos 4 = field centre)
              player goal  →  world_y ≈ −40.5  →  obs_y ≈ −36.5
              opponent goal → world_y ≈ +40.5  →  obs_y ≈ +44.5

  rod_slide  world X of the rod centre-of-mass  (same axis as ball_x, ±~9 cm)
  rod_rot    rotation angle in radians  [−π, π]

Obs layout (38 values)
-----------------------
  [0:3]   ball_pos   (obs_x, obs_y, 0)
  [3:6]   ball_vel   (vx, vy, 0)            per-frame finite-difference
  [6:14]  rod_slide_positions   [y_goal, y_def, y_mid, y_atk, b_goal, b_def, b_mid, b_atk]
  [14:22] rod_slide_velocities  same order, per-frame finite-difference
  [22:30] rod_rotate_positions  same order, radians
  [30:38] rod_rotate_velocities same order, per-frame finite-difference

Formation-rod → gym obs index mapping
---------------------------------------
  formation rod 0 (P-Goal) → obs block index 0  (y_goal)
  formation rod 2 (P-Def)  → obs block index 1  (y_def)
  formation rod 4 (P-Mid)  → obs block index 2  (y_mid)
  formation rod 6 (P-Atk)  → obs block index 3  (y_attack)
  formation rod 7 (O-Goal) → obs block index 4  (b_goal)
  formation rod 5 (O-Def)  → obs block index 5  (b_def)
  formation rod 3 (O-Mid)  → obs block index 6  (b_mid)
  formation rod 1 (O-Atk)  → obs block index 7  (b_attack)
"""

import cv2
import numpy as np

# ── Gym / MuJoCo coordinate constants ────────────────────────────────────────
BALL_BODY_Y_OFFSET = -4.0    # ball body default world-Y; qpos_y = world_y − offset
TABLE_HALF_WIDTH   = 24.25   # table width  48.5 cm  →  gym X in [−24.25, +24.25]
TABLE_HALF_LENGTH  = 40.5    # table length 81 cm    →  world-Y in [−40.5, +40.5]

# formation_rod_idx → flat index in each 8-element rod block of the obs vector
_FORM_TO_GYM_IDX: dict[int, int] = {
    0: 0,   # P-Goal → y_goal
    2: 1,   # P-Def  → y_def
    4: 2,   # P-Mid  → y_mid
    6: 3,   # P-Atk  → y_attack
    7: 4,   # O-Goal → b_goal
    5: 5,   # O-Def  → b_def
    3: 6,   # O-Mid  → b_mid
    1: 7,   # O-Atk  → b_attack
}


class ObsBuilder:
    """
    Stateful per-frame converter.

    Usage
    -----
        builder = ObsBuilder()
        ...
        obs = builder.update(ball_tracker, rods, H)
        if obs is not None:
            action, _ = model.predict(obs)
    """

    def __init__(self):
        self._prev_slides:   np.ndarray | None = None
        self._prev_rots:     np.ndarray | None = None
        self._prev_ball_obs: tuple[float, float] | None = None
        self.obs: np.ndarray | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        ball_tracker,           # KalmanBallTracker instance
        rods,                   # list[RodInfo] from formation_tracker
        H: np.ndarray | None,   # physical(length,width)→image homography
    ) -> np.ndarray | None:
        """
        Build and return the 38-dim observation, or None until a homography
        is available (requires ≥ 4 well-detected rods to be established).
        """
        if H is None:
            return None

        H_inv = np.linalg.inv(H)

        # ── Ball ──────────────────────────────────────────────────────────────
        ball_obs = self._ball_to_obs(ball_tracker, H_inv)
        if ball_obs is None:
            # Field centre in obs coordinates
            ball_obs = (0.0, -BALL_BODY_Y_OFFSET)  # (0, 4)

        if self._prev_ball_obs is not None:
            ball_vx = ball_obs[0] - self._prev_ball_obs[0]
            ball_vy = ball_obs[1] - self._prev_ball_obs[1]
        else:
            ball_vx, ball_vy = 0.0, 0.0
        self._prev_ball_obs = ball_obs

        # ── Rods ──────────────────────────────────────────────────────────────
        slides = self._rod_slides(rods, H_inv)
        rots   = self._rod_rotations(rods)

        slide_vels = (slides - self._prev_slides
                      if self._prev_slides is not None
                      else np.zeros(8, dtype=np.float32))
        rot_vels   = (rots   - self._prev_rots
                      if self._prev_rots   is not None
                      else np.zeros(8, dtype=np.float32))

        self._prev_slides = slides.copy()
        self._prev_rots   = rots.copy()

        # ── Assemble ──────────────────────────────────────────────────────────
        self.obs = np.array([
            ball_obs[0], ball_obs[1], 0.0,
            ball_vx,     ball_vy,     0.0,
            *slides,
            *slide_vels,
            *rots,
            *rot_vels,
        ], dtype=np.float32)

        return self.obs

    def antagonist_obs(self) -> np.ndarray | None:
        """
        Return the blue (antagonist) perspective observation derived from the
        last yellow obs.  Returns None if update() hasn't produced one yet.

        Transformation (mirrors _get_antagonist_obs in the gym):
          - Negate ball Y position and velocity
          - Swap the yellow 4-rod block (indices 0-3) with blue (indices 4-7)
            in all four rod groups (slide pos, slide vel, rot pos, rot vel).
        """
        if self.obs is None:
            return None

        obs = self.obs.copy()
        obs[1] = -obs[1]   # ball_pos y
        obs[4] = -obs[4]   # ball_vel vy

        for base in (6, 14, 22, 30):
            y_block = obs[base:base + 4].copy()
            b_block = obs[base + 4:base + 8].copy()
            obs[base:base + 4]     = b_block
            obs[base + 4:base + 8] = y_block

        return obs

    def reset(self) -> None:
        """Call at the start of each episode to clear velocity history."""
        self._prev_slides    = None
        self._prev_rots      = None
        self._prev_ball_obs  = None
        self.obs             = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ball_to_obs(
        self,
        ball_tracker,
        H_inv: np.ndarray,
    ) -> tuple[float, float] | None:
        """
        Convert the Kalman-tracked ball pixel position to (obs_x, obs_y).

        physical_length  0 → 81 cm  maps to  world_y  −40.5 → +40.5
        physical_width   0 → 48.5 cm maps to  world_x  −24.25 → +24.25
        obs_y = world_y − BALL_BODY_Y_OFFSET = world_y + 4
        """
        if not getattr(ball_tracker, '_initialized', False):
            return None

        trail = getattr(ball_tracker, 'trail', [])
        if not trail:
            return None

        px, py = trail[-1]
        pts  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        phys = cv2.perspectiveTransform(pts, H_inv)

        phys_length = float(phys[0][0][0])   # along table length  (0 → 81 cm)
        phys_width  = float(phys[0][0][1])   # across table width  (0 → 48.5 cm)

        world_y = phys_length - TABLE_HALF_LENGTH          # centred: −40.5 → +40.5
        obs_y   = world_y - BALL_BODY_Y_OFFSET             # qpos_y  = world_y + 4
        obs_x   = phys_width - TABLE_HALF_WIDTH            # centred: −24.25 → +24.25

        return float(obs_x), float(obs_y)

    def _rod_slides(self, rods, H_inv: np.ndarray) -> np.ndarray:
        """
        Rod slide = world X of the rod's centre-of-mass in gym space
                  = physical_width − 24.25  (range ≈ ±9 cm).

        Figures can slide along the rod (varying X in image), so the mean
        figure centroid X is a reasonable proxy for the rod's slide position.
        Rods with no detected figures default to 0 (rod centred).
        """
        slides = np.zeros(8, dtype=np.float32)

        for rod in rods:
            gym_idx = _FORM_TO_GYM_IDX.get(rod.rod_idx)
            if gym_idx is None or not rod.figures:
                continue

            mean_cx = float(np.mean([f.cx for f in rod.figures]))
            mean_cy = float(np.mean([f.cy for f in rod.figures]))
            pts  = np.array([[[mean_cx, mean_cy]]], dtype=np.float32)
            phys = cv2.perspectiveTransform(pts, H_inv)

            phys_width = float(phys[0][0][1])            # width is dim [1]
            slides[gym_idx] = phys_width - TABLE_HALF_WIDTH

        return slides

    @staticmethod
    def expected_rots() -> np.ndarray:
        """
        Expected / fallback rod rotations: all 0.0 (figures upright / neutral).
        Used as a reference for rods with no detected figures — mirrors the
        ghost-position overlay which also places figures at their expected spots.
        """
        return np.zeros(8, dtype=np.float32)

    def _rod_rotations(self, rods) -> np.ndarray:
        """
        Rod rotation for each of the 8 gym rods, in radians.

        Uses rod.rod_angle — the per-rod consensus kick angle (median of all
        detected figure angles on that rod).  Because all figures on the same
        physical rod share one axle, they rotate identically; the consensus
        angle is both more accurate and physically correct.

        Rods with no angle data fall back to 0.0 (upright / neutral).
        """
        rots = self.expected_rots()   # initialise to the expected (neutral) value

        for rod in rods:
            gym_idx = _FORM_TO_GYM_IDX.get(rod.rod_idx)
            if gym_idx is None:
                continue

            consensus_deg = rod.rod_angle   # median over rod's figures; None if unknown
            if consensus_deg is not None:
                rots[gym_idx] = float(np.radians(consensus_deg))

        return rots
