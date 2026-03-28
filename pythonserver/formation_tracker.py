"""
formation_tracker.py
--------------------
Assigns detected foosball figures to the eight known rods and classifies
each rod as belonging to the player (yellow) or opponent (blue) team.

Known formation – rod order left → right (player's perspective):
  Rod 0  Player  Goalie    (1 figure)
  Rod 1  Player  Attack    (3 figures)
  Rod 2  Opp.    Defense   (3 figures)
  Rod 3  Player  Midfield  (4 figures)
  Rod 4  Opp.    Midfield  (4 figures)
  Rod 5  Player  Defense   (3 figures)
  Rod 6  Opp.    Attack    (3 figures)
  Rod 7  Opp.    Goalie    (1 figure)

Classification strategy (two independent signals):
  1. Colour: HSV mask on each figure ROI → 'player' / 'opponent' / 'unknown'
  2. Formation position: x-clustering into 8 rods → expected team from the
     table above.  This is used as the fallback when colour is uncertain.

Tune YELLOW_LO/HI and BLUE_LO/HI below to match your specific table.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# Formation definition
# ──────────────────────────────────────────────────────────────────────────────

NUM_RODS = 8

# (short_label, team, expected_figure_count)
# Rod order from player's goal → opponent's goal.
# Each team's rods interleave: player goes Goalie→Def→Mid→Atk,
# opponent goes Atk→Mid→Def→Goalie (mirrored, facing the player).
FORMATION: list[tuple[str, str, int]] = [
    ("P-Goal", "player",   1),   # Rod 0 – player's goalie
    ("O-Atk",  "opponent", 3),   # Rod 1 – opponent's attack  (in player's zone)
    ("P-Def",  "player",   3),   # Rod 2 – player's defense
    ("O-Mid",  "opponent", 4),   # Rod 3 – opponent's midfield
    ("P-Mid",  "player",   4),   # Rod 4 – player's midfield
    ("O-Def",  "opponent", 3),   # Rod 5 – opponent's defense
    ("P-Atk",  "player",   3),   # Rod 6 – player's attack    (in opponent's zone)
    ("O-Goal", "opponent", 1),   # Rod 7 – opponent's goalie
]

# BGR display colours
TEAM_BGR: dict[str, tuple[int, int, int]] = {
    "player":   (0,   210, 255),   # yellow in BGR
    "opponent": (255,  80,  30),   # blue in BGR
    "unknown":  (150, 150, 150),   # grey
}

# ──────────────────────────────────────────────────────────────────────────────
# HSV colour ranges  ← tune these to match your actual table colours
#
# Player  team: yellow figures + white goalie
# Opponent team: red figures   + blue goalie
# ──────────────────────────────────────────────────────────────────────────────

# Yellow figures  (H 15–40)
YELLOW_LO = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HI = np.array([40, 255, 255], dtype=np.uint8)

# White figures (player goalie) – very low saturation, high brightness
WHITE_LO = np.array([  0,   0, 180], dtype=np.uint8)
WHITE_HI = np.array([179,  45, 255], dtype=np.uint8)

# Red figures (H wraps around 0/179 in OpenCV HSV)
RED_LO1 = np.array([  0, 100,  60], dtype=np.uint8)
RED_HI1 = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LO2 = np.array([160, 100,  60], dtype=np.uint8)
RED_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# Blue figures (opponent goalie)  (H 90–135)
BLUE_LO = np.array([ 90,  60,  40], dtype=np.uint8)
BLUE_HI = np.array([135, 255, 255], dtype=np.uint8)

# Fraction of ROI pixels that must match a team's colour(s) to be conclusive,
# AND the winning team must score at least COLOR_DOMINANCE × the other team.
MIN_COLOR_RATIO = 0.08
COLOR_DOMINANCE = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FigureInfo:
    xyxy:       np.ndarray
    cx:         float
    cy:         float
    color_team: str         # 'player', 'opponent', or 'unknown'
    angle:      float | None   # kick angle in degrees (from figure_angle.py)


@dataclass
class RodInfo:
    rod_idx:        int
    label:          str
    expected_team:  str
    expected_count: int
    figures:        list[FigureInfo] = field(default_factory=list)

    @property
    def assigned_team(self) -> str:
        """
        Team determined by colour-majority vote across this rod's figures.
        Falls back to the formation's expected team if colour is inconclusive.
        """
        if not self.figures:
            return self.expected_team
        votes = {"player": 0, "opponent": 0}
        for f in self.figures:
            if f.color_team in votes:
                votes[f.color_team] += 1
        if votes["player"] > votes["opponent"]:
            return "player"
        if votes["opponent"] > votes["player"]:
            return "opponent"
        # Tie or all-unknown → trust formation position
        return self.expected_team

    @property
    def rod_angle(self) -> float | None:
        """
        Consensus kick angle (degrees) for this rod.

        All figures on a physical rod rotate together, so we take the median
        of all detected figure angles.  Returns None when no figure on this
        rod has an angle estimate.
        """
        angles = [f.angle for f in self.figures if f.angle is not None]
        if not angles:
            return None
        return float(np.median(angles))


def unify_rod_angles(rods: list["RodInfo"]) -> None:
    """
    Replace each figure's individual angle with the rod-level consensus angle
    (median of all detected angles on that rod).

    Call this after assign_figures_to_rods() so that every figure on the same
    physical rod reports the same rotation in the obs vector and overlays.
    Figures on rods with no angle data keep angle=None.
    """
    for rod in rods:
        consensus = rod.rod_angle
        if consensus is not None:
            for f in rod.figures:
                f.angle = consensus


# ──────────────────────────────────────────────────────────────────────────────
# Colour classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify_figure_color(frame: np.ndarray, xyxy) -> str:
    """
    Classify a figure bounding box as 'player' or 'opponent' using HSV masking.

    Player  team: yellow figures OR white goalie.
    Opponent team: red figures   OR blue goalie.
    """
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    total = roi.shape[0] * roi.shape[1]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Player: yellow or white
    player_px = (
        cv2.countNonZero(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI))
        + cv2.countNonZero(cv2.inRange(hsv, WHITE_LO,  WHITE_HI))
    )

    # Opponent: red (wraps around hue) or blue
    opponent_px = (
        cv2.countNonZero(cv2.inRange(hsv, RED_LO1, RED_HI1))
        + cv2.countNonZero(cv2.inRange(hsv, RED_LO2, RED_HI2))
        + cv2.countNonZero(cv2.inRange(hsv, BLUE_LO,  BLUE_HI))
    )

    p_ratio = player_px   / total
    o_ratio = opponent_px / total

    if p_ratio >= MIN_COLOR_RATIO and p_ratio > o_ratio * COLOR_DOMINANCE:
        return "player"
    if o_ratio >= MIN_COLOR_RATIO and o_ratio > p_ratio * COLOR_DOMINANCE:
        return "opponent"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Build figure list from one YOLO result frame
# ──────────────────────────────────────────────────────────────────────────────

def build_figure_list(
    frame:          np.ndarray,
    result,                         # ultralytics Results object
    angle_data:     list,           # [(xyxy, angle|None), ...]  from figure_angle.py
    figure_class_id: int = 1,
) -> list[FigureInfo]:
    """
    Merge YOLO detections, colour classification, and pre-computed kick angles
    into a flat list of FigureInfo objects.
    """
    # Build angle lookup keyed on box top-left pixel (close enough for dedup)
    angle_map: dict[tuple[int, int], float | None] = {}
    for xyxy, ang in angle_data:
        key = (round(float(xyxy[0])), round(float(xyxy[1])))
        angle_map[key] = ang

    figures: list[FigureInfo] = []
    for box in result.boxes:
        if int(box.cls[0]) != figure_class_id:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        color = classify_figure_color(frame, xyxy)
        angle = angle_map.get((round(float(xyxy[0])), round(float(xyxy[1]))))
        figures.append(FigureInfo(xyxy=xyxy, cx=cx, cy=cy,
                                   color_team=color, angle=angle))
    return figures


# ──────────────────────────────────────────────────────────────────────────────
# Rod assignment  (y-clustering → formation mapping)
#
# Rods run HORIZONTALLY across the frame (goals at top / bottom).
# Different rods sit at different Y positions; figures on the same rod
# share a similar Y and vary in X.
# ──────────────────────────────────────────────────────────────────────────────

def assign_figures_to_rods(
    figures: list[FigureInfo],
    frame_height: int,
    rod_ys: list[int] | None = None,
) -> list[RodInfo]:
    """
    Assign figures to formation rods.

    If *rod_ys* is provided (from rod_detector.detect_rod_ys) and contains
    at least 2 entries, each figure is snapped to its nearest detected rod Y
    and that rod is mapped to the closest unoccupied formation slot.

    Otherwise falls back to adaptive Y-gap clustering on the figure centroids.

    Returns exactly NUM_RODS RodInfo objects in formation order.
    Empty rods (no figures detected) keep their formation metadata.
    """
    rods = [
        RodInfo(
            rod_idx=i,
            label=FORMATION[i][0],
            expected_team=FORMATION[i][1],
            expected_count=FORMATION[i][2],
        )
        for i in range(NUM_RODS)
    ]

    if not figures:
        return rods

    # ── Path A: exact formation split ────────────────────────────────────────
    # When the total number of detected figures exactly matches the expected
    # formation count we can skip all clustering: sort by Y and slice into
    # groups using the known per-rod counts.  This is the most accurate path.
    total_expected = sum(f[2] for f in FORMATION)
    if len(figures) == total_expected:
        sorted_figs = sorted(figures, key=lambda f: f.cy)
        idx = 0
        for rod in rods:
            rod.figures = sorted_figs[idx : idx + rod.expected_count]
            idx += rod.expected_count
        return rods

    # ── Path C: rod_ys provided by the rod detector ───────────────────────────
    # Only trust rod_ys when we found the majority of rods; otherwise the
    # snap-to-nearest logic causes figures from adjacent rods to pile up on
    # whichever detected rod is closest, making the formation readout worse
    # than the pure figure-clustering fallback.
    if rod_ys and len(rod_ys) >= max(NUM_RODS - 2, 4):
        sorted_ys = sorted(rod_ys)
        y_min, y_max = sorted_ys[0], sorted_ys[-1]
        y_span = max(y_max - y_min, 1.0)

        # Map each detected rod_y to the nearest unoccupied formation slot
        ry_to_slot: dict[int, int] = {}
        assigned_slots: set[int] = set()
        for ry in sorted_ys:
            frac = (ry - y_min) / y_span
            raw  = frac * (NUM_RODS - 1)
            for slot in sorted(range(NUM_RODS), key=lambda s: abs(s - raw)):
                if slot not in assigned_slots:
                    ry_to_slot[ry] = slot
                    assigned_slots.add(slot)
                    break

        # Assign each figure to its nearest detected rod
        for fig in figures:
            nearest_ry = min(sorted_ys, key=lambda ry: abs(fig.cy - ry))
            slot = ry_to_slot.get(nearest_ry)
            if slot is not None:
                rods[slot].figures.append(fig)

        return rods

    # ── Path D: fallback — cluster figure centroids by Y ─────────────────────
    sorted_figs = sorted(figures, key=lambda f: f.cy)

    y_range = sorted_figs[-1].cy - sorted_figs[0].cy
    # Divide by NUM_RODS*3 (was 1.5) so adjacent rods with similar Y values
    # are not merged into one cluster.
    gap_threshold = max(frame_height * 0.015, y_range / (NUM_RODS * 3))

    clusters: list[list[FigureInfo]] = []
    cur: list[FigureInfo] = [sorted_figs[0]]
    for fig in sorted_figs[1:]:
        if fig.cy - cur[-1].cy > gap_threshold:
            clusters.append(cur)
            cur = [fig]
        else:
            cur.append(fig)
    clusters.append(cur)

    if len(clusters) == 1:
        cy = float(np.mean([f.cy for f in clusters[0]]))
        rel = cy / max(frame_height, 1)
        rod_idx = max(0, min(NUM_RODS - 1, round(rel * (NUM_RODS - 1))))
        rods[rod_idx].figures = clusters[0]
        return rods

    cluster_ys = [float(np.mean([f.cy for f in c])) for c in clusters]
    y_min, y_max = cluster_ys[0], cluster_ys[-1]
    y_span = max(y_max - y_min, 1.0)

    assigned: set[int] = set()
    for cluster, cy in zip(clusters, cluster_ys):
        frac = (cy - y_min) / y_span
        raw  = frac * (NUM_RODS - 1)
        for rod_idx in sorted(range(NUM_RODS), key=lambda r: abs(r - raw)):
            if rod_idx not in assigned:
                rods[rod_idx].figures = cluster
                assigned.add(rod_idx)
                break

    return rods


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_rod_slope(rods: list[RodInfo]) -> tuple[float, float]:
    """
    Compute the single global rod slope (dy/dx) and tilt angle in degrees.

    Pools within-rod (dx, dy) deviations from every rod with ≥ 2 figures and
    solves a least-squares slope – rods with more figures carry more weight.

    Returns
    -------
    slope : float   dy/dx  (positive = lines tilt down to the right)
    angle : float   degrees from horizontal  (arctan of slope)
    """
    all_dx: list[float] = []
    all_dy: list[float] = []
    for rod in rods:
        if len(rod.figures) < 2:
            continue
        cx_mean = float(np.mean([f.cx for f in rod.figures]))
        cy_mean = float(np.mean([f.cy for f in rod.figures]))
        for f in rod.figures:
            all_dx.append(f.cx - cx_mean)
            all_dy.append(f.cy - cy_mean)

    if len(all_dx) >= 4:
        adx = np.array(all_dx, dtype=np.float64)
        ady = np.array(all_dy, dtype=np.float64)
        denom = float(np.dot(adx, adx))
        slope = float(np.dot(adx, ady) / denom) if denom > 1e-6 else 0.0
    else:
        # Fallback: median of individual per-rod slopes
        slopes: list[float] = []
        for rod in rods:
            if len(rod.figures) >= 2:
                xs = np.array([f.cx for f in rod.figures], dtype=np.float32)
                ys = np.array([f.cy for f in rod.figures], dtype=np.float32)
                if xs.max() - xs.min() > 5:
                    slopes.append(float(np.polyfit(xs, ys, 1)[0]))
        slope = float(np.median(slopes)) if slopes else 0.0

    angle = float(np.degrees(np.arctan(slope)))
    return slope, angle


def draw_formation_overlay(frame: np.ndarray, rods: list[RodInfo]) -> np.ndarray:
    """
    Draw:
      • A colour-coded team dot on every figure.
      • A thin HORIZONTAL rod-guide line + label to the left of each active rod.
      • A compact formation summary bar down the RIGHT edge of the frame
        (one horizontal slot per rod, stacked top → bottom).
    """
    fh, fw = frame.shape[:2]
    global_slope, _ = compute_rod_slope(rods)

    # ── Per-rod annotations ───────────────────────────────────────────────────
    for rod in rods:
        if not rod.figures:
            continue

        team  = rod.assigned_team
        color = TEAM_BGR[team]

        # Centre of the rod (mean of figure centroids)
        cx_mean = float(np.mean([f.cx for f in rod.figures]))
        cy_mean = float(np.mean([f.cy for f in rod.figures]))
        left_x  = int(min(f.xyxy[0] for f in rod.figures))
        right_x = int(max(f.xyxy[2] for f in rod.figures))

        # Guide line following the actual rod tilt (y = cy_mean + slope*(x - cx_mean))
        x0 = max(left_x - 18, 0)
        x1 = right_x
        y0 = int(round(cy_mean + global_slope * (x0 - cx_mean)))
        y1 = int(round(cy_mean + global_slope * (x1 - cx_mean)))
        cv2.line(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        # Rod label to the left of the leftmost figure
        label_x  = max(left_x - 55, 2)
        label_cy = int(round(cy_mean + global_slope * (label_x - cx_mean)))
        cv2.putText(frame, rod.label,
                    (label_x, label_cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{len(rod.figures)}/{rod.expected_count}",
                    (label_x, label_cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

        # Colour dot on each figure centroid
        for fig in rod.figures:
            dot = TEAM_BGR.get(fig.color_team, TEAM_BGR["unknown"])
            fcx, fcy = int(fig.cx), int(fig.cy)
            cv2.circle(frame, (fcx, fcy), 5, dot, -1)
            cv2.circle(frame, (fcx, fcy), 5, (255, 255, 255), 1)

    # ── Formation summary bar (right edge, stacked vertically) ───────────────
    bar_w  = 58
    bar_x0 = fw - bar_w
    slot_h = fh // NUM_RODS

    # Dark background strip
    cv2.rectangle(frame, (bar_x0, 0), (fw, fh), (18, 18, 18), -1)

    for rod in rods:
        sy    = rod.rod_idx * slot_h
        ey    = sy + slot_h
        team  = rod.assigned_team
        color = TEAM_BGR[team]

        # Faint tinted background per slot
        sub  = frame[sy:ey, bar_x0:fw]
        tint = np.full_like(sub, color)
        frame[sy:ey, bar_x0:fw] = cv2.addWeighted(sub, 0.75, tint, 0.25, 0)

        # Slot divider
        cv2.line(frame, (bar_x0, sy), (fw, sy), (60, 60, 60), 1)

        # Label line 1: rod name
        cv2.putText(frame, rod.label,
                    (bar_x0 + 3, sy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

        # Label line 2: detected / expected
        count_color = (100, 255, 100) if len(rod.figures) == rod.expected_count else (180, 180, 180)
        cv2.putText(frame, f"{len(rod.figures)}/{rod.expected_count}",
                    (bar_x0 + 3, sy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, count_color, 1, cv2.LINE_AA)

    return frame
