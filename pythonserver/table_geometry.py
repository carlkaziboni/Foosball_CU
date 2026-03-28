"""
table_geometry.py
-----------------
Physical geometry of the foosball table.

Provides:
  • PHYS_POSITIONS  – expected (length_cm, width_cm) for all 22 figures.
  • build_homography(rods) – fits a physical→image homography from rods
    where the detected figure count matches the expected count.
  • project_all_positions(H) – maps all 22 expected positions to pixels.
  • draw_ghost_overlay(frame, expected_img, rods) – draws faint rings at
    every expected position BEFORE the solid detection dots are drawn,
    so rings show through only where a figure was missed.

Physical coordinate system
--------------------------
  length_cm  – distance along the table from the PLAYER'S goal wall (0 cm)
               to the OPPONENT'S goal wall (81 cm).  Maps roughly to image Y.
  width_cm   – distance across the table from one side wall (0 cm) to the
               other (48.5 cm).  Maps roughly to image X.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Physical measurements (cm)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_LENGTH = 81.0     # goal wall → goal wall
TABLE_WIDTH  = 48.5     # side wall → side wall

BAR_TO_WALL  = 7.5      # goal wall → nearest rod centre
BAR_GAP      = 9.5      # rod centre → rod centre

SPACING_3BAR = 15.0     # figure spacing on 1- and 3-figure rods
SPACING_4BAR = 10.2     # figure spacing on 4-figure rods

# Must match FORMATION order in formation_tracker.py
ROD_COUNTS = [1, 3, 3, 4, 4, 3, 3, 1]

# Rod centres along table length (cm from player's goal wall)
ROD_LENGTHS: list[float] = [BAR_TO_WALL + i * BAR_GAP for i in range(8)]

_CW = TABLE_WIDTH / 2.0  # table width centre = 24.25 cm


def _figure_widths(n: int) -> list[float]:
    """
    Physical width positions (cm) of n evenly spaced figures, centred on rod.
    Goalie (n=1) sits at centre; 3-bar uses SPACING_3BAR; 4-bar uses SPACING_4BAR.
    """
    spacing = SPACING_3BAR if n <= 3 else SPACING_4BAR
    half = (n - 1) * spacing / 2.0
    return [_CW - half + i * spacing for i in range(n)]


# PHYS_POSITIONS[rod_idx][fig_idx] = (length_cm, width_cm)
PHYS_POSITIONS: list[list[tuple[float, float]]] = [
    [(ROD_LENGTHS[r], w) for w in _figure_widths(ROD_COUNTS[r])]
    for r in range(8)
]


# ─────────────────────────────────────────────────────────────────────────────
# Homography estimation
# ─────────────────────────────────────────────────────────────────────────────

def build_homography(rods) -> np.ndarray | None:
    """
    Estimate a physical(length, width) → image(pixel_x, pixel_y) homography.

    Only uses rods where detected figure count == expected count so that
    the physical↔image correspondences are unambiguous.

    Within each qualifying rod, detected figures are sorted left→right by
    image X and matched to expected physical positions sorted by width_cm.

    Returns None when fewer than 4 correspondences are available.
    """
    src: list[list[float]] = []   # physical (length, width)
    dst: list[list[float]] = []   # image    (cx, cy)

    for rod in rods:
        if len(rod.figures) != rod.expected_count:
            continue

        det   = sorted(rod.figures, key=lambda f: f.cx)
        phys  = sorted(PHYS_POSITIONS[rod.rod_idx], key=lambda p: p[1])

        for fig, (pl, pw) in zip(det, phys):
            src.append([pl, pw])
            dst.append([float(fig.cx), float(fig.cy)])

    if len(src) < 4:
        return None

    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        cv2.RANSAC,
        8.0,
    )
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Position projection
# ─────────────────────────────────────────────────────────────────────────────

def project_all_positions(H: np.ndarray) -> list[list[tuple[float, float]]]:
    """
    Project all 22 expected physical positions through homography H.
    Returns a list mirroring PHYS_POSITIONS: [rod_idx][fig_idx] = (px_x, px_y).
    """
    result: list[list[tuple[float, float]]] = []
    for rod_positions in PHYS_POSITIONS:
        pts  = np.array(rod_positions, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H)
        result.append([(float(p[0][0]), float(p[0][1])) for p in proj])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formation refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_formation_by_geometry(figures, H) -> list:
    """
    Re-assign ALL detected figures to rods using the homography.

    Each figure snaps to whichever rod's projected centre Y is closest to the
    figure's cy.  This is the "majority" approach: it maximises correct rod
    assignments regardless of how many figures are currently detected.

    Works for both full (22/22) and partial detection sets.
    Replaces whatever the initial gap-clustering / snap-to-rod gave.
    """
    from formation_tracker import RodInfo, FORMATION, NUM_RODS

    expected_img = project_all_positions(H)

    # Projected centre Y of each rod (mean over its expected figure positions)
    rod_cy = [
        float(np.mean([p[1] for p in rod_pos]))
        for rod_pos in expected_img
    ]

    rod_fig_lists: dict[int, list] = {i: [] for i in range(NUM_RODS)}
    for fig in figures:
        best = min(range(NUM_RODS), key=lambda i: abs(fig.cy - rod_cy[i]))
        rod_fig_lists[best].append(fig)

    return [
        RodInfo(
            rod_idx=i,
            label=FORMATION[i][0],
            expected_team=FORMATION[i][1],
            expected_count=FORMATION[i][2],
            figures=rod_fig_lists[i],
        )
        for i in range(NUM_RODS)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-rod summary  (label → expected positions → angle)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RodSummary:
    label:              str
    expected_positions: list[tuple[float, float]]   # pixel (x, y) per figure slot
    angle_deg:          float | None                # consensus kick angle; None = unknown


def get_formation_data(
    rods,
    H: np.ndarray | None,
) -> list[RodSummary]:
    """
    Return one RodSummary per rod in formation order (rod 0 → rod 7).

    Each entry carries:
      label              – rod name, e.g. "P-Goal"
      expected_positions – projected pixel (x, y) for every expected figure slot
                           (empty list when no homography is available)
      angle_deg          – per-rod consensus kick angle in degrees (None if unknown)

    This is the canonical data format for feeding rod info into the RL app:
      for s in get_formation_data(rods, H):
          print(s.label, s.expected_positions, s.angle_deg)
    """
    expected_img = project_all_positions(H) if H is not None else [[] for _ in range(8)]

    summaries: list[RodSummary] = []
    for rod in rods:
        summaries.append(RodSummary(
            label=rod.label,
            expected_positions=expected_img[rod.rod_idx],
            angle_deg=rod.rod_angle,
        ))
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Ghost overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_ghost_overlay(
    frame: np.ndarray,
    expected_img: list[list[tuple[float, float]]],
    rods,
) -> np.ndarray:
    """
    Draw faint hollow rings at every expected figure position.

    Call this BEFORE draw_formation_overlay so the solid detected-figure dots
    (drawn by formation_tracker) cover the rings where detection succeeded.
    Rings that remain visible indicate missed detections.

    Ring colour matches the rod's expected team colour (faint version).
    A small cross-hair is added so the centre is visible at any zoom level.
    """
    from formation_tracker import TEAM_BGR

    fh, fw = frame.shape[:2]
    # Scale ring size proportionally to image height
    r_outer = max(fh // 180, 8)
    r_inner = max(r_outer - 4, 3)

    for rod in rods:
        base_color = TEAM_BGR.get(rod.expected_team, (120, 120, 120))
        # Dim the colour to 40 % brightness so rings don't dominate the view
        ghost_color = tuple(int(c * 0.4) for c in base_color)

        for exp_x, exp_y in expected_img[rod.rod_idx]:
            ix, iy = int(round(exp_x)), int(round(exp_y))
            if not (0 <= ix < fw and 0 <= iy < fh):
                continue

            # Outer ring
            cv2.circle(frame, (ix, iy), r_outer, ghost_color, 1, cv2.LINE_AA)
            # Small cross-hair centre
            ch = r_inner // 2
            cv2.line(frame, (ix - ch, iy), (ix + ch, iy), ghost_color, 1, cv2.LINE_AA)
            cv2.line(frame, (ix, iy - ch), (ix, iy + ch), ghost_color, 1, cv2.LINE_AA)

    return frame
