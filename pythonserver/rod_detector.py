"""
rod_detector.py
---------------
Detects the Y-positions of the 8 chrome/silver foosball rods using:

  1. HSV masking for silver/chrome colours (low saturation, high value).
  2. A horizontal morphological open to keep only wide horizontal runs
     (eliminates small silver highlights on figures, screws, etc.).
  3. Row-sum peak detection to find the Y coordinate of each rod.
  4. Gap-based clustering to collapse the few rows of each rod into a
     single representative Y value.

Returns a sorted list of rod Y-coordinates (top → bottom in image coords).

Tune SILVER_LO / SILVER_HI below if your rods look darker or more
bluish/golden under your specific lighting.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Colour range for chrome / silver rods
#   H : any hue (chrome is near-neutral)
#   S : 0–55  (very low saturation)
#   V : 120–255 (bright)
# Increase S_HI a little if rods appear slightly tinted under coloured lights.
# Decrease V_LO if they look darker.
# ──────────────────────────────────────────────────────────────────────────────

SILVER_LO = np.array([  0,  0, 100], dtype=np.uint8)
SILVER_HI = np.array([180, 70, 255], dtype=np.uint8)

# Fraction of frame width that a row must contain to be counted as a rod row.
# Lowered to 0.06 so partially-occluded rods (blocked by figures) are still found.
MIN_ROW_FILL = 0.06


# ──────────────────────────────────────────────────────────────────────────────
# Core detector
# ──────────────────────────────────────────────────────────────────────────────

def detect_rod_ys(
    frame: np.ndarray,
    num_rods: int = 8,
) -> list[int]:
    """
    Detect the Y-coordinates of up to *num_rods* horizontal rods.

    Parameters
    ----------
    frame    : BGR frame
    num_rods : expected number of rods (used to prune excess detections)

    Returns
    -------
    list[int]
        Sorted list of rod Y positions (image-row coordinates).
        May be shorter than num_rods if not all rods are visible.
    """
    fh, fw = frame.shape[:2]

    # ── Step 1: silver mask ───────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SILVER_LO, SILVER_HI)

    # ── Step 2: small horizontal dilation + open ──────────────────────────────
    # A single-iteration narrow dilation bridges small gaps where a figure
    # interrupts the rod, without creating wide false blobs.
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    mask     = cv2.dilate(mask, dilate_k, iterations=1)
    kernel_w = max(fw // 15, 15)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ── Step 3: row-sum histogram ─────────────────────────────────────────────
    row_sums = h_mask.sum(axis=1) / 255.0      # silver-pixel count per row

    # Lightly smooth to merge the few rows that make up each rod's cross-section
    smooth_k = max(fh // 400, 3)
    if smooth_k % 2 == 0:
        smooth_k += 1
    row_f    = row_sums.astype(np.float32).reshape(-1, 1)
    row_sums = cv2.GaussianBlur(row_f, (1, smooth_k), 0).flatten()

    # ── Step 4: non-maximum suppression to find rod peaks ─────────────────────
    # A rod shows up as a distinct peak in the row-sum histogram.
    # We find local maxima separated by at least min_sep rows.
    min_height = fw * MIN_ROW_FILL
    min_sep    = max(fh // (num_rods + 4), 20)   # minimum rows between two rods
    half_win   = min_sep // 2

    raw_peaks: list[int] = []
    for y in range(half_win, fh - half_win):
        val = float(row_sums[y])
        if val < min_height:
            continue
        window = row_sums[y - half_win: y + half_win + 1]
        if val == float(window.max()):
            raw_peaks.append(y)

    # Greedy deduplication: keep first peak in each min_sep window
    rod_ys: list[int] = []
    for y in raw_peaks:
        if not rod_ys or y - rod_ys[-1] >= min_sep:
            rod_ys.append(y)

    # ── Step 5: if over-detected keep the strongest num_rods ──────────────────
    if len(rod_ys) > num_rods:
        heights = [(y, float(row_sums[y])) for y in rod_ys]
        heights.sort(key=lambda x: x[1], reverse=True)
        rod_ys  = sorted(y for y, _ in heights[:num_rods])

    return rod_ys


# ──────────────────────────────────────────────────────────────────────────────
# Debug mask (optional – useful for tuning colour ranges)
# ──────────────────────────────────────────────────────────────────────────────

def silver_debug_mask(frame: np.ndarray) -> np.ndarray:
    """
    Return the binary silver mask (after the horizontal open) as a BGR image
    suitable for display / saving alongside the main annotated frame.
    """
    fh, fw   = frame.shape[:2]
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask     = cv2.inRange(hsv, SILVER_LO, SILVER_HI)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    mask     = cv2.dilate(mask, dilate_k, iterations=1)
    kernel_w = max(fw // 15, 15)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(h_mask, cv2.COLOR_GRAY2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helper
# ──────────────────────────────────────────────────────────────────────────────

def draw_rod_lines(
    frame: np.ndarray,
    rod_ys: list[int],
    color: tuple[int, int, int] = (180, 180, 180),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a faint horizontal line across the frame for each detected rod Y.
    Call this *before* draw_formation_overlay so the formation overlay
    renders on top.
    """
    fw = frame.shape[1]
    for y in rod_ys:
        cv2.line(frame, (0, y), (fw, y), color, thickness, cv2.LINE_AA)
    return frame
