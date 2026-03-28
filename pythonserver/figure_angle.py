"""
figure_angle.py
---------------
Estimates the kick angle of each detected foosball figure.

Since the dataset only has axis-aligned bounding boxes (no rotation labels),
the angle is computed at runtime from the figure's ROI:

  1. Extract the bounding-box crop from the frame.
  2. Threshold to isolate the figure shape.
  3. Run PCA on the contour points → principal axis direction.
  4. Convert to angle-from-vertical (0° = upright, ±90° = kicked flat).
  5. Fall back to the bounding-box aspect-ratio heuristic if the contour
     is too small or the ROI is degenerate.

Angle convention
----------------
  0°     – figure is standing upright (rod not rotated)
  +90°   – figure is kicked forward / tilted right in image
  -90°   – figure is kicked backward / tilted left in image

  The sign follows image-x: positive = the figure's top leans to the right.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core angle estimator
# ---------------------------------------------------------------------------

def estimate_figure_angle(frame: np.ndarray, xyxy) -> float | None:
    """
    Estimate the kick angle of a single figure bounding box.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    xyxy : array-like of length 4
        Bounding box in pixel coords [x1, y1, x2, y2].

    Returns
    -------
    float | None
        Angle in degrees (see module docstring), or None if the ROI
        is too small to process.
    """
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

    # Clamp to frame
    h_f, w_f = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_f, x2); y2 = min(h_f, y2)

    bw = x2 - x1
    bh = y2 - y1

    if bw < 3 or bh < 3:
        return None

    # --- Aspect-ratio fallback (always available) ---
    # arctan(w/h): 0° when figure is tall/upright, 90° when wide/flat.
    aspect_angle = float(np.degrees(np.arctan2(bw, bh)))

    # --- ROI-based PCA estimate ---
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold – figures are typically darker than the table surface
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return aspect_angle

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return aspect_angle

    # PCA on contour point cloud → principal axis
    pts = largest.reshape(-1, 2).astype(np.float32)
    _, eigvecs = cv2.PCACompute(pts, mean=None)

    # eigvecs[0] is the direction of maximum variance (longest axis)
    vx, vy = float(eigvecs[0, 0]), float(eigvecs[0, 1])

    # Angle of that axis from the positive x-axis, in degrees
    axis_angle = np.degrees(np.arctan2(vy, vx))   # [-180, +180]

    # Convert to angle-from-vertical with sign:
    #   - project onto "tilt" convention: 0° = vertical, ±90° = horizontal
    #   - We take the acute angle between the principal axis and vertical (y-axis),
    #     then give it the sign of vx (positive = top leans right).
    angle_from_vertical = 90.0 - abs(axis_angle % 180.0 - 90.0)
    signed_angle = np.sign(vx) * angle_from_vertical

    # Sanity check: if contour is tiny relative to the box, trust aspect ratio more
    contour_area = cv2.contourArea(largest)
    box_area = bw * bh
    if contour_area / box_area < 0.05:
        return aspect_angle

    return float(signed_angle)


# ---------------------------------------------------------------------------
# Batch helper that operates on a full Ultralytics result
# ---------------------------------------------------------------------------

def estimate_angles_for_result(frame: np.ndarray, result, figure_class_id: int = 1):
    """
    Compute kick angles for every figure box in *result*.

    Returns
    -------
    list of (xyxy, angle | None)
        Parallel to the figure detections in result.boxes.
    """
    output = []
    for box in result.boxes:
        if int(box.cls[0]) != figure_class_id:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        angle = estimate_figure_angle(frame, xyxy)
        output.append((xyxy, angle))
    return output


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def draw_figure_angles(
    frame: np.ndarray,
    angle_data: list,
) -> np.ndarray:
    """
    Overlay kick-angle information on *frame* for each (xyxy, angle) entry.

    Visual language
    ---------------
    - A small oriented line through the figure centre shows tilt direction.
    - The angle label is colour-coded:
        green  → near upright (0°)
        yellow → mid-kick (~45°)
        red    → fully kicked (±90°)
    """
    for xyxy, angle in angle_data:
        if angle is None:
            continue

        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Colour gradient green → yellow → red based on |angle|
        t = min(abs(angle) / 90.0, 1.0)
        if t < 0.5:
            r = int(255 * 2 * t)
            g = 255
        else:
            r = 255
            g = int(255 * 2 * (1.0 - t))
        color = (0, g, r)   # BGR

        # Oriented tilt line (length proportional to how kicked)
        line_len = 14
        angle_rad = np.radians(angle)
        dx = int(line_len * np.sin(angle_rad))
        dy = int(line_len * np.cos(angle_rad))
        cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 2)

        # Angle label above the box
        label = f"{angle:+.0f}\u00b0"   # e.g. "+34°" or "-12°"
        cv2.putText(
            frame, label, (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return frame
