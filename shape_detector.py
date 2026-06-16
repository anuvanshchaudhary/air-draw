"""
shape_detector.py — Geometric shape snapping for completed strokes.

Analyzes a stroke's point cloud and attempts to snap it to a recognized shape
(circle or rectangle) if the confidence is above a threshold.
"""

import math
import cv2
import numpy as np


class ShapeDetector:
    """Detects circles and rectangles from stroke point data."""

    def __init__(self, circularity_threshold=0.75, min_points=15):
        """
        Args:
            circularity_threshold: Minimum circularity score (0-1) to snap to a circle.
            min_points: Minimum number of points in a stroke to attempt shape detection.
        """
        self.circularity_threshold = circularity_threshold
        self.min_points = min_points

    def detect(self, points):
        """
        Analyze a list of (x, y) points and determine if they form a recognizable shape.

        Args:
            points: List of (x, y) tuples representing the stroke.

        Returns:
            A dict with shape info, or None if no shape detected.
            For circle:  {"type": "circle", "center": (cx, cy), "radius": r}
            For rect:    {"type": "rectangle", "pt1": (x1, y1), "pt2": (x2, y2)}
            For line:    {"type": "line", "pt1": (x1, y1), "pt2": (x2, y2)}
            For none:    None (keep freehand stroke)
        """
        if len(points) < self.min_points:
            return None

        pts = np.array(points, dtype=np.int32)

        # Check if the stroke is closed (start and end points are near each other)
        start = pts[0]
        end = pts[-1]
        stroke_length = cv2.arcLength(pts.reshape(-1, 1, 2), closed=False)

        if stroke_length < 50:
            return None  # Too short to analyze

        closure_distance = math.hypot(end[0] - start[0], end[1] - start[1])
        closure_ratio = closure_distance / stroke_length

        # --- Line snapping: open strokes that are nearly straight ---
        if closure_ratio > 0.25:
            line_result = self._check_line(pts, start, end, stroke_length)
            if line_result is not None:
                return line_result
            return None

        # Build a contour from the points
        contour = pts.reshape(-1, 1, 2)

        # --- Circle detection ---
        circle_result = self._check_circle(contour)
        if circle_result is not None:
            return circle_result

        # --- Rectangle detection ---
        rect_result = self._check_rectangle(contour)
        if rect_result is not None:
            return rect_result

        return None

    def _check_circle(self, contour):
        """Check if the contour approximates a circle."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter == 0:
            return None

        # Circularity: 1.0 for a perfect circle
        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        if circularity >= self.circularity_threshold:
            # Compute the minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            return {
                "type": "circle",
                "center": (int(cx), int(cy)),
                "radius": int(radius),
            }

        return None

    def _check_rectangle(self, contour):
        """Check if the contour approximates a rectangle."""
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, closed=True)

        if len(approx) == 4:
            # Verify it's roughly rectangular by checking angles
            # Use the bounding rect as the snapped shape
            x, y, w, h = cv2.boundingRect(approx)

            # Check aspect ratio is reasonable (not a sliver)
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect < 6.0:
                # Also check that the bounding rect area is close to the contour area
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                if rect_area > 0:
                    fill_ratio = contour_area / rect_area
                    if fill_ratio > 0.6:
                        return {
                            "type": "rectangle",
                            "pt1": (x, y),
                            "pt2": (x + w, y + h),
                        }

        return None

    def _check_line(self, pts, start, end, stroke_length):
        """Check if an open stroke is nearly straight and should snap to a line."""
        direct_dist = math.hypot(end[0] - start[0], end[1] - start[1])

        # Line must have meaningful length
        if direct_dist < 40:
            return None

        # Compute maximum perpendicular deviation from the start-end line
        line_vec = np.array([end[0] - start[0], end[1] - start[1]], dtype=np.float64)
        line_len = math.hypot(line_vec[0], line_vec[1])
        if line_len < 1e-5:
            return None

        normal = np.array([-line_vec[1], line_vec[0]]) / line_len
        pts_float = pts.astype(np.float64)
        offsets = pts_float - start.astype(np.float64)
        projections = np.abs(offsets @ normal)
        max_deviation = float(projections.max())

        # Straightness ratio: deviation relative to line length
        straightness = max_deviation / direct_dist

        # Also check efficiency: direct distance vs arc length
        efficiency = direct_dist / stroke_length

        if straightness < 0.06 and efficiency > 0.85:
            return {
                "type": "line",
                "pt1": (int(start[0]), int(start[1])),
                "pt2": (int(end[0]), int(end[1])),
            }

        return None
