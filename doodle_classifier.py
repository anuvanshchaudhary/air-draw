"""
doodle_classifier.py — Rule-based geometric doodle classifier.

Analyzes completed stroke geometry to classify drawings into categories
like star, heart, triangle, circle, rectangle, line, or arrow.
Uses convex hull analysis, peak detection, and geometric ratios
rather than ML models for lightweight, zero-dependency inference.
"""

import math
import cv2
import numpy as np


class DoodleClassifier:
    """Classifies completed strokes into doodle categories using geometric heuristics."""

    # Classification labels with emoji
    LABELS = {
        "star": "Star ⭐",
        "heart": "Heart ❤️",
        "triangle": "Triangle 🔺",
        "circle": "Circle ⭕",
        "rectangle": "Rectangle ⬛",
        "line": "Line 📏",
        "arrow": "Arrow ➡️",
        "zigzag": "Zigzag ⚡",
    }

    def __init__(self, min_points=8):
        """
        Args:
            min_points: Minimum stroke points required to attempt classification.
        """
        self.min_points = min_points

    def classify(self, points):
        """
        Classify a completed stroke into a doodle category.

        Args:
            points: List of (x, y) tuples from the stroke.

        Returns:
            A string label like "Star ⭐" or None if unrecognized.
        """
        if len(points) < self.min_points:
            return None

        pts = np.array(points, dtype=np.float32)

        # Basic metrics
        start = pts[0]
        end = pts[-1]
        stroke_len = self._arc_length(pts)
        if stroke_len < 30:
            return None

        closure_dist = math.hypot(end[0] - start[0], end[1] - start[1])
        closure_ratio = closure_dist / stroke_len

        # Bounding box
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bb_w = x_max - x_min
        bb_h = y_max - y_min
        if bb_w < 10 or bb_h < 5:
            # Could be a near-horizontal/vertical line
            pass

        # --- Open shapes (not closed) ---
        if closure_ratio > 0.20:
            return self._classify_open(pts, stroke_len, start, end, bb_w, bb_h)

        # --- Closed shapes ---
        return self._classify_closed(pts, stroke_len, bb_w, bb_h)

    def _classify_open(self, pts, stroke_len, start, end, bb_w, bb_h):
        """Classify open (non-closed) strokes: lines, arrows, zigzags."""
        # Check for straightness: how far do points deviate from the start-end line?
        deviation = self._max_perpendicular_deviation(pts, start, end)
        direct_dist = math.hypot(end[0] - start[0], end[1] - start[1])

        if direct_dist > 0:
            straightness = deviation / direct_dist
        else:
            straightness = 999

        # Near-straight line
        if straightness < 0.08 and direct_dist > 40:
            return self.LABELS["line"]

        # Zigzag: many direction reversals
        reversals = self._count_direction_reversals(pts)
        if reversals >= 4 and stroke_len > 80:
            return self.LABELS["zigzag"]

        # Arrow: straight segment + small fork at end
        if straightness < 0.15 and direct_dist > 60:
            # Check if last 20% of points fork
            tail_start = int(len(pts) * 0.8)
            tail_pts = pts[tail_start:]
            if len(tail_pts) > 3:
                tail_dev = self._max_perpendicular_deviation(tail_pts, tail_pts[0], tail_pts[-1])
                if tail_dev > 8:
                    return self.LABELS["arrow"]

        return None

    def _classify_closed(self, pts, stroke_len, bb_w, bb_h):
        """Classify closed strokes: circles, rectangles, triangles, stars, hearts."""
        int_pts = pts.astype(np.int32)
        contour = int_pts.reshape(-1, 1, 2)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            return None

        # Circularity: 1.0 for perfect circle
        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        # Convex hull analysis
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5)

        # Polygon approximation
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, closed=True)
        num_vertices = len(approx)

        # Center of bounding box
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2

        # --- Star detection ---
        # Stars have low solidity (many concavities) but moderate area
        if solidity < 0.55 and num_vertices >= 8 and area > 500:
            # Count concave indentations using radial distance from center
            peaks, valleys = self._count_radial_peaks(pts, cx, cy)
            if peaks >= 4 and valleys >= 4:
                return self.LABELS["star"]

        # --- Heart detection ---
        # Hearts have two lobes at the top and a point at the bottom
        if 0.45 < solidity < 0.75 and bb_h > 30:
            if self._check_heart_shape(pts, cx, cy, bb_w, bb_h):
                return self.LABELS["heart"]

        # --- Circle ---
        if circularity >= 0.78:
            return self.LABELS["circle"]

        # --- Rectangle ---
        if num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            if rect_area > 0:
                fill_ratio = area / rect_area
                aspect = max(w, h) / (min(w, h) + 1e-5)
                if fill_ratio > 0.6 and aspect < 6.0:
                    return self.LABELS["rectangle"]

        # --- Triangle ---
        if num_vertices == 3:
            tri_hull_area = cv2.contourArea(approx)
            if tri_hull_area > 0:
                tri_fill = area / tri_hull_area
                if tri_fill > 0.5:
                    return self.LABELS["triangle"]

        # Relaxed triangle: 3-5 vertices with triangular aspect
        if 3 <= num_vertices <= 5 and solidity > 0.7:
            approx_loose = cv2.approxPolyDP(contour, 0.06 * perimeter, closed=True)
            if len(approx_loose) == 3:
                return self.LABELS["triangle"]

        return None

    def _arc_length(self, pts):
        """Compute total arc length of a point sequence."""
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

    def _max_perpendicular_deviation(self, pts, start, end):
        """Compute maximum perpendicular distance from any point to the start-end line."""
        line_vec = end - start
        line_len = math.hypot(line_vec[0], line_vec[1])
        if line_len < 1e-5:
            # All points are at the same location
            return 0.0

        # Unit normal of the line
        normal = np.array([-line_vec[1], line_vec[0]]) / line_len

        # Project all points onto the normal
        offsets = pts - start
        projections = np.abs(offsets @ normal)
        return float(projections.max())

    def _count_direction_reversals(self, pts):
        """Count how many times the stroke reverses direction along its primary axis."""
        if len(pts) < 3:
            return 0
        # Use differences in the dominant axis
        diffs = np.diff(pts, axis=0)
        # Check sign changes in x-direction
        x_signs = np.sign(diffs[:, 0])
        x_reversals = np.sum(np.abs(np.diff(x_signs)) > 1)
        # Check sign changes in y-direction
        y_signs = np.sign(diffs[:, 1])
        y_reversals = np.sum(np.abs(np.diff(y_signs)) > 1)
        return int(max(x_reversals, y_reversals))

    def _count_radial_peaks(self, pts, cx, cy):
        """Count peaks and valleys in the radial distance from center."""
        distances = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        if len(distances) < 5:
            return 0, 0

        # Smooth slightly to reduce noise
        kernel_size = max(3, len(distances) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(distances, kernel, mode='same')

        mean_dist = smoothed.mean()
        peaks = 0
        valleys = 0
        above = smoothed[0] > mean_dist
        for d in smoothed[1:]:
            currently_above = d > mean_dist
            if currently_above and not above:
                valleys += 1
            elif not currently_above and above:
                peaks += 1
            above = currently_above

        return peaks, valleys

    def _check_heart_shape(self, pts, cx, cy, bb_w, bb_h):
        """Check if the stroke looks like a heart shape."""
        # A heart has: two lobes in the upper half, a point at the bottom
        # Split into upper and lower halves
        upper = pts[pts[:, 1] < cy]
        lower = pts[pts[:, 1] >= cy]

        if len(upper) < 5 or len(lower) < 5:
            return False

        # Upper half should be wider than lower half
        upper_width = upper[:, 0].max() - upper[:, 0].min()
        lower_width = lower[:, 0].max() - lower[:, 0].min()

        if upper_width < lower_width * 0.8:
            return False

        # Check for two lobes: the upper half should have a dip in the middle
        # Split upper into left and right of center
        upper_left = upper[upper[:, 0] < cx]
        upper_right = upper[upper[:, 0] >= cx]

        if len(upper_left) < 2 or len(upper_right) < 2:
            return False

        # Both sides should have points above center
        left_top = upper_left[:, 1].min()
        right_top = upper_right[:, 1].min()

        # The center top should dip down (higher y value)
        center_band = upper[np.abs(upper[:, 0] - cx) < bb_w * 0.15]
        if len(center_band) < 1:
            return False

        center_top = center_band[:, 1].min()

        # The two lobes should be higher (lower y) than the center dip
        dip_amount = center_top - min(left_top, right_top)
        if dip_amount > bb_h * 0.05:
            return True

        return False
