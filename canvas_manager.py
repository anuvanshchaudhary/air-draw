"""
canvas_manager.py — Stroke-based canvas with full history, undo/redo, and shape snapping.

Instead of painting onto a raw bitmap, strokes are stored as objects so we can
perform per-stroke undo, shape snapping, and replay.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import cv2
import numpy as np

from shape_detector import ShapeDetector


@dataclass
class Stroke:
    """Represents a single drawing stroke."""

    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 0, 0)  # BGR
    thickness: int = 5
    shape_type: Optional[str] = None  # "circle", "rectangle", or None (freehand)
    shape_params: Optional[Dict] = None  # Parameters for snapped shapes


class CanvasManager:
    """Manages all drawing strokes, active stroke, undo/redo, and rendering."""

    SMOOTHING_WINDOW = 5
    MAX_UNDO = 500

    def __init__(self):
        self.strokes: List[Stroke] = []
        self.redo_stack: List[Stroke] = []
        self.active_stroke: Optional[Stroke] = None
        self.shape_detector = ShapeDetector()

        # Smoothing buffer for the active stroke
        self._smooth_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

    def start_stroke(self, color, thickness):
        """Begin a new stroke with the given color and thickness."""
        self.active_stroke = Stroke(color=color, thickness=thickness)
        self._smooth_buffer.clear()

    def add_point(self, x, y):
        """
        Add a point to the active stroke with smoothing applied.

        The raw point is added to a rolling buffer, and the smoothed
        (averaged) position is recorded in the stroke.
        """
        if self.active_stroke is None:
            return

        self._smooth_buffer.append((x, y))

        # Compute rolling average
        avg_x = int(sum(p[0] for p in self._smooth_buffer) / len(self._smooth_buffer))
        avg_y = int(sum(p[1] for p in self._smooth_buffer) / len(self._smooth_buffer))

        self.active_stroke.points.append((avg_x, avg_y))

    def end_stroke(self):
        """
        Finish the current stroke.
        Runs shape detection and commits to stroke history.
        """
        if self.active_stroke is None:
            return
        if len(self.active_stroke.points) < 2:
            self.active_stroke = None
            return

        # Attempt shape snapping
        shape = self.shape_detector.detect(self.active_stroke.points)
        if shape is not None:
            self.active_stroke.shape_type = shape["type"]
            self.active_stroke.shape_params = shape

        # Commit to history
        self.strokes.append(self.active_stroke)

        # Trim history if it exceeds the undo cap
        if len(self.strokes) > self.MAX_UNDO:
            self.strokes = self.strokes[-self.MAX_UNDO :]

        # Clear redo stack on new stroke
        self.redo_stack.clear()

        self.active_stroke = None
        self._smooth_buffer.clear()

    def undo(self):
        """Undo the last stroke."""
        if self.strokes:
            stroke = self.strokes.pop()
            self.redo_stack.append(stroke)

    def redo(self):
        """Redo the last undone stroke."""
        if self.redo_stack:
            stroke = self.redo_stack.pop()
            self.strokes.append(stroke)

    def clear(self):
        """Clear all strokes from the canvas."""
        self.strokes.clear()
        self.redo_stack.clear()
        self.active_stroke = None
        self._smooth_buffer.clear()

    def render(self, width, height):
        """
        Render all committed strokes and the active stroke onto a blank canvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            A numpy array (BGR) with all strokes drawn on a black background.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw committed strokes
        for stroke in self.strokes:
            self._draw_stroke(canvas, stroke)

        # Draw the active (in-progress) stroke
        if self.active_stroke is not None:
            self._draw_stroke(canvas, self.active_stroke)

        return canvas

    def _draw_stroke_primitive(self, img, stroke, thickness, color):
        """Draw the raw stroke shape (circle, rectangle, line, or freehand) on the image."""
        if stroke.shape_type == "circle" and stroke.shape_params:
            cv2.circle(
                img,
                stroke.shape_params["center"],
                stroke.shape_params["radius"],
                color,
                thickness,
                cv2.LINE_AA
            )
        elif stroke.shape_type == "rectangle" and stroke.shape_params:
            cv2.rectangle(
                img,
                stroke.shape_params["pt1"],
                stroke.shape_params["pt2"],
                color,
                thickness,
                cv2.LINE_AA
            )
        elif stroke.shape_type == "line" and stroke.shape_params:
            cv2.line(
                img,
                stroke.shape_params["pt1"],
                stroke.shape_params["pt2"],
                color,
                thickness,
                cv2.LINE_AA
            )
        else:
            pts = stroke.points
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    def _draw_stroke_sparkles(self, canvas, stroke):
        """Draw deterministic, non-flickering starry sparkles along the stroke path."""
        pts = stroke.points
        if len(pts) < 2 and not stroke.shape_type:
            return

        # Simple star drawer helper
        def draw_star(img, cx, cy, size, color):
            # Draw primary vertical and horizontal spikes
            cv2.line(img, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
            # Draw secondary diagonal spikes for larger stars
            if size > 4:
                d_size = int(size * 0.6)
                cv2.line(img, (cx - d_size, cy - d_size), (cx + d_size, cy + d_size), color, 1, cv2.LINE_AA)
                cv2.line(img, (cx + d_size, cy - d_size), (cx - d_size, cy + d_size), color, 1, cv2.LINE_AA)
            # Core bright dot
            cv2.circle(img, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)

        # 1. Circle shapes
        if stroke.shape_type == "circle" and stroke.shape_params:
            center = stroke.shape_params["center"]
            radius = stroke.shape_params["radius"]
            num_sparkles = max(3, int(radius / 25))
            for i in range(num_sparkles):
                angle = (i * 2 * np.pi / num_sparkles) + (radius % 10)
                cx = int(center[0] + radius * np.cos(angle))
                cy = int(center[1] + radius * np.sin(angle))
                size = 2 + (i % 2)
                star_color = tuple(min(c + 60, 255) for c in stroke.color)
                draw_star(canvas, cx, cy, size, star_color)

        # 2. Rectangle shapes
        elif stroke.shape_type == "rectangle" and stroke.shape_params:
            pt1 = stroke.shape_params["pt1"]
            pt2 = stroke.shape_params["pt2"]
            x1, y1 = pt1
            x2, y2 = pt2
            perimeter_pts = []
            steps_w = max(2, int(abs(x2 - x1) / 50))
            steps_h = max(2, int(abs(y2 - y1) / 50))
            for i in range(steps_w):
                t = i / steps_w
                perimeter_pts.append((int(x1 + t*(x2 - x1)), y1))
                perimeter_pts.append((int(x1 + t*(x2 - x1)), y2))
            for i in range(steps_h):
                t = i / steps_h
                perimeter_pts.append((x1, int(y1 + t*(y2 - y1))))
                perimeter_pts.append((x2, int(y1 + t*(y2 - y1))))
            for idx, (cx, cy) in enumerate(perimeter_pts):
                if idx % 4 == 0:
                    size = 2 + (idx % 2)
                    star_color = tuple(min(c + 60, 255) for c in stroke.color)
                    draw_star(canvas, cx, cy, size, star_color)

        # 3. Freehand paths and line shapes
        else:
            # For line shapes, use the snapped endpoints; for freehand, use stroke points
            if stroke.shape_type == "line" and stroke.shape_params:
                pt1 = stroke.shape_params["pt1"]
                pt2 = stroke.shape_params["pt2"]
                line_pts = []
                num_sparkle_pts = max(2, int(math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]) / 35))
                for i in range(num_sparkle_pts + 1):
                    t = i / num_sparkle_pts
                    line_pts.append((int(pt1[0] + t*(pt2[0]-pt1[0])), int(pt1[1] + t*(pt2[1]-pt1[1]))))
                sparkle_pts = line_pts
            else:
                sparkle_pts = pts

            for i in range(0, len(sparkle_pts), 15):
                pt = sparkle_pts[i]
                # Deterministic coordinate math for coordinates offset (prevents dancing/flickering)
                offset_x = ((pt[0] * 7 + pt[1] * 13) % 7) - 3
                offset_y = ((pt[0] * 11 + pt[1] * 3) % 7) - 3
                cx = pt[0] + offset_x
                cy = pt[1] + offset_y
                size = 2 + ((pt[0] + pt[1]) % 3)
                star_color = tuple(min(c + 60, 255) for c in stroke.color)
                draw_star(canvas, cx, cy, size, star_color)

    def _draw_stroke(self, canvas, stroke):
        """Draw a single stroke onto the canvas as a solid, premium antialiased line."""
        if stroke.color != (0, 0, 0):
            # Outer glow: thickest line, lowest intensity
            self._draw_stroke_primitive(canvas, stroke, stroke.thickness + 12, tuple(int(c * 0.15) for c in stroke.color))
            # Medium glow: medium thickness, medium intensity
            self._draw_stroke_primitive(canvas, stroke, stroke.thickness + 6, tuple(int(c * 0.4) for c in stroke.color))
            # Core line: target thickness, full color intensity
            self._draw_stroke_primitive(canvas, stroke, stroke.thickness, stroke.color)
            # Center bright core: thinnest line, off-white/high-contrast color
            center_color = tuple(min(255, int(c + (255 - c) * 0.6)) for c in stroke.color)
            self._draw_stroke_primitive(canvas, stroke, max(1, int(stroke.thickness * 0.25)), center_color)
            
            # Starry sparkles along the stroke path
            self._draw_stroke_sparkles(canvas, stroke)
        else:
            # Smart Eraser: keep eraser strokes clean and black
            self._draw_stroke_primitive(canvas, stroke, stroke.thickness, stroke.color)

    def has_content(self):
        """Check if there are any strokes on the canvas."""
        return len(self.strokes) > 0 or self.active_stroke is not None
