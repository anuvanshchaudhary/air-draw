"""
ui_manager.py — Visual overlays: top bar, color swatches, FPS, brush indicator, mini preview.

Handles all non-drawing visual elements rendered on top of the webcam frame.
"""

import time
import cv2
import numpy as np


class UIManager:
    """Manages the visual UI overlays for the Air Canvas application."""

    # Top bar configuration
    TOP_BAR_HEIGHT = 80

    # Modernized color palette (BGR format for OpenCV)
    COLORS = {
        "Red": (59, 76, 255),       # Soft Coral Red
        "Orange": (0, 153, 255),    # Vibrant Orange
        "Yellow": (0, 220, 255),    # Sunny Yellow
        "Green": (100, 217, 121),   # Emerald Green
        "Cyan": (230, 224, 73),     # Soft Teal/Cyan
        "Blue": (255, 120, 0),      # Electric Blue
        "Purple": (222, 90, 160),   # Violet/Purple
        "Pink": (180, 105, 255),    # Hot Pink
        "White": (245, 245, 245),   # Off-white
    }
    COLOR_ORDER = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink", "White"]

    # Default brush settings
    DEFAULT_COLOR_NAME = "Blue"
    DEFAULT_THICKNESS = 5
    ERASER_THICKNESS = 40

    # Pinch-to-resize mapping (normalized by palm length)
    PINCH_DIST_MIN = 0.25
    PINCH_DIST_MAX = 1.0
    BRUSH_SIZE_MIN = 2
    BRUSH_SIZE_MAX = 30

    # Mini canvas preview
    PREVIEW_WIDTH = 200
    PREVIEW_HEIGHT = 150
    PREVIEW_MARGIN = 15

    def __init__(self):
        self.active_color_name = self.DEFAULT_COLOR_NAME
        self.active_color = self.COLORS[self.active_color_name]
        self.brush_thickness = self.DEFAULT_THICKNESS
        self.draw_thickness = self.DEFAULT_THICKNESS
        self.eraser_mode = False
        self.show_overlay = True

        # Button regions: list of dicts representing button details
        self.buttons = []
        self._last_width = 0

        # "Saved!" notification state
        self._save_msg_time = None
        self.SAVE_MSG_DURATION = 1.5

        # FPS tracking
        self._frame_times = []
        self._fps = 0.0

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=8, thickness=-1):
        """Draw a rounded rectangle using OpenCV primitives."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure radius doesn't exceed half the width/height
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        r = min(radius, w // 2, h // 2)
        
        if thickness < 0:
            # Filled rounded rectangle
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
        else:
            # Outlined rounded rectangle
            # Draw straight edges
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            
            # Draw corners
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _build_buttons(self, width):
        """Pre-compute button regions for the top bar based on frame width."""
        self.buttons = []
        
        # Scale factor based on standard 640 width
        scale = width / 640.0
        margin = int(10 * scale)

        # Clear button
        w_clear = int(55 * scale)
        x = margin
        self.buttons.append({
            "label": "Clear",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_clear,
            "y2": 65,
            "color": (50, 50, 180),  # Soft Red
            "text": "CLEAR",
        })
        x += w_clear + margin

        # Undo button
        w_undo = int(45 * scale)
        self.buttons.append({
            "label": "Undo",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_undo,
            "y2": 65,
            "color": (100, 100, 100),  # Neutral Gray
            "text": "UNDO",
        })
        x += w_undo + margin

        # Redo button
        w_redo = int(45 * scale)
        self.buttons.append({
            "label": "Redo",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_redo,
            "y2": 65,
            "color": (100, 100, 100),  # Neutral Gray
            "text": "REDO",
        })
        x += w_redo + margin

        # Size- button
        w_size_dec = int(30 * scale)
        self.buttons.append({
            "label": "Size-",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_size_dec,
            "y2": 65,
            "color": (80, 80, 80),
            "text": "-",
        })
        x += w_size_dec + margin

        # SizeIndicator display widget
        w_indicator = int(45 * scale)
        self.buttons.append({
            "label": "SizeIndicator",
            "type": "indicator",
            "x1": x,
            "y1": 15,
            "x2": x + w_indicator,
            "y2": 65,
        })
        x += w_indicator + margin

        # Size+ button
        w_size_inc = int(30 * scale)
        self.buttons.append({
            "label": "Size+",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_size_inc,
            "y2": 65,
            "color": (80, 80, 80),
            "text": "+",
        })
        x += w_size_inc + margin

        # Overlay button
        w_overlay = int(70 * scale)
        self.buttons.append({
            "label": "Overlay",
            "type": "action",
            "x1": x,
            "y1": 15,
            "x2": x + w_overlay,
            "y2": 65,
            "color": (222, 90, 160),  # Will be overridden dynamically in rendering
            "text": "OVERLAY",
        })
        x += w_overlay + margin

        # Colors swatches start after Overlay button
        color_start_x = x + int(15 * scale)
        color_end_x = width - margin - int(15 * scale)
        num_colors = len(self.COLOR_ORDER)

        available_width = color_end_x - color_start_x
        ideal_step = int(50 * scale)
        total_palette_width = (num_colors - 1) * ideal_step

        if total_palette_width <= available_width:
            # Center the palette in the available space
            start_x = color_start_x + (available_width - total_palette_width) // 2
            step = ideal_step
        else:
            # Compress to fit
            start_x = color_start_x
            step = available_width / max(1, num_colors - 1)

        r = int(14 * scale)
        for i, color_name in enumerate(self.COLOR_ORDER):
            cx = int(start_x + i * step)
            cy = 40
            # Add to buttons
            self.buttons.append({
                "label": color_name,
                "type": "color",
                "cx": cx,
                "cy": cy,
                "r": r,
                # Bounding box for easy detection
                "x1": cx - r - 4,
                "y1": cy - r - 4,
                "x2": cx + r + 4,
                "y2": cy + r + 4,
            })

    def draw_top_bar(self, frame, hover_pos=None):
        """Draw the top menu bar with action buttons, color swatches, and active color glow."""
        h, w = frame.shape[:2]

        # Rebuild buttons if width changed
        if self._last_width != w:
            self._last_width = w
            self._build_buttons(w)

        # Semi-transparent dark background for the bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.TOP_BAR_HEIGHT), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Dynamic active color glow line along the bottom border of the top bar
        cv2.line(frame, (0, self.TOP_BAR_HEIGHT), (w, self.TOP_BAR_HEIGHT), self.active_color, 2, cv2.LINE_AA)

        # Unpack hover coordinates
        hx, hy = hover_pos if hover_pos is not None else (None, None)

        for btn in self.buttons:
            # Check if hovered
            is_hovered = False
            if hx is not None and hy is not None:
                if btn["x1"] <= hx <= btn["x2"] and btn["y1"] <= hy <= btn["y2"]:
                    is_hovered = True

            if btn["type"] == "action":
                x1, y1, x2, y2 = btn["x1"], btn["y1"], btn["x2"], btn["y2"]
                if btn["label"] == "Overlay":
                    color = (222, 90, 160) if self.show_overlay else (80, 80, 80)
                else:
                    color = btn["color"]
                
                # Dynamic hover halo outline or lighter background
                if is_hovered:
                    # Draw a nice glowing border
                    self.draw_rounded_rect(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 10, 2)
                    bg_color = tuple(min(c + 40, 255) for c in color)
                else:
                    bg_color = color

                # Draw filled button
                self.draw_rounded_rect(frame, (x1, y1), (x2, y2), bg_color, 8, -1)
                # Draw subtle outline
                self.draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), 8, 1)

                # Center and draw text (larger scale for '-' and '+' to make them look nice)
                text = btn["text"]
                scale = w / 640.0
                if text == "OVERLAY":
                    font_scale = 0.32 * scale
                    font_thick = max(1, int(1 * scale))
                elif text in ("-", "+"):
                    font_scale = 0.6 * scale
                    font_thick = max(1, int(2 * scale))
                else:
                    font_scale = 0.4 * scale
                    font_thick = max(1, int(1 * scale))
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                tx = x1 + (x2 - x1 - text_w) // 2
                ty = y1 + (y2 - y1 + text_h) // 2
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            elif btn["type"] == "indicator":
                x1, y1, x2, y2 = btn["x1"], btn["y1"], btn["x2"], btn["y2"]
                if is_hovered:
                    # Draw hover glow border
                    self.draw_rounded_rect(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 10, 2)
                    bg_color = (60, 60, 60)
                else:
                    bg_color = (40, 40, 40)

                # Draw a subtle container for the indicator
                self.draw_rounded_rect(frame, (x1, y1), (x2, y2), bg_color, 8, -1)
                self.draw_rounded_rect(frame, (x1, y1), (x2, y2), (70, 70, 70), 8, 1)

                # Draw a circle representing the active brush size inside the indicator
                cx = x1 + (x2 - x1) // 2
                cy = y1 + 18
                draw_r = max(2, self.brush_thickness // 2)
                cv2.circle(frame, (cx, cy), draw_r, self.active_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), draw_r, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw thickness text below the circle
                text = f"{self.brush_thickness}px"
                font_scale = 0.35 * scale
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx = x1 + (x2 - x1 - text_w) // 2
                ty = y2 - 8
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

            elif btn["type"] == "color":
                cx, cy, r = btn["cx"], btn["cy"], btn["r"]
                color_name = btn["label"]
                color = self.COLORS[color_name]

                # Interactive hover scaling
                draw_r = r + 4 if is_hovered else r

                # Draw color circle
                cv2.circle(frame, (cx, cy), draw_r, color, -1)
                # Outer stroke for contrast
                cv2.circle(frame, (cx, cy), draw_r, (255, 255, 255), 1, cv2.LINE_AA)

                # Highlight active color with double ring selection border
                if color_name == self.active_color_name:
                    cv2.circle(frame, (cx, cy), draw_r + 4, (255, 255, 255), 2, cv2.LINE_AA)

    def check_top_bar_click(self, x, y):
        """
        Check if a point (finger tip in hover mode) is within a top bar button.

        Returns:
            The button label ("Clear", "Red", "Undo", etc.) or None.
        """
        if y > self.TOP_BAR_HEIGHT:
            return None

        for btn in self.buttons:
            if btn["type"] == "indicator":
                continue
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                return btn["label"]

        return None

    def handle_button_press(self, label):
        """
        Handle a top bar button press.

        Args:
            label: The button label that was pressed.

        Returns:
            "clear", "undo", "redo", "size", or "color" if changed, None otherwise.
        """
        if label == "Clear":
            return "clear"
        elif label == "Undo":
            return "undo"
        elif label == "Redo":
            return "redo"
        elif label == "Size-":
            self.brush_thickness = max(self.BRUSH_SIZE_MIN, self.brush_thickness - 3)
            return "size"
        elif label == "Size+":
            self.brush_thickness = min(self.BRUSH_SIZE_MAX, self.brush_thickness + 3)
            return "size"
        elif label == "Overlay":
            self.show_overlay = not self.show_overlay
            return "overlay"
        elif label in self.COLORS:
            self.active_color_name = label
            self.active_color = self.COLORS[label]
            return "color"
        return None

    def get_draw_color(self):
        """Return the current active drawing color (BGR tuple)."""
        return self.active_color

    def get_draw_thickness(self):
        """Return the current brush thickness."""
        return self.brush_thickness

    def set_eraser_mode(self, active):
        """Toggle eraser mode (black color, large brush)."""
        if active == self.eraser_mode:
            return
        
        self.eraser_mode = active
        if active:
            # Save current drawing thickness before entering eraser mode
            self.draw_thickness = self.brush_thickness
            self.brush_thickness = self.ERASER_THICKNESS
        else:
            # Restore drawing thickness when exiting eraser mode
            self.brush_thickness = self.draw_thickness

    def get_eraser_color(self):
        """Return the eraser color (black, matching canvas background)."""
        return (0, 0, 0)

    def update_brush_from_pinch(self, pinch_distance):
        """
        Map normalized pinch distance to brush size.

        Args:
            pinch_distance: Normalized distance between thumb tip and index tip.
        """
        # Clamp to range
        d = max(self.PINCH_DIST_MIN, min(pinch_distance, self.PINCH_DIST_MAX))
        # Linear interpolation
        ratio = (d - self.PINCH_DIST_MIN) / (self.PINCH_DIST_MAX - self.PINCH_DIST_MIN)
        target_thickness = int(
            self.BRUSH_SIZE_MIN + ratio * (self.BRUSH_SIZE_MAX - self.BRUSH_SIZE_MIN)
        )
        # Apply Exponential Moving Average (EMA) with alpha = 0.15 for ultra-smooth transition
        self.brush_thickness = int(0.15 * target_thickness + 0.85 * self.brush_thickness)
        self.brush_thickness = max(self.BRUSH_SIZE_MIN, min(self.brush_thickness, self.BRUSH_SIZE_MAX))

    def draw_brush_indicator(self, frame, position):
        """
        Draw a circle at the given position showing the current brush size.

        Args:
            frame: The frame to draw on.
            position: (x, y) tuple for the indicator center.
        """
        if position is None:
            return
        x, y = position
        
        # Draw dynamic hover circle showing brush size
        cv2.circle(frame, (x, y), self.brush_thickness, self.active_color, 2, cv2.LINE_AA)
        
        # Clean background label for the brush size text
        label_text = f"{self.brush_thickness}px"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        
        lx1 = x + self.brush_thickness + 5
        ly1 = y - th - 3
        lx2 = lx1 + tw + 6
        ly2 = y + 5
        
        overlay = frame.copy()
        self.draw_rounded_rect(overlay, (lx1, ly1), (lx2, ly2), (30, 30, 30), 4, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        self.draw_rounded_rect(frame, (lx1, ly1), (lx2, ly2), (100, 100, 100), 4, 1)
        
        cv2.putText(
            frame, label_text,
            (lx1 + 3, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA
        )

    def draw_fist_progress(self, frame, progress, position):
        """
        Draw a circular progress indicator for fist-hold-to-clear.

        Args:
            frame: The frame to draw on.
            progress: Float 0.0 to 1.0.
            position: (x, y) center of the indicator.
        """
        if progress <= 0 or position is None:
            return

        x, y = position
        radius = 40
        angle = int(360 * progress)

        # Background circle with overlay
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius + 8, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Progress arc
        cv2.ellipse(frame, (x, y), (radius, radius), -90, 0, angle, (59, 76, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (80, 80, 80), 1, cv2.LINE_AA)
        
        # Label inside the circle
        text = "CLEAR"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 2)
        cv2.putText(
            frame, text,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (59, 76, 255), 2, cv2.LINE_AA
        )

    def draw_gesture_label(self, frame, gesture):
        """Show the active gesture name in a modern floating HUD pill."""
        h = frame.shape[0]
        label_map = {
            "draw": ("DRAW", (100, 217, 121)),      # Emerald green
            "hover": ("HOVER", (0, 153, 255)),      # Orange
            "erase": ("ERASE", (180, 105, 255)),    # Pink/Magenta
            "none": ("---", (150, 150, 150)),        # Gray
            "clear": ("CLEARING", (59, 76, 255)),   # Red/Coral
        }
        text, color = label_map.get(gesture, ("---", (150, 150, 150)))

        # Draw a modern semi-transparent pill container
        pill_x1, pill_y1, pill_x2, pill_y2 = 15, h - 52, 185, h - 17
        overlay = frame.copy()
        self.draw_rounded_rect(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), 10, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        # White thin border
        self.draw_rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (80, 80, 80), 10, 1)

        # Status dot
        dot_cx, dot_cy = pill_x1 + 18, pill_y1 + 18
        cv2.circle(frame, (dot_cx, dot_cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (dot_cx, dot_cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Mode text
        mode_text = f"MODE: {text.upper()}"
        cv2.putText(
            frame, mode_text,
            (pill_x1 + 35, pill_y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 2, cv2.LINE_AA
        )

    def overlay_artwork(self, frame, canvas):
        """
        Overlay the canvas drawing directly on the webcam frame.
        Since the canvas has a black background, we only overlay the non-black pixels.
        """
        if not self.show_overlay:
            return

        # Create mask of drawing area (where canvas is not black)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Convert mask to 3 channels for array operations
        mask_3ch = cv2.merge([mask, mask, mask])

        # Blend the drawing and the frame
        # We use alpha = 0.85 so that the lines are vivid, but still slightly transparent.
        # This gives a premium "augmented reality holographic drawing" feel.
        alpha = 0.85
        blended = cv2.addWeighted(frame, 1.0 - alpha, canvas, alpha, 0)

        # Apply the blended drawing pixels only where the mask is active
        np.copyto(frame, blended, where=(mask_3ch == 255))

    def draw_mini_preview(self, frame, canvas):
        """
        Draw a small thumbnail of the canvas in the bottom-right corner.

        Args:
            frame: Main display frame to overlay onto.
            canvas: The drawing canvas (black background).
        """
        h, w = frame.shape[:2]
        preview = cv2.resize(canvas, (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT))

        # Position in bottom-right
        x1 = w - self.PREVIEW_WIDTH - self.PREVIEW_MARGIN
        y1 = h - self.PREVIEW_HEIGHT - self.PREVIEW_MARGIN
        x2 = x1 + self.PREVIEW_WIDTH
        y2 = y1 + self.PREVIEW_HEIGHT

        # Draw a beautiful glassmorphic card container background for preview
        card_x1, card_y1 = x1 - 10, y1 - 25
        card_x2, card_y2 = x2 + 10, y2 + 10
        
        overlay = frame.copy()
        self.draw_rounded_rect(overlay, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), 12, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        self.draw_rounded_rect(frame, (card_x1, card_y1), (card_x2, card_y2), (80, 80, 80), 12, 1)

        # Label above preview
        cv2.putText(
            frame, "CANVAS PREVIEW",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Draw a clean border around the actual thumbnail
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (100, 100, 100), 1)

        # Overlay the preview
        frame[y1:y2, x1:x2] = preview

    def update_fps(self):
        """Update FPS calculation. Call once per frame."""
        now = time.time()
        self._frame_times.append(now)

        # Keep only the last 30 frame timestamps
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed



    def draw_fps(self, frame):
        """Draw FPS counter as an elegant status pill in the top bar."""
        w = frame.shape[1]
        
        # Color-coding based on FPS
        if self._fps >= 20:
            status_color = (100, 217, 121)  # Emerald green
        elif self._fps >= 10:
            status_color = (0, 153, 255)    # Orange
        else:
            status_color = (59, 76, 255)    # Red/Coral
            
        # Draw a small pill container in the top-right just below top bar
        pill_x1, pill_y1 = w - 105, 95
        pill_x2, pill_y2 = w - 15, 125
        
        overlay = frame.copy()
        self.draw_rounded_rect(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), 8, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        self.draw_rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (80, 80, 80), 8, 1)
        
        # Status text
        fps_text = f"{int(self._fps)} FPS"
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
        tx = pill_x1 + (pill_x2 - pill_x1 - text_w) // 2
        ty = pill_y1 + (pill_y2 - pill_y1 + text_h) // 2
        
        # Small colored status dot
        cv2.circle(frame, (pill_x1 + 12, pill_y1 + (pill_y2 - pill_y1) // 2), 4, status_color, -1, cv2.LINE_AA)
        
        cv2.putText(
            frame, fps_text,
            (pill_x1 + 22, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA
        )

    def trigger_save_notification(self):
        """Start showing the 'Saved!' notification."""
        self._save_msg_time = time.time()

    def draw_save_notification(self, frame):
        """Draw the 'Saved!' notification as a beautiful centered glassmorphic card."""
        if self._save_msg_time is None:
            return

        elapsed = time.time() - self._save_msg_time
        if elapsed > self.SAVE_MSG_DURATION:
            self._save_msg_time = None
            return

        h, w = frame.shape[:2]
        
        # Center of the screen coordinates
        cx, cy = w // 2, h // 2
        card_w, card_h = 240, 100
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2

        # Translucent glassmorphic card with rounded corners
        overlay = frame.copy()
        self.draw_rounded_rect(overlay, (x1, y1), (x2, y2), (20, 30, 20), 15, -1) # Dark green-ish tint
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Glowing green outer border
        self.draw_rounded_rect(frame, (x1, y1), (x2, y2), (100, 217, 121), 15, 2)

        # Success green check circle
        check_cx, check_cy = cx, y1 + 35
        cv2.circle(frame, (check_cx, check_cy), 15, (100, 217, 121), -1, cv2.LINE_AA)
        
        # Draw checkmark lines
        cv2.line(frame, (check_cx - 6, check_cy), (check_cx - 2, check_cy + 4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (check_cx - 2, check_cy + 4), (check_cx + 7, check_cy - 5), (255, 255, 255), 2, cv2.LINE_AA)

        # Notification text
        text = "Canvas Saved!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(
            frame, text,
            (cx - tw // 2, y2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

    def draw_canvas_fist_progress(self, canvas, progress):
        """
        Draw a circular progress indicator for fist-hold-to-clear in the center of the canvas.

        Args:
            canvas: The drawing canvas (right pane) to draw on.
            progress: Float 0.0 to 1.0.
        """
        if progress <= 0:
            return

        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2
        radius = 60
        angle = int(360 * progress)

        # Background circle with overlay
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), radius + 15, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        
        # Progress arc
        cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0, angle, (59, 76, 255), 6, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, (80, 80, 80), 1, cv2.LINE_AA)
        
        # Label inside the circle
        text1 = "CLEARING"
        text2 = f"{int(progress * 100)}%"
        
        (tw1, th1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        
        cv2.putText(
            canvas, text1,
            (cx - tw1 // 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 76, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, text2,
            (cx - tw2 // 2, cy + th2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA
        )

    def draw_doodle_recognition(self, canvas, label):
        """
        Draw a beautiful floating badge displaying the recognized doodle on the canvas side.

        Args:
            canvas: The drawing canvas (right pane) to draw on.
            label: The recognized doodle label (e.g., "Star ⭐").
        """
        if not label:
            return

        h, w = canvas.shape[:2]
        
        # Let's place it at the top center of the canvas
        cx = w // 2
        cy = 50
        
        text = f"Detected: {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        card_w, card_h = tw + 40, th + 24
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2
        
        # Translucent glassmorphic card
        overlay = canvas.copy()
        self.draw_rounded_rect(overlay, (x1, y1), (x2, y2), (20, 20, 25), 10, -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        
        # Glowing border with yellow/cyan color
        self.draw_rounded_rect(canvas, (x1, y1), (x2, y2), (0, 220, 255), 10, 2)
        
        # Draw doodle text
        cv2.putText(
            canvas, text,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
