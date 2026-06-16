"""
main.py — Air Canvas: Gesture-controlled drawing application.

Orchestrates webcam capture, hand tracking, canvas management, and UI rendering.
Supports drawing, hovering, erasing, shape snapping, undo/redo, and save/export.

Controls:
    Hand Gestures:
        - Index finger only:          Draw mode
        - Index + Middle fingers:     Hover mode (move freely, select colors, resize brush)
        - Index + Middle + Ring:      Eraser mode
        - Fist (hold 2 seconds):      Clear canvas

    Keyboard:
        - S:       Save canvas as PNG
        - Ctrl+Z:  Undo last stroke
        - Ctrl+Y:  Redo last undone stroke
        - Q:       Quit
"""

import os
import time
import cv2
import numpy as np
from datetime import datetime
from hand_tracker import HandTracker
from canvas_manager import CanvasManager
from ui_manager import UIManager
from doodle_classifier import DoodleClassifier


def save_canvas(canvas, frame, save_dir="saves"):
    """
    Save the current canvas in two formats:
      1. Drawing only (black background)
      2. Composite (drawing overlaid on webcam frame)

    Args:
        canvas: The drawing canvas numpy array.
        frame: The current webcam frame.
        save_dir: Directory to save files in.

    Returns:
        Tuple of (canvas_path, composite_path).
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Canvas only
    canvas_path = os.path.join(save_dir, f"canvas_{timestamp}.png")
    cv2.imwrite(canvas_path, canvas)

    # Composite: blend drawing over the webcam frame (vivid 85% opacity, original background brightness)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])

    alpha = 0.85
    blended = cv2.addWeighted(frame, 1.0 - alpha, canvas, alpha, 0)

    composite = frame.copy()
    np.copyto(composite, blended, where=(mask_3ch == 255))

    composite_path = os.path.join(save_dir, f"composite_{timestamp}.png")
    cv2.imwrite(composite_path, composite)

    return canvas_path, composite_path


def main():
    """Main application loop."""
    # --- Initialize components ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Request widescreen resolution at a highly optimized, high-FPS scale
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    tracker = HandTracker()
    canvas_mgr = CanvasManager()
    ui = UIManager()
    doodle_classifier = DoodleClassifier()

    # Create resizable window with aspect ratio constraints
    cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Canvas", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # Doodle recognition display state
    current_doodle_label = None
    doodle_recognized_time = 0.0

    # Track previous gesture for pen-up detection
    prev_gesture = HandTracker.GESTURE_NONE
    # Track if we were in a top-bar selection cooldown to prevent rapid re-triggers
    top_bar_cooldown = 0

    print("Air Canvas started. Press 'Q' to quit.")
    print("Controls:")
    print("  Index finger:          Draw")
    print("  Index + Middle:        Hover / Select colors")
    print("  Index + Middle + Ring:  Erase")
    print("  Fist (2 sec hold):     Clear canvas")
    print("  S key:                 Save drawing")
    print("  Ctrl+Z / Ctrl+Y:       Undo / Redo")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Mirror the frame for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        clean_frame = frame.copy()

        # --- Hand tracking ---
        landmarks = tracker.find_hands(frame)
        gesture = tracker.classify_gesture()
        index_tip = tracker.get_index_tip()

        # --- Pen-up detection: transitioning away from draw or erase ---
        if prev_gesture in (HandTracker.GESTURE_DRAW, HandTracker.GESTURE_ERASE) and \
           gesture not in (HandTracker.GESTURE_DRAW, HandTracker.GESTURE_ERASE):
            canvas_mgr.end_stroke()
            if prev_gesture == HandTracker.GESTURE_DRAW and canvas_mgr.strokes:
                last_stroke = canvas_mgr.strokes[-1]
                # Classify the completed stroke
                label = doodle_classifier.classify(last_stroke.points)
                if label:
                    current_doodle_label = label
                    doodle_recognized_time = time.time()

        # --- Gesture handling ---
        if gesture == HandTracker.GESTURE_DRAW:
            if prev_gesture != HandTracker.GESTURE_DRAW:
                # Starting a new stroke
                canvas_mgr.start_stroke(
                    color=ui.get_draw_color(),
                    thickness=ui.get_draw_thickness(),
                )
                ui.set_eraser_mode(False)

            if index_tip is not None:
                canvas_mgr.add_point(index_tip[0], index_tip[1])

        elif gesture == HandTracker.GESTURE_HOVER:
            ui.set_eraser_mode(False)

            # Brush resize via pinch (only when hovering over the SizeIndicator and actively pinching)
            if index_tip is not None:
                size_indicator_btn = next((b for b in ui.buttons if b["label"] == "SizeIndicator"), None)
                if size_indicator_btn and size_indicator_btn["x1"] <= index_tip[0] <= size_indicator_btn["x2"] and size_indicator_btn["y1"] <= index_tip[1] <= size_indicator_btn["y2"]:
                    pinch_dist = tracker.get_pinch_distance()
                    if pinch_dist is not None and pinch_dist < 0.45:
                        ui.update_brush_from_pinch(pinch_dist)

            # Top bar selection (requires pinch-to-click to prevent jittery hover selection)
            if index_tip is not None and top_bar_cooldown <= 0:
                pinch_dist = tracker.get_pinch_distance()
                if pinch_dist is not None and pinch_dist < 0.45:
                    btn = ui.check_top_bar_click(index_tip[0], index_tip[1])
                    if btn is not None:
                        result = ui.handle_button_press(btn)
                        if result == "clear":
                            canvas_mgr.clear()
                        elif result == "undo":
                            canvas_mgr.undo()
                        elif result == "redo":
                            canvas_mgr.redo()
                        top_bar_cooldown = 15  # ~0.5 sec at 30fps

        elif gesture == HandTracker.GESTURE_ERASE:
            if prev_gesture != HandTracker.GESTURE_ERASE:
                # Starting an erase stroke
                ui.set_eraser_mode(True)
                canvas_mgr.start_stroke(
                    color=ui.get_eraser_color(),
                    thickness=ui.get_draw_thickness(),
                )

            if index_tip is not None:
                canvas_mgr.add_point(index_tip[0], index_tip[1])

        elif gesture == HandTracker.GESTURE_CLEAR:
            canvas_mgr.clear()

        # Decrement cooldown
        if top_bar_cooldown > 0:
            top_bar_cooldown -= 1

        prev_gesture = gesture

        # --- Render canvas ---
        canvas = canvas_mgr.render(w, h)

        # --- Draw UI overlays on the canvas (right side) ---
        if current_doodle_label:
            if time.time() - doodle_recognized_time < 2.0:
                ui.draw_doodle_recognition(canvas, current_doodle_label)
            else:
                current_doodle_label = None

        # --- Draw UI overlays directly on the camera frame (left side) ---
        ui.overlay_artwork(frame, canvas)
        tracker.draw_landmarks(frame)
        ui.draw_top_bar(frame, index_tip if gesture == HandTracker.GESTURE_HOVER else None)
        ui.update_fps()
        ui.draw_fps(frame)
        ui.draw_gesture_label(frame, gesture)
        ui.draw_save_notification(frame)

        # Draw brush size indicator in hover mode
        if gesture == HandTracker.GESTURE_HOVER and index_tip is not None:
            ui.draw_brush_indicator(frame, index_tip)

        # Draw fist progress indicator
        fist_progress = tracker.get_fist_progress()
        if fist_progress > 0:
            wrist = tracker.pixel_landmarks[0] if tracker.pixel_landmarks else None
            ui.draw_fist_progress(frame, fist_progress, wrist)
            ui.draw_canvas_fist_progress(canvas, fist_progress)

        # --- Side-by-side layout: stack frame (with UI) and canvas horizontally ---
        display = np.hstack([frame, canvas])

        # --- Display ---
        cv2.imshow("Air Canvas", display)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord('s') or key == ord('S'):
            canvas_only = canvas_mgr.render(w, h)
            c_path, comp_path = save_canvas(canvas_only, clean_frame)
            ui.trigger_save_notification()
            print(f"Saved: {c_path}, {comp_path}")

        elif key == ord('o') or key == ord('O'):
            ui.show_overlay = not ui.show_overlay

        elif key == 26:  # Ctrl+Z
            canvas_mgr.undo()

        elif key == 25:  # Ctrl+Y
            canvas_mgr.redo()

    # --- Cleanup ---
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Air Canvas closed.")


if __name__ == "__main__":
    main()
