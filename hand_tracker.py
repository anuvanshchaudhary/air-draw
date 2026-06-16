"""
hand_tracker.py — Hand detection and gesture classification using MediaPipe.

Wraps MediaPipe HandLandmarker (Tasks API) to extract landmarks and classify gestures:
  - Draw:  Index finger only up
  - Hover: Index + Middle up
  - Erase: Index + Middle + Ring up
  - Clear: Fist (0 fingers) held for 2 seconds
"""

import os
import time
import math
from collections import deque, Counter
import mediapipe as mp
import cv2

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)


class HandTracker:
    """Detects a single hand and classifies gestures from webcam frames."""

    # Landmark indices
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    WRIST = 0

    # Gesture constants
    GESTURE_NONE = "none"
    GESTURE_DRAW = "draw"
    GESTURE_HOVER = "hover"
    GESTURE_ERASE = "erase"
    GESTURE_CLEAR = "clear"

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        # Locate the model file relative to this script
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

        # State for fist-hold timer
        self._fist_start_time = None
        self._clear_triggered = False
        self.FIST_HOLD_DURATION = 2.0  # seconds

        # Last detected data
        self._normalized_landmarks = None  # Raw NormalizedLandmark list
        self.pixel_landmarks = None
        self.gesture = self.GESTURE_NONE
        self.gesture_history = deque(maxlen=5)

    def find_hands(self, frame):
        """
        Process a BGR frame and extract hand landmarks.

        Args:
            frame: BGR numpy array from webcam.

        Returns:
            List of (x, y) pixel coordinates for 21 landmarks, or None if no hand detected.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image from the numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Increment timestamp (monotonically increasing, in milliseconds)
        self._frame_timestamp_ms += 33  # ~30 fps
        results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]  # List of NormalizedLandmark
            self._normalized_landmarks = hand
            self.pixel_landmarks = [
                (int(lm.x * w), int(lm.y * h)) for lm in hand
            ]
            return self.pixel_landmarks

        self._normalized_landmarks = None
        self.pixel_landmarks = None
        return None

    def draw_landmarks(self, frame):
        """Draw hand landmarks on the frame for visual debugging."""
        if self._normalized_landmarks is not None:
            drawing_utils.draw_landmarks(
                frame,
                self._normalized_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )

    def _is_finger_up(self, landmarks, tip_id, pip_id):
        """Check if a finger is extended (tip is above its PIP joint)."""
        return landmarks[tip_id][1] < landmarks[pip_id][1]

    def _is_thumb_up(self, landmarks):
        """
        Check if the thumb is extended.
        Uses horizontal distance: thumb tip should be farther from wrist
        than the thumb IP joint along the X axis.
        """
        # Determine handedness by checking if wrist is left or right of middle MCP
        wrist_x = landmarks[self.WRIST][0]
        middle_mcp_x = landmarks[9][0]  # Middle finger MCP

        if wrist_x < middle_mcp_x:
            # Right hand (mirrored view): thumb tip should be to the LEFT of thumb IP
            return landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_IP][0]
        else:
            # Left hand (mirrored view): thumb tip should be to the RIGHT of thumb IP
            return landmarks[self.THUMB_TIP][0] > landmarks[self.THUMB_IP][0]

    def get_fingers_up(self, landmarks=None):
        """
        Determine which fingers are up.

        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky]
        """
        if landmarks is None:
            landmarks = self.pixel_landmarks
        if landmarks is None:
            return [False] * 5

        return [
            self._is_thumb_up(landmarks),
            self._is_finger_up(landmarks, self.INDEX_TIP, 5),      # INDEX MCP
            self._is_finger_up(landmarks, self.MIDDLE_TIP, 9),     # MIDDLE MCP
            self._is_finger_up(landmarks, self.RING_TIP, 13),      # RING MCP
            self._is_finger_up(landmarks, self.PINKY_TIP, 17),     # PINKY MCP
        ]

    def classify_gesture(self, landmarks=None):
        """
        Classify the current hand gesture.

        Returns:
            One of GESTURE_DRAW, GESTURE_HOVER, GESTURE_ERASE, GESTURE_CLEAR, GESTURE_NONE.
        """
        if landmarks is None:
            landmarks = self.pixel_landmarks
        if landmarks is None:
            self._fist_start_time = None
            self._clear_triggered = False
            self.gesture_history.append(self.GESTURE_NONE)
            self.gesture = self.GESTURE_NONE
            return self.GESTURE_NONE

        fingers = self.get_fingers_up(landmarks)
        # fingers = [thumb, index, middle, ring, pinky]
        _, index_up, middle_up, ring_up, pinky_up = fingers
        num_fingers_up = sum(fingers[1:])  # Exclude thumb for primary gesture detection

        # --- Fist detection (Clear): 0 non-thumb fingers up ---
        if num_fingers_up == 0:
            now = time.time()
            if self._fist_start_time is None:
                self._fist_start_time = now
                self._clear_triggered = False

            elapsed = now - self._fist_start_time
            if elapsed >= self.FIST_HOLD_DURATION and not self._clear_triggered:
                self._clear_triggered = True
                raw_gesture = self.GESTURE_CLEAR
            else:
                # Fist is held but not long enough yet — treat as no active gesture
                raw_gesture = self.GESTURE_NONE
        else:
            self._fist_start_time = None
            self._clear_triggered = False

            # --- Erase: Index + Middle + Ring up, Pinky down ---
            if index_up and middle_up and ring_up and not pinky_up:
                raw_gesture = self.GESTURE_ERASE
            # --- Hover: Middle up, Ring and Pinky down (allows Index to bend/pinch) ---
            elif middle_up and not ring_up and not pinky_up:
                raw_gesture = self.GESTURE_HOVER
            # --- Draw: Only Index up ---
            elif index_up and not middle_up and not ring_up and not pinky_up:
                raw_gesture = self.GESTURE_DRAW
            else:
                raw_gesture = self.GESTURE_NONE

        # If clear is triggered, bypass temporal filter to trigger instantly
        if raw_gesture == self.GESTURE_CLEAR:
            self.gesture_history.clear()
            self.gesture_history.append(self.GESTURE_CLEAR)
            self.gesture = self.GESTURE_CLEAR
            return self.GESTURE_CLEAR

        # Add raw gesture to history
        self.gesture_history.append(raw_gesture)

        # Apply temporal filtering: return majority vote
        most_common = Counter(self.gesture_history).most_common(1)[0][0]
        self.gesture = most_common
        return most_common

    def get_index_tip(self):
        """Return the (x, y) pixel position of the index finger tip."""
        if self.pixel_landmarks is None:
            return None
        return self.pixel_landmarks[self.INDEX_TIP]

    def get_pinch_distance(self):
        """
        Calculate the normalized distance between thumb tip and index tip.
        Normalized by palm length (wrist to middle MCP) to be distance-invariant.

        Returns:
            Normalized distance (float), or None if no hand detected.
        """
        if self.pixel_landmarks is None:
            return None
        thumb = self.pixel_landmarks[self.THUMB_TIP]
        index = self.pixel_landmarks[self.INDEX_TIP]
        raw_dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
        
        # Calculate palm length for normalization (wrist to middle finger MCP joint)
        wrist = self.pixel_landmarks[self.WRIST]
        mcp = self.pixel_landmarks[9]
        palm_len = math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1])
        
        if palm_len < 1e-5:
            return 0.5
            
        return raw_dist / palm_len

    def get_fist_progress(self):
        """
        Returns the progress (0.0 to 1.0) of the fist-hold timer.
        Useful for showing a visual indicator before clearing.
        """
        if self._fist_start_time is None:
            return 0.0
        elapsed = time.time() - self._fist_start_time
        return min(elapsed / self.FIST_HOLD_DURATION, 1.0)

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
